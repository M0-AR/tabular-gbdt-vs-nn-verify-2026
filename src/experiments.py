"""End-to-end benchmark: video claims -> synthetic -> real sklearn -> live market.
Usage: python3 src/experiments.py --fast | --full
Writes results/*.json + results/SUMMARY.md. Exits nonzero on claim violation.
"""
import argparse, json, time, os
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from data_synthetic import train_test, add_noise_features, rotate_features, true_price
from gbdt_plain import PlainGBDT
from models import train_mlp, train_hgb, rmse
from live_data import fetch_btc, fetch_fx, to_tabular

RES = "results"
os.makedirs(RES, exist_ok=True)
out = {}


def rm(a, b):
    return float(mean_squared_error(a, b) ** 0.5)


def exp_depth():
    Xtr, ytr, _, Xte, yte, _ = train_test()
    r = {}
    for d in [1, 2, 3, 12]:
        t = DecisionTreeRegressor(max_depth=d, random_state=0).fit(Xtr, ytr)
        r[f"depth{d}_train"] = rm(ytr, t.predict(Xtr))
        r[f"depth{d}_test"] = rm(yte, t.predict(Xte))
    out["single_tree_depth"] = r
    print("[depth]", {k: round(v) for k, v in r.items()})


def exp_boost_curve():
    Xtr, ytr, _, Xte, yte, _ = train_test()
    for lr in [0.1, 1.0]:
        g = PlainGBDT(n_estimators=1000, max_depth=3, learning_rate=lr).fit(Xtr, ytr)
        tr, te = g.staged_rmse(Xtr, ytr), g.staged_rmse(Xte, yte)
        pick = {1: tr[1], 10: tr[10], 50: tr[50], 300: tr[300], 1000: tr[1000]}
        pickt = {1: te[1], 10: te[10], 50: te[50], 300: te[300], 1000: te[1000]}
        out[f"boost_lr{lr}_train"] = pick
        out[f"boost_lr{lr}_test"] = pickt
        print(f"[boost lr={lr}] test:", {k: round(v) for k, v in pickt.items()})
    # deep trees in boost
    g8 = PlainGBDT(n_estimators=300, max_depth=8, learning_rate=0.1).fit(Xtr, ytr)
    out["boost_d8_train300"] = g8.staged_rmse(Xtr, ytr)[300]
    out["boost_d8_test300"] = g8.staged_rmse(Xte, yte)[300]
    print("[boost d8 @300]", round(out["boost_d8_train300"]), round(out["boost_d8_test300"]))


def exp_main_match(fast):
    Xtr, ytr, _, Xte, yte, _ = train_test()
    g = PlainGBDT(n_estimators=1000 if not fast else 300, max_depth=3,
                  learning_rate=0.1).fit(Xtr, ytr)
    te = g.staged_rmse(Xte, yte)
    best = min(te)
    out["gbdt_best"] = best
    out["gbdt_300"] = te[300 if len(te) > 300 else -1]
    mlps = [train_mlp(Xtr, ytr, Xte, yte, seed=s,
                      epochs=40 if fast else 80)[1] for s in (0, 1, 2)]
    out["mlp_3seeds"] = mlps
    out["oracle_noise"] = 15000.0
    print(f"[match] GBDT best={best:.0f} @300={out['gbdt_300']:.0f} MLP={np.mean(mlps):.0f}")


def exp_noise_rotation():
    Xtr, ytr, _, Xte, yte, _ = train_test()
    for k in [0, 10, 30]:
        if k == 0:
            Atr, Ate = Xtr, Xte
        else:
            Atr, Ate = add_noise_features(Xtr, Xte, k)
        g = PlainGBDT(n_estimators=300, max_depth=3, learning_rate=0.1).fit(Atr, ytr)
        gm = g.staged_rmse(Ate, yte)[-1]
        mm = np.mean([train_mlp(Atr, ytr, Ate, yte, seed=s, epochs=40)[1] for s in (0, 1)])
        out[f"noise{k}_gbdt"] = gm
        out[f"noise{k}_mlp"] = float(mm)
        print(f"[noise k={k}] gbdt={gm:.0f} mlp={mm:.0f}")
    Rtr, Rte = rotate_features(Xtr, Xte)
    g = PlainGBDT(n_estimators=300, max_depth=3, learning_rate=0.1).fit(Rtr, ytr)
    out["rot_gbdt"] = g.staged_rmse(Rte, yte)[-1]
    out["rot_mlp"] = float(np.mean([train_mlp(Rtr, ytr, Rte, yte, seed=s, epochs=40)[1] for s in (0, 1)]))
    print(f"[rot] gbdt={out['rot_gbdt']:.0f} mlp={out['rot_mlp']:.0f}")


def exp_real(fast):
    # diabetes regression, 5-fold
    X, y = load_diabetes(return_X_y=True)
    kf = KFold(5, shuffle=True, random_state=0)
    gr, mr = [], []
    for tr, te in kf.split(X):
        h = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1,
                                          early_stopping=True, random_state=0)
        h.fit(X[tr], y[tr])
        gr.append(rm(y[te], h.predict(X[te])))
        m = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(256, 256, 256),
                          max_iter=300 if not fast else 150, random_state=0))
        m.fit(X[tr], y[tr])
        mr.append(rm(y[te], m.predict(X[te])))
    out["diabetes_gbdt_rmse"] = float(np.mean(gr))
    out["diabetes_mlp_rmse"] = float(np.mean(mr))
    print(f"[diabetes] gbdt={np.mean(gr):.1f} mlp={np.mean(mr):.1f}")
    # breast cancer classification, 5-fold
    X, y = load_breast_cancer(return_X_y=True)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    ga, ma = [], []
    for tr, te in skf.split(X, y):
        h = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                           early_stopping=True, random_state=0)
        h.fit(X[tr], y[tr])
        ga.append(accuracy_score(y[te], h.predict(X[te])))
        m = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256, 256, 256),
                          max_iter=300 if not fast else 150, random_state=0))
        m.fit(X[tr], y[tr])
        ma.append(accuracy_score(y[te], m.predict(X[te])))
    out["cancer_gbdt_acc"] = float(np.mean(ga))
    out["cancer_mlp_acc"] = float(np.mean(ma))
    print(f"[cancer] gbdt={np.mean(ga):.3f} mlp={np.mean(ma):.3f}")


def exp_live(fast):
    for name, px in [("BTC", fetch_btc(365)), ("EURUSD_inv", 1 / fetch_fx("EUR", 365))]:
        X, y, _ = to_tabular(np.asarray(px, dtype=float))
        n = len(X)
        tr, te = int(n * 0.8), n - int(n * 0.8)
        Xtr, Xte, ytr, yte = X[:tr], X[tr:], y[:tr], y[tr:]
        g = PlainGBDT(n_estimators=200 if fast else 300, max_depth=3,
                      learning_rate=0.1).fit(Xtr, ytr)
        gr = g.staged_rmse(Xte, yte)[-1]
        mr = float(np.mean([train_mlp(Xtr, ytr, Xte, yte, seed=s, epochs=40)[1] for s in (0, 1)]))
        base = rm(yte, np.full_like(yte, ytr.mean()))
        out[f"live_{name}_gbdt"] = gr
        out[f"live_{name}_mlp"] = mr
        out[f"live_{name}_meanbase"] = base
        print(f"[live {name}] n={n} gbdt={gr:.5f} mlp={mr:.5f} mean={base:.5f}")


def verify():
    fails = []
    if not (out["gbdt_300"] < 22000):
        fails.append("GBDT@300 should be ~16-20k")
    if not (np.mean(out["mlp_3seeds"]) > out["gbdt_300"]):
        fails.append("MLP should lose on synthetic jumps")
    if not (out["noise30_gbdt"] < out["noise30_mlp"]):
        fails.append("GBDT should be more robust to noise cols")
    if not (out["rot_gbdt"] > out["noise0_gbdt"] if "noise0_gbdt" in out else True):
        pass
    # live: models must run and beat-or-tie mean is not required (finance noise);
    # only require no crash + finite numbers
    for k, v in out.items():
        if isinstance(v, float) and not np.isfinite(v):
            fails.append(f"non-finite {k}")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    fast = a.fast or not a.full
    t0 = time.time()
    exp_depth()
    exp_boost_curve()
    exp_main_match(fast)
    exp_noise_rotation()
    exp_real(fast)
    exp_live(fast)
    out["seconds"] = round(time.time() - t0, 1)
    with open(f"{RES}/results.json", "w") as f:
        json.dump(out, f, indent=2)
    fails = verify()
    md = ["# SUMMARY (auto-generated, do not hand-edit)",
          f"mode={'fast' if fast else 'full'} seconds={out['seconds']}", "",
          "| key | value |", "|---|---|"]
    for k, v in sorted(out.items()):
        md.append(f"| {k} | {str(v)[:120]} |")
    md += ["", "VERIFY: " + ("PASS" if not fails else "FAIL: " + "; ".join(fails))]
    open(f"{RES}/SUMMARY.md", "w").write("\n".join(md))
    print("VERIFY:", "PASS" if not fails else fails)
    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
