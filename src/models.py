"""Model wrappers: sklearn GBDT + PyTorch MLP (3x256) with 3 seeds."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_squared_error


def rmse(a, b):
    return float(mean_squared_error(a, b) ** 0.5)


class MLPReg(nn.Module):
    def __init__(self, d, width=256, depth=3):
        super().__init__()
        layers, p = [], d
        for _ in range(depth):
            layers += [nn.Linear(p, width), nn.ReLU()]
            p = width
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(Xtr, ytr, Xte, yte, seed=0, epochs=60, width=256, depth=3, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    ym, ys = ytr.mean(), ytr.std() + 1e-9
    Xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    yt = torch.tensor((ytr - ym) / ys, dtype=torch.float32)
    Xe = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
    m = MLPReg(Xtr.shape[1], width, depth)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    loss = nn.MSELoss()
    m.train()
    for _ in range(epochs):
        opt.zero_grad()
        l = loss(m(Xt), yt)
        l.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        pred = m(Xe).numpy() * ys + ym
    return pred, rmse(yte, pred)


def train_hgb(Xtr, ytr, Xte, yte, seed=0, task="reg"):
    if task == "reg":
        h = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1,
                                          max_depth=5, early_stopping=True, random_state=seed)
        h.fit(Xtr, ytr)
        return h.predict(Xte), rmse(yte, h.predict(Xte))
    h = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                       max_depth=5, early_stopping=True, random_state=seed)
    h.fit(Xtr, ytr)
    return h.predict(Xte), float((h.predict(Xte) == yte).mean())
