"""Synthetic housing generator — faithful to the video's true rule."""
import numpy as np

FEATURES = ["size", "rooms", "dist_sea", "age", "garage", "bathrooms"]


def true_price(X: np.ndarray) -> np.ndarray:
    size, rooms, dist, age, garage, bath = (X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5])
    p = 60_000 + 2_000 * size + 10_000 * rooms
    p = p + np.where(dist < 2.0, 120_000, 0.0)
    p = p + np.where(dist > 20.0, -40_000, 0.0)
    p = p + 50_000 * garage
    p = p + 5_000 * bath
    p = p + np.where(age > 50.0, -60_000, 0.0)
    return p


def sample_houses(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    size = rng.uniform(50, 300, n)
    rooms = rng.integers(1, 7, n).astype(float)
    dist = rng.uniform(0, 30, n)
    age = rng.uniform(0, 100, n)
    garage = rng.integers(0, 2, n).astype(float)
    bath = rng.integers(1, 4, n).astype(float)
    X = np.column_stack([size, rooms, dist, age, garage, bath])
    clean = true_price(X)
    noise = rng.normal(0, 15_000, n)
    return X, clean + noise, clean


def train_test(seed: int = 0, n_train: int = 2000, n_test: int = 1000):
    X, y, clean = sample_houses(n_train + n_test, seed=seed)
    return (X[:n_train], y[:n_train], clean[:n_train],
            X[n_train:], y[n_train:], clean[n_train:])


def add_noise_features(Xtr, Xte, k: int, seed: int = 1):
    rng = np.random.default_rng(seed)
    Ntr = rng.normal(0, 1, (Xtr.shape[0], k))
    Nte = rng.normal(0, 1, (Xte.shape[0], k))
    return np.hstack([Xtr, Ntr]), np.hstack([Xte, Nte])


def rotate_features(Xtr, Xte, seed: int = 2):
    """Random orthogonal rotation (QR of Gaussian), applied on standardized features.
    Preserves all information; destroys axis alignment. Mean/std from train only."""
    rng = np.random.default_rng(seed)
    d = Xtr.shape[1]
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    return ((Xtr - mu) / sd) @ Q, ((Xte - mu) / sd) @ Q
