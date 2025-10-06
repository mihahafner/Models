import numpy as np
import pandas as pd
from pathlib import Path

def main():
    rng = np.random.default_rng(7)

    centers = np.array([[0, 0], [5, 5], [-5, 5]], dtype=float)
    covs = [
        np.array([[0.6, 0.1],[0.1, 0.6]]),
        np.array([[0.8,-0.2],[-0.2,0.8]]),
        np.array([[0.5, 0.0],[0.0, 0.5]])
    ]

    n_per = 200
    pts, lab = [], []
    for k, (mu, cov) in enumerate(zip(centers, covs)):
        p = rng.multivariate_normal(mu, cov, size=n_per)
        pts.append(p)
        lab.append(np.full(n_per, k, int))

    X = np.vstack(pts)
    y = np.concatenate(lab)
    df = pd.DataFrame(X, columns=["x", "y"])
    df["true_label"] = y

    out = Path(__file__).resolve().parent / "kmeans_sample.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out} (shape={df.shape})")

if __name__ == "__main__":
    main()
