# --- make repo root importable when run as a script ---
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------------------------




import argparse, pandas as pd
from pathlib import Path
from All_Models import get_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="kmeans")
    ap.add_argument("--data", default=str(Path("Data/kmeans_sample.csv")))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--scale", action="store_true")
    args = ap.parse_args()

    if args.model == "kmeans":
        df = pd.read_csv(args.data)[["x", "y"]]
        K = get_model("kmeans")
        mdl = K(n_clusters=args.k, scale=args.scale).fit(df)
        labels, centers = mdl.predict(df)
        print("centers:\n", centers)
        print("counts:\n", pd.Series(labels).value_counts())

if __name__ == "__main__":
    main()
