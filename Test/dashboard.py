# --- Ensure project root is on sys.path (Streamlit often changes cwd) ---
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
from pathlib import Path

from All_Models import get_model
from Utils.metrics import safe_silhouette
from Utils.plotting import scatter_2d

st.set_page_config(page_title="Models Dashboard", layout="wide")
st.title("🧪 Models Dashboard")

# Sidebar
model_name = st.sidebar.selectbox("Choose model", ["kmeans", "transformer"], index=0)
st.sidebar.markdown("---")

# K-means data source
df = None
if model_name == "kmeans":
    st.sidebar.subheader("Data source")
    choice = st.sidebar.radio("Load dataset", ["Sample (Data/kmeans_sample.csv)", "Upload CSV"], index=0)

    if choice.startswith("Sample"):
        sample_path = ROOT / "Data" / "kmeans_sample.csv"
        if not sample_path.exists():
            st.error("Sample CSV not found. Run:  python -m Data.make_kmeans_sample")
            st.stop()
        df = pd.read_csv(sample_path)
        st.caption(f"Loaded sample: {sample_path.name} — shape={df.shape}")
    else:
        up = st.sidebar.file_uploader("Upload CSV with columns x,y", type=["csv"])
        if up is not None:
            df = pd.read_csv(up)

    st.sidebar.markdown("---")
    st.sidebar.subheader("K-means parameters")
    k = st.sidebar.slider("n_clusters (k)", 2, 10, 3)
    scale = st.sidebar.checkbox("Standardize features", value=True)
    random_state = st.sidebar.number_input("random_state", value=42)

    if df is not None:
        X = df[["x", "y"]].copy()

        KMeansClass = get_model("kmeans")
        model = KMeansClass(n_clusters=k, scale=scale, random_state=int(random_state))
        model.fit(X)
        labels, centers = model.predict(X)

        plot_df = X.copy()
        plot_df["cluster"] = labels.astype(int)
        sil = safe_silhouette(X.values, labels)

        c1, c2 = st.columns([3, 2])
        with c1:
            st.subheader("Clusters")
            fig = scatter_2d(plot_df, x="x", y="y", color="cluster",
                             centers=centers, title=f"K-means (k={k})")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Details")
            st.write("Cluster centers:", pd.DataFrame(centers, columns=["x", "y"]))
            st.metric("Silhouette (higher is better)", f"{sil:.3f}" if sil is not None else "n/a")
            st.write("Counts by cluster:", plot_df["cluster"].value_counts().rename_axis("cluster").to_frame("count"))

elif model_name == "transformer":
    st.info("Transformer dashboard coming next. Use your training script for now. 👍")
