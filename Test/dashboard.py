# Models/Test/dashboard.py
# Streamlit K-Means demo using your project structure

# -- make repo root importable when run as a script --
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]  # .../Models
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------------------

import streamlit as st
import pandas as pd
from All_Models import get_model  # returns the KMeans class in your project

st.set_page_config(page_title="K-Means Demo", layout="centered")
st.title("K-Means Demo")

# --- controls ---
data_path = st.text_input(
    "CSV path (expects columns x,y):",
    value=str(ROOT / "Data" / "kmeans_sample.csv")
)
k = st.number_input("Number of clusters (k)", min_value=1, max_value=10, value=3, step=1)
scale = st.checkbox("Scale features", value=True)

# --- load data ---
try:
    df = pd.read_csv(data_path)[["x", "y"]]
except Exception as e:
    st.error(f"Failed to read data from {data_path}\n{e}")
    st.stop()

st.write("First rows:", df.head())

# --- build & fit model (instantiate the class, then fit) ---
K = get_model("kmeans")                 # class
mdl = K(n_clusters=k, scale=scale).fit(df)  # instance, now fitted

# --- predict and get centers ---
labels, centers = mdl.predict(df)       # your project returns both

st.subheader("Cluster centers")
st.write(pd.DataFrame(centers, columns=["x", "y"]))

st.subheader("Counts per cluster")
st.write(pd.Series(labels).value_counts().sort_index())

# --- plot (Streamlit built-in chart) ---
st.subheader("Scatter plot")
plot_df = df.copy()
plot_df["cluster"] = labels
st.scatter_chart(plot_df, x="x", y="y", color="cluster")

#streamlit run Test/dashboard.py
