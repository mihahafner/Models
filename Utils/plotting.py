import pandas as pd
import plotly.express as px

def scatter_2d(df: pd.DataFrame, x="x", y="y", color="cluster", centers=None, title="Clusters"):
    fig = px.scatter(df, x=x, y=y, color=color, title=title, opacity=0.8)
    if centers is not None:
        cx, cy = centers[:, 0], centers[:, 1]
        fig.add_scatter(x=cx, y=cy, mode="markers",
                        marker_symbol="x", marker_size=14, name="centers")
    return fig
