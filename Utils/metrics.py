import numpy as np
from sklearn.metrics import silhouette_score

def safe_silhouette(X, labels):
    if len(np.unique(labels)) < 2:
        return None
    try:
        return silhouette_score(X, labels)
    except Exception:
        return None
