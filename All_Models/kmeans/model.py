import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from All_Models.base import BaseModel
from All_Models import register

@register("kmeans")
class KMeansClustering(BaseModel):
    """
    Simple K-means wrapper.
    fit(train_data) expects DataFrame/array with numeric columns.
    predict(data) returns (labels, centers).
    """

    def __init__(self, n_clusters=3, random_state=42, scale=True):
        self.n_clusters = int(n_clusters)
        self.random_state = int(random_state)
        self.scale = bool(scale)
        self.scaler = None
        # n_init="auto" avoids FutureWarnings on new sklearn
        self.model = KMeans(n_clusters=self.n_clusters, n_init="auto",
                            random_state=self.random_state)

    def _to_array(self, X):
        if isinstance(X, pd.DataFrame):
            return X.values
        return np.asarray(X)

    def fit(self, train_data, **kwargs):
        X = self._to_array(train_data)
        if self.scale:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        self.model.set_params(n_clusters=self.n_clusters,
                              random_state=self.random_state)
        self.model.fit(X)
        return self

    def predict(self, data, **kwargs):
        X = self._to_array(data)
        if self.scale and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        labels = self.model.predict(X_scaled)
        centers = self.model.cluster_centers_
        if self.scale and self.scaler is not None:
            centers = self.scaler.inverse_transform(centers)
        return labels, centers
