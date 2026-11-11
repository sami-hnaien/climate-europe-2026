"""Clustering of NUTS 2 regions on climate indicators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from climate_europe.utils.logging import get_logger


class ClusteringAnalysis:
    """Standardise, reduce (PCA), cluster and describe regions.

    Steps: scale -> PCA -> choose k (elbow + silhouette) -> K-Means and hierarchical
    clustering -> cluster profiles. Features and k range come from config.yaml.
    Keep in mind: neighbouring regions are not independent (spatial autocorrelation).

    ``indicators`` here is a region-level table (one row per region, index or column
    ``region_id``) holding the feature columns.
    """

    def __init__(self, config: dict[str, Any], features: list[str]) -> None:
        self.config = config
        self.features = features
        self.random_state = config["analysis"]["clustering"]["random_state"]
        self.logger = get_logger("analysis.clustering")
        self.scaler_ = None
        self.pca_ = None

    def _region_index(self, indicators: pd.DataFrame) -> pd.DataFrame:
        return indicators.set_index("region_id") if "region_id" in indicators.columns else indicators

    def prepare(self, indicators: Any) -> Any:
        """Select the features and standardise them (StandardScaler).

        Regions with a missing feature are dropped (and logged): K-Means cannot use them.
        """
        from sklearn.preprocessing import StandardScaler

        data = self._region_index(indicators)[self.features]
        missing = data.index[data.isna().any(axis=1)]
        if len(missing):
            self.logger.info("%d regions dropped (missing feature): %s", len(missing), list(missing))
        data = data.dropna()
        self.scaler_ = StandardScaler().fit(data)
        return pd.DataFrame(self.scaler_.transform(data), index=data.index, columns=self.features)

    def reduce(self, scaled: Any, n_components: int = 2) -> Any:
        """PCA projection, also used for the final scatter plot.

        The fitted PCA is kept in ``pca_`` (explained variance, loadings).
        """
        from sklearn.decomposition import PCA

        self.pca_ = PCA(n_components=n_components, random_state=self.random_state).fit(scaled)
        columns = [f"pc{i + 1}" for i in range(n_components)]
        return pd.DataFrame(self.pca_.transform(scaled), index=scaled.index, columns=columns)

    def loadings(self) -> pd.DataFrame:
        """Contribution of each feature to the principal components."""
        columns = [f"pc{i + 1}" for i in range(self.pca_.n_components_)]
        return pd.DataFrame(self.pca_.components_.T, index=self.features, columns=columns)

    def choose_k(self, scaled: Any) -> Any:
        """Inertia (elbow) and silhouette score for each k in the configured range."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import davies_bouldin_score, silhouette_score

        k_min, k_max = self.config["analysis"]["clustering"]["k_range"]
        rows = []
        for k in range(k_min, k_max + 1):
            model = KMeans(n_clusters=k, n_init=20, random_state=self.random_state).fit(scaled)
            rows.append(
                {
                    "k": k,
                    "inertia": model.inertia_,
                    "silhouette": silhouette_score(scaled, model.labels_),
                    "davies_bouldin": davies_bouldin_score(scaled, model.labels_),
                }
            )
        return pd.DataFrame(rows)

    def fit(self, scaled: Any, k: int, method: str = "kmeans") -> Any:
        """Return one cluster label per region (kmeans or hierarchical).

        Hierarchical clustering uses Ward linkage (same objective as K-Means: minimise
        within-cluster variance), which makes the two methods comparable. Labels are
        renumbered 1..k by decreasing mean of the first feature, so that they do not
        change arbitrarily between runs.
        """
        from sklearn.cluster import AgglomerativeClustering, KMeans

        if method == "kmeans":
            model = KMeans(n_clusters=k, n_init=50, random_state=self.random_state)
        elif method == "hierarchical":
            model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        else:
            raise ValueError(f"unknown method {method!r} (use 'kmeans' or 'hierarchical')")
        labels = pd.Series(model.fit_predict(scaled), index=scaled.index, name="cluster")
        order = scaled.iloc[:, 0].groupby(labels).mean().sort_values(ascending=False).index
        mapping = {old: new for new, old in enumerate(order, start=1)}
        return labels.map(mapping)

    def profiles(self, indicators: Any, labels: Any) -> Any:
        """Mean of each indicator per cluster, to name and interpret the groups."""
        data = self._region_index(indicators).loc[labels.index]
        numeric = data.select_dtypes(include=np.number)
        table = numeric.groupby(labels).mean()
        table.insert(0, "n_regions", labels.value_counts().sort_index())
        return table
