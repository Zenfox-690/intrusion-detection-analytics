"""Model definitions and feature transformations.

Implements K-Means clustering (K=5) for centroid extraction, Euclidean distance
transformation into a 5-dimensional feature space, and Naïve Bayes classifiers
(both proposed K-Means+NB and baseline NB) following the reference paper methodology.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

DISTANCE_FEATURE_NAMES = [
    'dist_cluster_0',
    'dist_cluster_1',
    'dist_cluster_2',
    'dist_cluster_3',
    'dist_cluster_4'
]


def fit_kmeans(
    X_train: np.ndarray,
    n_clusters: int = 5,
    random_state: int = 42
) -> KMeans:
    """Fit K-Means on the preprocessed training dataset.

    Note: K-Means cluster IDs (0 to 4) are mathematical partition labels and
    do NOT automatically correspond to semantic attack class names.
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=10,
        max_iter=300,
        random_state=random_state
    )
    kmeans.fit(X_train)
    return kmeans


def transform_distance_features(kmeans: KMeans, X: np.ndarray) -> np.ndarray:
    """Transform records into a 5-dimensional distance representation.

    For each record x and each learned centroid c_k (k=0..4), computes the
    Euclidean distance ||x - c_k||_2 using vectorized scikit-learn transform.
    """
    return kmeans.transform(X)


def get_cluster_distribution(kmeans: KMeans, X: np.ndarray) -> Dict[int, int]:
    """Calculate the distribution of nearest clusters for a given dataset."""
    labels = kmeans.predict(X)
    unique, counts = np.unique(labels, return_counts=True)
    return {int(cluster_id): int(count) for cluster_id, count in zip(unique, counts)}


def train_nb_kmeans_model(X_train_dist: np.ndarray, y_train: np.ndarray) -> GaussianNB:
    """Train the proposed Gaussian Naïve Bayes classifier on 5D distance features."""
    nb_model = GaussianNB()
    nb_model.fit(X_train_dist, y_train)
    return nb_model


def train_baseline_nb_model(X_train: np.ndarray, y_train: np.ndarray) -> GaussianNB:
    """Train baseline Gaussian Naïve Bayes on full preprocessed feature space."""
    nb_baseline = GaussianNB()
    nb_baseline.fit(X_train, y_train)
    return nb_baseline


def train_decision_tree_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42
) -> DecisionTreeClassifier:
    """Train a baseline Decision Tree classifier for lightweight comparison."""
    dt = DecisionTreeClassifier(max_depth=12, random_state=random_state)
    dt.fit(X_train, y_train)
    return dt
