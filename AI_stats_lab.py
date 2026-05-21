import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans


# ============================================================
# Question 1
# ============================================================

def load_iris_unlabeled(feature_indices=(0, 1)):
    data = load_iris()

    X = data.data[:, feature_indices]
    feature_names = [data.feature_names[i] for i in feature_indices]

    return {
        "X": X,
        "feature_names": feature_names
    }


def standardize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    std_safe = np.where(std == 0, 1.0, std)

    X_scaled = (X - mean) / std_safe

    return {
        "X_scaled": X_scaled,
        "mean": mean,
        "std": std_safe
    }


def fit_kmeans(X, K, random_state=0, n_init=10):
    model = KMeans(n_clusters=K, random_state=random_state, n_init=n_init)
    model.fit(X)

    return {
        "centroids": model.cluster_centers_,
        "labels": model.labels_,
        "objective": model.inertia_,
        "n_iter": model.n_iter_
    }


def compute_kmeans_objective(X, centroids, labels):
    total = 0.0

    for i in range(len(X)):
        c = centroids[labels[i]]
        diff = X[i] - c
        total += np.sum(diff ** 2)

    return total


# ============================================================
# Question 2
# ============================================================

def evaluate_k_values(X, k_values, random_state=0, n_init=10):
    objectives = []
    improvements = []

    prev_obj = None

    for i, k in enumerate(k_values):
        result = fit_kmeans(X, k, random_state, n_init)
        obj = result["objective"]

        objectives.append(obj)

        if i == 0:
            improvements.append(0.0)
        else:
            improvement = (prev_obj - obj) / prev_obj
            improvements.append(improvement)

        prev_obj = obj

    return {
        "k_values": k_values,
        "objectives": objectives,
        "relative_improvements": improvements
    }


def choose_elbow_k(k_values, objectives):
    if len(k_values) < 3:
        return k_values[0]

    x1, y1 = k_values[0], objectives[0]
    x2, y2 = k_values[-1], objectives[-1]

    distances = []

    for i in range(len(k_values)):
        x0 = k_values[i]
        y0 = objectives[i]

        numerator = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
        denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

        dist = numerator / denominator if denominator != 0 else 0
        distances.append(dist)

    max_index = np.argmax(distances)
    return k_values[max_index]


def cluster_size_summary(labels, K):
    result = {}

    for k in range(K):
        result[k] = 0

    for label in labels:
        result[label] += 1

    return result


def identify_outliers_by_distance(X, centroids, labels, top_n=5):
    distances = []

    for i in range(len(X)):
        c = centroids[labels[i]]
        diff = X[i] - c
        dist = np.sum(diff ** 2)
        distances.append(dist)

    distances = np.array(distances)

    sorted_indices = np.argsort(-distances)  # descending

    top_indices = sorted_indices[:top_n]
    top_distances = distances[top_indices]

    return {
        "indices": top_indices,
        "distances": top_distances
    }


def diagnose_clustering_fit(K, elbow_k):
    if K < elbow_k:
        return "underfitting"
    elif K == elbow_k:
        return "good_fit"
    else:
        return "overfitting"


# ============================================================
# Question 3
# ============================================================

def plot_unlabeled_data(X, feature_names=None, title="Unlabeled Data"):
    fig, ax = plt.subplots()

    ax.scatter(X[:, 0], X[:, 1])

    ax.set_title(title)

    if feature_names:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])

    return fig, ax


def plot_kmeans_clusters(X, labels, centroids, feature_names=None, title="K-Means Clusters"):
    fig, ax = plt.subplots()

    ax.scatter(X[:, 0], X[:, 1], c=labels)
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200)

    ax.set_title(title)

    if feature_names:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])

    return fig, ax


def plot_elbow_curve(k_values, objectives, title="Elbow Method"):
    fig, ax = plt.subplots()

    ax.plot(k_values, objectives, marker='o')

    ax.set_title(title)
    ax.set_xlabel("Number of clusters K")
    ax.set_ylabel("Objective value")

    return fig, ax


if __name__ == "__main__":
    print("Implement all required functions.")
