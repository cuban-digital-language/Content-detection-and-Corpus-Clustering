import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import sys
from sklearn.model_selection import ShuffleSplit

from sklearn.neighbors import KNeighborsClassifier
from itertools import cycle, islice
import numpy as np

from preprocessing import loads, view_points


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    return dendrogram(linkage_matrix, **kwargs)


def agglomerative(vectors):
    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(
        distance_threshold=0, n_clusters=None, affinity='cosine', linkage='single')
    print("### Training ###")
    model = model.fit(vectors)
    plt.title("Hierarchical Clustering Dendrogram")
    # plot the top three levels of the dendrogram
    print("### PRINT ###")
    data = plot_dendrogram(model)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()


def fit_cluster_class(vectors):
    cluster = AgglomerativeClustering(
        distance_threshold=0.3,
        n_clusters=None,
        affinity='cosine',
        linkage='single'
    )

    print("##### Agglomerative ########")
    Y = cluster.fit_predict(vectors)
    print(f"n cluster", cluster.n_clusters_)

    model = KNeighborsClassifier(
        2 * cluster.n_clusters_ + 1, metric='cosine', n_jobs=5)

    print("##### KNN ########")
    model.fit(vectors, Y)

    return cluster, model


def colors(model):

    y_pred = model.labels_.astype(int)

    color = np.array(
        list(
            islice(
                cycle(
                    [
                        "#377eb8",
                        "#ff7f00",
                        "#4daf4a",
                        "#f781bf",
                        "#a65628",
                        "#984ea3",
                        "#999999",
                        "#e41a1c",
                        "#dede00",
                    ]
                ),
                int(max(y_pred) + 1),
            )
        )
    )

    return color, y_pred


def saveData(data: str, path: str, type: str = ''):
    with open(os.path.join(f'{os.getcwd()}/result', path), f'w{type}') as f:
        f.write(data)


def loadData(path: str, type: str = ''):
    with open(os.path.join(f'{os.getcwd()}/result', path), f'r{type}') as f:
        text = f.read()
        f.close()
    return text


if __name__ == '__main__':
    v, _ = loads()
    split = ShuffleSplit(n_splits=1, test_size=.9)
    v = [v[index] for index in next(split.split(v))[0]]
    cluster, knn = fit_cluster_class(v)
    print("##### SAVE ########")
    serialize_cluster = pickle.dumps(knn)
    saveData(serialize_cluster, 'cluster', 'b')

    print("##### PLOT ########")
    color, y = colors(cluster)
    view_points(v, color[y])
