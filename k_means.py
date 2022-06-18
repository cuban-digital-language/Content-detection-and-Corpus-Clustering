from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from preprocessing import loads


def plot_results(inertials):
    x, y = zip(*[inertia for inertia in inertials])
    plt.plot(x, y, 'ro-', markersize=8, lw=2)
    plt.grid(True)
    plt.xlabel('Num Clusters')
    plt.ylabel('Inertia')
    plt.show()


def select_clusters(points, loops, max_iterations, init_cluster, tolerance, verbose=0):
    # Read data set
    inertia_clusters = list()
    for i in range(1, loops + 1, 1):
        # Object KMeans
        kmeans = KMeans(n_clusters=i, max_iter=max_iterations,
                        init=init_cluster, tol=tolerance, verbose=verbose)

        # Calculate Kmeans
        kmeans.fit(points)

        # Obtain inertia
        inertia_clusters.append([i, kmeans.inertia_])

    plot_results(inertia_clusters)


if __name__ == '__main__':
    v, _ = loads()
    select_clusters(v, 100, 10000, "k-means++", 0.0001, verbose=1)
