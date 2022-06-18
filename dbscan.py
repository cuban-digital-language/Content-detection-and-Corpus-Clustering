from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import DistanceMetric

from preprocessing import pb, loads


def metric(vectors):
    vector = preprocessing.normalize(vectors)

    dist = DistanceMetric.get_metric("euclidean")
    matsim = dist.pairwise(vector)
    minPits = 5
    A = kneighbors_graph(vector, minPits, include_self=False).toarray()

    seq = []
    _len_ = len(vector)
    bar = pb(_len_ * _len_, f' {_len_ * _len_} matrix ')
    for i, s in enumerate(vector):
        for j in range(len(vector)):
            if A[i][j] != 0:
                seq.append(matsim[i][j])

            bar.update(i+1)
    bar.finish()

    seq.sort()
    plt.plot(seq)
    plt.show()


if __name__ == '__main__':
    v, _ = loads()
    metric(v)
