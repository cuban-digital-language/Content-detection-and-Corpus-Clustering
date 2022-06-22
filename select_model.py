from preprocessing import loads
from sklearn.model_selection import ShuffleSplit
from sklearn.cluster import AgglomerativeClustering
import pickle
from agglomerative import saveData, loadData
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np

from sklearn.model_selection import train_test_split
from plot_learning_curve import plot_learning_curve
import matplotlib.pyplot as plt


def training_cluster():
    # files = files_list('tokens')
    # print("total", len(files))
    # split = ShuffleSplit(n_splits=1, test_size=.7)
    # files = [files[index] for index in next(split.split(files))[0]]
    # print('split', len(files))

    print('load vectors')
    vectors, documents = loads()
    print('split data')
    split = ShuffleSplit(n_splits=1, test_size=.8)

    vectors = [vectors[index] for index in next(split.split(vectors))[0]]
    # print(vectors)
    # # save(v, d)
    # view_points(vectors)

    cluster = AgglomerativeClustering(
        distance_threshold=0.3,
        n_clusters=None,
        affinity='cosine',
        linkage='single'
    )

    print("##### Agglomerative ########")
    Y = cluster.fit_predict(vectors)
    print(f"n cluster", cluster.n_clusters_)

    serialize_cluster = pickle.dumps(cluster)
    saveData(serialize_cluster, 'cluster', 'b')
    saveData(Y, 'labels', 'b')
    save2(vectors)


def save2(vectors):
    np.save('vectors/vectors.npy', vectors)


def load2():
    return np.load('vectors/vectors.npy')


def training_models():
    cluster = pickle.loads(loadData('cluster', 'b'))
    vectors = load2()
    labels = pickle.loads(loadData('labels', 'b'))
    # KNN
    knn = KNeighborsClassifier(
        2 * cluster.n_clusters_ + 1, metric='cosine', n_jobs=5)

    print("##### KNN ########")
    knn.fit(vectors, labels)

    # SVM
    svm = SVC(2 * cluster.n_clusters_ + 1, metric='cosine', n_jobs=5)

    print("##### SVM #####")
    svm.fit(vectors, labels)

    # save models
    # save knn
    serialize_knn = pickle.dumps(knn)
    saveData(serialize_knn, 'knn_model', 'b')
    # save svm
    serialize_svm = pickle.dumps(svm)
    saveData(serialize_svm, 'svm_model', 'b')


def grafic_models():
    knn = pickle.loads(loadData('knn_model', 'b'))
    svm = pickle.loads(loadData('svm_model', 'b'))
    vectors = load2()
    labels = pickle.loads(loadData('labels', 'b'))
    xtrain, xtest, ytrain, ytest = train_test_split(vectors, labels, train_size=0.6)

    # train and test
    # knn
    knn.fit(xtrain, ytrain)
    knn.score(xtest, ytest)

    svm.fit(xtrain, ytrain)
    svm.score(xtest, ytest)

    # ver si esto se puede quitar
    cv = ShuffleSplit(n_splits=1, test_size=.7)

    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    plot_learning_curve(
        knn, "", vectors, labels, axes=axes[:, 0], ylim=(0.7, 1.01), cv=cv, n_jobs=4
    )


training_cluster()
