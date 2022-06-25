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
from sklearn import tree


def training_cluster():
    print('load vectors')
    vectors, documents = loads()
    print('split data')
    split = ShuffleSplit(n_splits=1, test_size=.92)

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
    print('passs')
    serialize_cluster = pickle.dumps(cluster)
    saveData(serialize_cluster, 'cluster', 'b')
    serialize_labels = pickle.dumps(Y)
    saveData(serialize_labels, 'labels', 'b')
    save2(vectors)
    print('end cluster training')


def save2(vectors):
    np.save('vectors/vectors.npy', vectors)


def load2():
    return np.load('vectors/vectors.npy')


def training_models():
    cluster = pickle.loads(loadData('cluster', 'b'))
    vectors = load2()
    labels = pickle.loads(loadData('labels', 'b'))
    # KNN
    print(cluster.n_clusters_)
    knn = KNeighborsClassifier(
        cluster.n_clusters_, metric='cosine', n_jobs=5)

    print("##### KNN ########")
    knn.fit(vectors, labels)

    # SVM
    svm = SVC()

    print("##### SVM #####")
    svm.fit(vectors, labels)

    print("###### Descicion Tree")
    clf = tree.DecisionTreeClassifier(random_state=0)
    clf.fit(vectors, labels)

    # save models
    # save knn
    serialize_knn = pickle.dumps(knn)
    saveData(serialize_knn, 'knn_model', 'b')
    # save svm
    serialize_svm = pickle.dumps(svm)
    saveData(serialize_svm, 'svm_model', 'b')

    # save DecisionTree
    serialize_clf = pickle.dumps(clf)
    saveData(serialize_clf, 'clf_model', 'b')


def graphic_models():
    # load models
    # knn = pickle.loads(loadData('knn_model', 'b'))
    # svm = pickle.loads(loadData('svm_model', 'b'))
    # clf = pickle.loads(loadData('clf_model', 'b'))

    cluster = pickle.loads(loadData('cluster', 'b'))
    # models
    knn = KNeighborsClassifier(
        cluster.n_clusters_, metric='cosine', n_jobs=5)
    svm = SVC(decision_function_shape='ovr', random_state=False)
    clf = tree.DecisionTreeClassifier(random_state=0)

    # load training vectors and labels
    vectors = load2()
    labels = pickle.loads(loadData('labels', 'b'))

    X, y = vectors, labels

    # title = "Learning Curves (KNN)"
    # # Cross validation with 50 iterations to get smoother mean test and train
    # # score curves, each time with 20% data randomly selected as a validation set.
    # cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
    #
    # estimator = knn
    # print('graphic learning curve knn')
    # plot_learning_curve(
    #     estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4
    # )
    # plt.show()

    title = "Learning Curves (SVM)"
    # Cross validation with 50 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    print('graphic learning curve svm')
    estimator = svm
    plot_learning_curve(
        estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4
    )
    plt.show()

    title = "Learning Curves (CLF)"
    # Cross validation with 50 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=1, test_size=0.5, random_state=0)

    estimator = clf
    print('graphic learning curve clf')
    plot_learning_curve(
        estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4
    )

    plt.show()


def Model_tests(model, name: str = ''):
    # load training vectors and labels
    vectors = load2()
    labels = pickle.loads(loadData('labels', 'b'))

    X, y = vectors, labels

    # graphic learning curve
    title = "Learning Curves (KNN)"
    cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

    estimator = model
    print(f'graphic learning curve {name}')
    plot_learning_curve(
        estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4
    )
    plt.show()

    print('start split data')
    xtrain, xtest, ytrain, ytest = train_test_split(vectors, labels, train_size=0.3)

    # train and test
    # knn
    print(f'fit {name}')
    model.fit(xtrain, ytrain)
    print(model.score(xtest, ytest))


def AllModelsTest():
    cluster = pickle.loads(loadData('cluster', 'b'))

    knn = KNeighborsClassifier(
        cluster.n_clusters_, metric='cosine', n_jobs=5)
    svm = SVC(random_state=0)
    clf = tree.DecisionTreeClassifier(random_state=0)

    Model_tests(knn, 'knn')
    Model_tests(svm, 'svm')
    Model_tests(clf, 'DesicionTree')


AllModelsTest()
