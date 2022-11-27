from sklearn.neighbors import KNeighborsClassifier


def knn_prediction(x_train, y_train, x_test, k=3):
    """
    :param x_train:
    :param y_train:
    :param x_test:
    :param k:
    :return:
    """
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(x_train, y_train)
    return neigh.predict(x_test)