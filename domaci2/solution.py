import numpy as np
import pandas as pd
import sys

# Calculate RMSE
def mse(y_actual, y_predicted):
    n = len(y_actual)
    s = 0.0
    for i in range(n):
        s += (y_predicted[i] - y_actual[i]) ** 2
    loss = s / n
    return loss


def rmse(y_actual, y_predicted):
    return np.sqrt(mse(y_actual, y_predicted))


"""
Helper functions:
    * Normalize
    * Random data shuffling
    * K-fold cross validation with possible shuffling
    * Thresholding
    * Eucledian distance
    * Manhattan distance
    * Pearson correlation distance
    * Cosine similarity
"""


def normalize(X):
    l2 = np.atleast_1d(np.linalg.norm(X, 2, -1))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, -1)


def shuffle_data(x, y, seed=None):
    # https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.shuffle.html
    if seed:
        np.random.seed(seed)
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    return x[idx], y[idx]


def k_fold_cross_validation(x, y, k, shuffle=True):
    if shuffle:
        x, y = shuffle_data(x, y)

    number_of_samples = len(y)
    unused_samples = {}
    number_of_unused_samples = (number_of_samples % k)

    if number_of_unused_samples != 0:
        unused_samples["x"], unused_samples["y"] = x[-number_of_unused_samples:], y[-number_of_unused_samples:]
        # remove unused samples from x and y
        x, y = x[:-number_of_unused_samples], y[:-number_of_unused_samples]

    x_split, y_split = np.split(x, k), np.split(y, k)

    k_fold_cross_validation_sets = []

    for i in range(k):
        x_test, y_test = x_split[i], y_split[i]
        x_train, y_train = np.concatenate(x_split[:i] + x_split[i + 1:], axis=0), np.concatenate(
            y_split[:i] + y_split[i + 1:], axis=0)
        k_fold_cross_validation_sets.append([x_train, x_test, y_train, y_test])

    # unused samples are appended to the last set as training examples
    if number_of_unused_samples != 0:
        np.append(k_fold_cross_validation_sets[-1][0], unused_samples["x"], axis=0)
        np.append(k_fold_cross_validation_sets[-1][2], unused_samples["y"], axis=0)

    return np.array(k_fold_cross_validation_sets)


def thresholding(value):
    if value == 0:
        return 0
    elif value > 0:
        return 1.0
    else:
        return -1.0


def eucledian_distance(point_x, point_y):
    s = 0.0
    for i in range(len(point_x)):
      s += (point_x[i] - point_y[i]) ** 2

    dist = np.sqrt(s)
    return dist


def manhattan_distance(point_x, point_y):
    dist = 0.0
    for i in range(len(point_x)):
        dist += np.abs(point_x[i] - point_y[i])

    return dist


def pearson_correlation_distance(point_x, point_y):
    x_mean = np.average(point_x)
    y_mean = np.average(point_y)
    xy_dist = 0.0
    x_squared_dist = 0.0
    y_squared_dist = 0.0
    for i in range(len(point_x)):
        xy_dist += (point_x[i] - x_mean) * (point_y[i] - y_mean)
        x_squared_dist += (point_x[i] - x_mean) ** 2
        y_squared_dist += (point_y[i] - y_mean) ** 2

    dist = 1 - (xy_dist / np.sqrt(x_squared_dist * y_squared_dist))
    return dist


def cosine_similarity(point_x, point_y):
    dot_prod = np.dot(point_x, point_y)
    x_magnitude = np.linalg.norm(point_x)
    y_magnitude = np.linalg.norm(point_y)
    cos = dot_prod / (x_magnitude * y_magnitude)
    return cos


"""
Data preprocessing.
"""

def remove_outlayers_hc(df):
    for index, row in df.iterrows():
        if  row['godina_iskustva'] >= 60:
            df.drop(index, inplace=True)
        elif  row['plata'] > 200000:
            df.drop(index, inplace=True)
        elif row['zvanje'] == 'Prof' and row['oblast'] == 'B':
            df.drop(index, inplace=True)



def data_preprocessing(train_path, test_path):
    all_labels = ['zvanje', 'oblast', 'godina_doktor', 'godina_iskustva', 'pol', 'plata']
    extract_labels = ['godina_doktor',
                      'godina_iskustva',
                    #   'zvanje',
                      'zvanje_AssocProf',
                      'zvanje_AsstProf',
                      'zvanje_Prof',
                    #   'oblast',
                    #   'pol',
                      'oblast_A',
                      'oblast_B',
                      'plata']

    label = ['plata']

    labels_one_hot = ['godina_doktor',
                      'godina_iskustva',
                    #   'zvanje',
                      'zvanje_AssocProf',
                      'zvanje_AsstProf',
                      'zvanje_Prof',
                    #   'oblast'
                      'oblast_A',
                      'oblast_B'
                    #   'pol'

                      ]

    train_df = pd.read_csv(train_path)

    remove_outlayers_hc(train_df)

    train_df['zvanje'] = pd.Categorical(train_df['zvanje'])
    train_df_d = pd.get_dummies(train_df['zvanje'], prefix='zvanje')
    train_df = pd.concat([train_df, train_df_d], axis=1)

    train_df['oblast'] = pd.Categorical(train_df['oblast'])
    train_df_d = pd.get_dummies(train_df['oblast'], prefix='oblast')
    train_df = pd.concat([train_df, train_df_d], axis=1)

    sd_godina_iskustva = np.std(train_df['godina_iskustva'])
    mean_godina_iskustva = np.mean(train_df['godina_iskustva'])

    sd_godina_doktor = np.std(train_df['godina_doktor'])
    mean_godina_doktor = np.mean(train_df['godina_doktor'])

    train_df['godina_doktor'] = (train_df['godina_doktor'] - mean_godina_doktor) / sd_godina_doktor
    train_df['godina_iskustva'] = (train_df['godina_iskustva'] - mean_godina_iskustva) / sd_godina_iskustva
    # train_df['oblast'] = (train_df['oblast'] - mean_oblast) / sd_oblast

    train_df = train_df[extract_labels]
    test_df = pd.read_csv(test_path)[all_labels]


    test_df['zvanje'] = pd.Categorical(test_df['zvanje'])
    test_df_d = pd.get_dummies(test_df['zvanje'], prefix='zvanje')
    test_df = pd.concat([test_df, test_df_d], axis=1)

    test_df['oblast'] = pd.Categorical(test_df['oblast'])
    test_df_d = pd.get_dummies(test_df['oblast'], prefix='oblast')
    test_df = pd.concat([test_df, test_df_d], axis=1)

    test_df['godina_doktor'] = (test_df['godina_doktor'] - mean_godina_doktor) / sd_godina_doktor
    test_df['godina_iskustva'] = (test_df['godina_iskustva'] - mean_godina_iskustva) / sd_godina_iskustva
    # test_df['oblast'] = (test_df['oblast'] - mean_oblast) / sd_oblast

    test_df = test_df[extract_labels]

    X = train_df[labels_one_hot].to_numpy()
    Y = train_df[label].values.reshape(-1, 1)
    Y = Y.reshape(Y.shape[0], )

    x = test_df[labels_one_hot].to_numpy()
    y = test_df[label].values.reshape(-1, 1)

    return X, Y, x, y


"""
Multiple regression models.
We have tried:
    * Ridge Regression
    * Lasso Regression
    * Elastic Net Regression
At the end, model of choice is: Elastic Net with dominant Lasso
"""


class ElasticNetRegression(object):
    """
    Useful resources:
    * https://statweb.stanford.edu/~jhf/ftp/glmnet.pdf
    * https://xavierbourretsicotte.github.io/
    """
    def __init__(self, learning_rate=0.01, rho_value=0.00667, n_iterations=2000, weights=np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])):
        # 23860.48217225028 learning_rate=1.48, rho_value=0.00268
        self.n_iterations = n_iterations
        self.weights = weights
        self.learning_rate = learning_rate
        self.rho_value = rho_value

    def preprocess_coordinate_descent(self, x):
        m, n = x.shape  # m - number of examples, n - number of features
        l2_regularization_parameter = m * self.learning_rate * (1.0 - self.rho_value)
        l1_regularization_parameter = m * self.learning_rate * self.rho_value
        #self.weights = np.zeros(n)
        # Return an array (ndim >= 1) laid out in Fortran order in memory, for a little bit of a faster computing.
        x = np.asfortranarray(x)
        return x, l1_regularization_parameter, l2_regularization_parameter

    def fit(self, x, y):
        # minimize (1/2) * norm(y - X w, 2)^2 + l1_reg ||w||_21 + (1/2) * l2_reg norm(w, 2)^2
        # (1 / (2 * n_samples)) * ||Y - XW||^2_Fro + alpha * ||W||_21
        #  ||W||_21 = \\sum_i \\sqrt{\\sum_j w_{ij}^2}
        init_step = True
        x, l1, l2 = self.preprocess_coordinate_descent(x)
        xy_dot = np.dot(x.T, y)
        n = x.shape[1]  # number of features
        feature_set = set(range(n))
        feature_correlations = np.zeros(shape=(n, n))
        get_norm = np.sum(x**2, axis=0)

        # initialize gradients
        grads = np.zeros(n)

        for i in range(self.n_iterations):
            for j in feature_set:
                weights = self.weights[j]
                if init_step:
                    feature_correlations[:, j] = np.dot(x[:, j], x)
                    grads[j] = xy_dot[j] - np.dot(feature_correlations[:, j], self.weights)

                self.weights[j] = thresholding(grads[j] + weights * get_norm[j]) * max(
                    abs(grads[j] + weights * get_norm[j]) - l2, 0) / (get_norm[j] + l1)

                # Update gradients if there is a change in weights
                if weights != self.weights[j]:
                    for k in feature_set:
                        if self.n_iterations >= 1 or k <= j:
                            grads[k] -= feature_correlations[j, k] * (self.weights[j] - weights)

            init_step = False

            feature_set_copy = set.copy(feature_set)
            for f in feature_set_copy:
                if self.weights[f] == 0:
                    feature_set.remove(f)

    def predict(self, x):
        return x @ self.weights.T


"""
KNN implementation
"""


class KnnRegression(object):

    def __init__(self, k, metric=eucledian_distance):
        """
        :param k: Number of nearest neighbors to consider for prediction
        :param metric: Metric function that calculates distance between two vectors (numpy arrays)
        """
        self.k = k
        self.metric = metric
        self.X = np.array([])
        self.Y = np.array([])

    def train(self, X, Y):
        self.X = X
        self.Y = Y

    def predict(self, sample):
        # tuples containing distance to sample and the value for each point
        neighbors = [(self.metric(sample, x), y) for x, y in zip(self.X, self.Y)]
        # sort data by distance to given sample
        neighbors.sort(key=lambda neighbor_data: neighbor_data[0])
        # get values of nearest neghbors
        values_sum = 0.0
        k = min(self.k, len(self.Y))

        for i in range(k):
            values_sum += neighbors[i][1]

        estimate = values_sum / k
        return estimate

# Helper function for hyper-parameter tuning
def optimize_k(min_k, max_k, X, Y):
    losses_by_k = []
    best_loss = float('inf')
    best_k = 0
    k_axis = np.arange(min_k, max_k + 1)
    for k in k_axis:
        knn_reg = KnnRegression(k)
        loss = 0.0
        folds = 5
        for [X_train, X_test, y_train, y_test] in k_fold_cross_validation(X, Y, folds):
            knn_reg.train(X_train, y_train)
            y_predicted = [knn_reg.predict(x) for x in X_train]
            loss += rmse(y_test, y_predicted)

        loss /= folds
        losses_by_k.append(loss)
        if loss < best_loss:
            best_loss = loss
            best_k = k

    return best_k


"""
Kernel regression implementation
"""


class KernelRegression:

    def __init__(self, bandwidth):
        self.bandwidth = bandwidth
        self.X = np.array([])
        self.Y = np.array([])

    def epanechnikov_kernel(self, target, sample):
        distance = manhattan_distance(target, sample)
        t = distance / self.bandwidth
        if t < 1:
            return 0.75 * (1 - t ** 2)
        else:
            return 0

    def triweight_kernel(self, target, sample):
        distance = manhattan_distance(target, sample)
        t = distance / self.bandwidth
        if t < 1:
            return 35 * (1 - t ** 2) ** 3 / 32
        else:
            return 0

    def gaussian_kernel(self, target, sample):
        distance = manhattan_distance(target, sample)
        w = np.exp(-1 * distance ** 2 / self.bandwidth)
        w /= np.sqrt(2 * np.pi)
        return w

    def train(self, X, Y):
        self.X = X
        self.Y = Y

    def predict(self, sample):
        weights = [self.gaussian_kernel(sample, x) for x in self.X]
        weighted_values_sum = 0.0
        for i in range(len(self.Y)):
            weighted_values_sum += weights[i] * self.Y[i]

        weights_sum = sum(weights)
        kernel_weighted_average = weighted_values_sum / weights_sum
        return kernel_weighted_average

# Helper function for hyper-parameter tuning
def optimize_bandwidth(X, Y):
    """
    Uses k-fold cross validation to estimate optimal bandwidth for kernel regression

    :param X: Training set data
    :param Y: Training set labels
    :return: Optimal bandwidth (bandwidth which results with minimal loss)
    """
    folds = 5
    losses = []
    best_loss = float('inf')
    optimal_bandwidth = 0
    bandwidths = np.arange(start=0.01, stop=0.5, step=0.01)
    for bandwidth in bandwidths:
        total_loss = 0.0
        for [X_train, X_test, y_train, y_test] in k_fold_cross_validation(X, Y, folds):
            reg = KernelRegression(bandwidth)
            reg.train(X_train, y_train)
            predictions = [reg.predict(x) for x in X_test]
            loss = rmse(y_test, predictions)
            total_loss += loss

        average_loss = total_loss / folds
        losses.append(average_loss)
        if average_loss < best_loss:
            best_loss = average_loss
            optimal_bandwidth = bandwidth

    return optimal_bandwidth


if __name__ == '__main__':
    train_path = sys.argv[1]
    test_path = sys.argv[2]

    X, Y, x, y = data_preprocessing(train_path, test_path)

    elastic_net_model = ElasticNetRegression()
    elastic_net_model.fit(X, Y)
    y_pred = elastic_net_model.predict(x)
    rmserr = rmse(y, y_pred)

    print(float(rmserr))

    # Uncomment the following code for KNN
    """
    X, Y, x, y = data_preprocessing(train_path, test_path)
    X = normalize(X)
    x = normalize(x)
    knn_reg = KnnRegression(10, metric=manhattan_distance)
    knn_reg.train(X, Y)
    y_predicted = [knn_reg.predict(data) for data in x]
    loss = rmse(y, y_predicted)
    print(loss[0])
    """

    # Uncomment the following code for Kernel Regression
    """
    X, Y, x, y = data_preprocessing(train_path, test_path)
    X = normalize(X)
    x = normalize(x)
    kernel_reg = KernelRegression(0.01)
    kernel_reg.train(X, Y)
    y_predicted = [kernel_reg.predict(data) for data in x]
    loss = rmse(y, y_predicted)
    print(loss[0])
    """
