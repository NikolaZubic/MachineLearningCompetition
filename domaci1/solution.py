import pandas as pd
import numpy as np
import sys
#import matplotlib.pyplot as plt


def mse(y_actual, y_predicted):
    n = len(y_actual)
    s = 0.0
    for i in range(n):
        s += (y_predicted[i] - y_actual[i]) ** 2
    loss = s / n
    return loss


def rmse(y_actual, y_predicted):
    return np.sqrt(mse(y_actual, y_predicted))


def remove_outliers(train_dataset):
    # Save all that are below 2 standard deviations because they are not considered as 'weird' / outliers
    x = train_dataset['size'] / train_dataset['weight']
    condition = abs(x - np.mean(x)) < 2.8 * np.std(x)
    return train_dataset[condition]


class SimpleLinearRegression:
    def __init__(self, thetas=[0, 0]):
        self.thetas = thetas

    def predict(self, x):
        return self.thetas[0] + self.thetas[1] * x

    def train_bgd(self, X, Y, max_iters, learning_rate):
        # Batch gradient descent
        n = len(X)
        losses_by_iter = []
        # Initialize parameters
        init_theta = 0.003
        self.thetas = [init_theta, init_theta]
        # fit parameters
        for it in range(max_iters):
            predictions = [self.predict(x) for x in X]
            loss = mse(Y, predictions)
            losses_by_iter.append(loss)
            # print("Iteration %d: Loss: %f, theta0: %f, theta1: %f" % (it, loss, self.thetas[0], self.thetas[1]))
            residuals = [pred - y for pred, y in zip(predictions, Y)]
            # for each parameter
            for d in range(2):
                # calculate the gradient
                chain = 1
                s = 0.0
                for i in range(n):
                    if d == 1:
                        chain = X[i]
                    s += residuals[i] * chain
                gradient = 2 * s / n
                # update parameter
                self.thetas[d] -= learning_rate * gradient

        return losses_by_iter

    # params 1e-10, 100, [332, 0.2635]
    def fit(self, x_train, y_train, alpha, epochas, start_point='?'):
        if not start_point == '?':
            self.thetas = start_point
        n = len(x_train)
        losses = []

        best_thetas = self.thetas

        for i in range(epochas):
            y_p = self.predict(x_train)
            error = y_p - y_train
            self.thetas[0] = self.thetas[0] - alpha * 2 * np.sum(error) / n
            self.thetas[1] = self.thetas[1] - alpha * 2 * np.sum(error * x_train) / n

            current_loss = rmse(y_p, y_train)
            losses.append(current_loss)

            if (current_loss <= min(losses)):
                best_thetas = self.thetas.copy()

        self.thetas = best_thetas.copy()

    def train_norm(self, X, Y):
        x_train = np.array(X)
        y_train = np.array(Y)
        x_train.shape = (len(X), 1)
        ones = np.ones((len(X), 1))
        x_train = np.concatenate((ones, x_train), axis=1)

        y_train.shape = (len(X), 1)
        trained_thetas = np.array(self.thetas)
        trained_thetas.shape = (2, 1)
        x_train_transposed = x_train.transpose()
        trained_thetas = np.matmul(
            np.matmul(
                np.linalg.inv(
                    np.matmul(x_train_transposed, x_train)
                ), x_train_transposed),
            y_train)
        self.thetas[0] = trained_thetas[0]
        self.thetas[1] = trained_thetas[1]


if __name__ == '__main__':
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]

    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    train_data = remove_outliers(train_data)

    train_x = train_data['size'].values
    train_y = train_data['weight'].values

    test_x = test_data["size"].values
    test_y = test_data["weight"].values

    model = SimpleLinearRegression()

    model.fit(train_x, train_y, 1e-10, 100, [332, 0.2635])

    predicted_y = model.predict(test_x)

    error = rmse(predicted_y, test_y)
    print(error)
