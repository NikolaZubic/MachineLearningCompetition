import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import v_measure_score


class Performer:
    def __init__(self, classifier, normalizer):
        self.normalizer = normalizer
        self.classifier = classifier

    def fit(self, training_data):
        x = self.__normalize_training_data(training_data.drop('region', axis=1).values)
        y = training_data['region'].values
        self.classifier.fit(x, y)

    def predict(self, test_data):
        x = self.__normalize_test_data(test_data.drop('region', axis=1).values)
        y = test_data['region'].values
        y_predicted = self.classifier.predict(x)
        return self.__calculate_error(y, y_predicted)

    def __normalize_training_data(self, training_data):
        return pd.DataFrame(self.normalizer.fit_transform(training_data))

    def __normalize_test_data(self, test_data):
        return pd.DataFrame(self.normalizer.transform(test_data))

    def __calculate_error(self, y_true, y_predicted):
        return v_measure_score(y_true, y_predicted)


if __name__ == '__main__':
    training_data = pd.read_csv(sys.argv[1])
    test_data = pd.read_csv(sys.argv[2])

    le = LabelEncoder()
    categorical_column_headers = ['region', 'oil']
    for column in categorical_column_headers:
        training_data[column] = le.fit_transform(training_data[column])
        test_data[column] = le.transform(test_data[column])

    df_mean = training_data.mean()
    infant_mean = df_mean.infant
    training_data['infant'] = training_data['infant'].fillna(infant_mean)

    performer = Performer(
        classifier=GaussianMixture(n_components=4, covariance_type='diag', reg_covar=0.0005, max_iter=12, n_init=11,
                                   random_state=290, weights_init=[0.098, 0.089, 0.026, 0.787]),
        normalizer=StandardScaler())

    performer.fit(training_data=training_data)

    score = performer.predict(test_data=test_data)

    print(score)
