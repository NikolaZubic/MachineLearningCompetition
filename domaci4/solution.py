import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
import warnings
from sklearn.exceptions import DataConversionWarning
import sys

def read_train_and_test_data():
    return pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])


def preprocessing_and_feature_selection(data):
    label_encoder = LabelEncoder()

    data = data.drop("dead", axis=1)  # information available in injSeverity
    data = data.drop("airbag", axis=1)  # information available in abcat
    data = data.drop("deploy", axis=1)  # information available in abcat

    data["speed"] = label_encoder.fit_transform(data["speed"])
    data["seatbelt"] = label_encoder.fit_transform(data["seatbelt"])
    data["frontal"] = label_encoder.fit_transform(data["frontal"])
    data["sex"] = label_encoder.fit_transform(data["sex"])
    data["abcat"] = label_encoder.fit_transform(data["abcat"])
    data["occRole"] = label_encoder.fit_transform(data["occRole"])
    data["injSeverity"] = label_encoder.fit_transform(data["injSeverity"])

    return data


if __name__ == '__main__':
    # ignore DataConversionWarning that happens when converting dtype float to float64
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)

    train_data, test_data = read_train_and_test_data()

    # Since train data contains missing values, remove them
    train_data = train_data.dropna()

    train_data = preprocessing_and_feature_selection(train_data)
    test_data = preprocessing_and_feature_selection(test_data)

    X_train, Y_train = train_data.drop("speed", axis=1), train_data["speed"]
    X_test, Y_test = test_data.drop("speed", axis=1), test_data["speed"]

    # normalize before using ensemble methods
    standard_scaler = StandardScaler()
    X_train = standard_scaler.fit_transform(X_train)
    X_test = standard_scaler.transform(X_test)

    ada_boost = AdaBoostClassifier(n_estimators=70, learning_rate=0.7, random_state=120)

    gradient_boosting = GradientBoostingClassifier(n_estimators=320, subsample=0.9, min_samples_split=0.0001,
                                                   min_samples_leaf=0.00003, min_weight_fraction_leaf=0.00002,
                                                   max_depth=6, random_state=1234, max_features=9)

    decision_tree = DecisionTreeClassifier(random_state=360, max_depth=6, max_features='auto', min_samples_leaf=1,
                                           min_samples_split=2)

    gaussian_naive_bayes = GaussianNB(var_smoothing=1e-5)

    estimators_list = [('ada_boost', ada_boost), ('gradient_boosting', gradient_boosting),
                       ('decision_tree', decision_tree), ('gaussian_naive_bayes', gaussian_naive_bayes)]
    soft_voting_classifier = VotingClassifier(estimators=estimators_list, weights=[1.3333, 0.7190, 0.2, 0.0576],
                                              voting="soft")

    soft_voting_classifier.fit(X_train, Y_train)
    Y_pred = soft_voting_classifier.predict(X_test)

    print(f1_score(Y_test, Y_pred, average='micro'))
