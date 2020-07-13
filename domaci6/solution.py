import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.utils import resample


class Preprocessor:

    def __init__(self, training_data, test_data, for_label, x_labels, y_labels, scaler):
        self.training_data = training_data
        self.test_data = test_data
        self.for_label = for_label
        self.x_labels = x_labels
        self.y_labels = y_labels
        self.scaler = scaler

    def _label_encoding(self):
        label_encoder = LabelEncoder()
        for label in self.for_label:
            self.training_data[label] = label_encoder.fit_transform(self.training_data[label].astype(str))
            self.test_data[label] = label_encoder.transform(self.test_data[label].astype(str))

    def _processing_nan(self):
        #self.training_data.dropna(inplace=True)
        df_mean = self.training_data.mean()
        self.training_data['year'] = self.training_data['year'].fillna(df_mean.year)
        self.training_data['age'] = self.training_data['age'].fillna(df_mean.age)
        self.training_data['maritl'] = self.training_data['maritl'].fillna(df_mean.maritl)
        self.training_data['education'] = self.training_data['education'].fillna(df_mean.education)
        self.training_data['jobclass'] = self.training_data['jobclass'].fillna(df_mean.jobclass)
        self.training_data['health'] = self.training_data['health'].fillna(df_mean.health)
        self.training_data['health_ins'] = self.training_data['health_ins'].fillna(df_mean.health_ins)
        self.training_data['wage'] = self.training_data['wage'].fillna(df_mean.wage)

    def sets(self):
        self._label_encoding()
        self._processing_nan()

        x_train = self.training_data[self.x_labels].values
        y_train = self.training_data[self.y_labels].values.ravel()

        x_test = self.test_data[self.x_labels].values
        y_test = self.test_data[self.y_labels].values.ravel()

        x_train = self.scaler.fit_transform(x_train)
        x_test = self.scaler.transform(x_test)

        return x_train, y_train, x_test, y_test


class Performer:
    def __init__(self, pca, classifier):
        self.pca = pca
        self.classifier = classifier

    def fit(self, x_train, y_train):
        x_train = self.pca.fit_transform(x_train)
        self.classifier.fit(x_train, y_train)

    def predict(self, x_test):
        x_test = self.pca.transform(x_test)
        return self.classifier.predict(x_test)


if __name__ == "__main__":
    training_data = pd.read_csv(sys.argv[1])
    test_data = pd.read_csv(sys.argv[2])

    black = training_data[training_data['race'] == '2. Black']
    non_black = training_data[training_data['race'] != '2. Black']
    black_upsampled = resample(black, replace=True, random_state=13, n_samples=300)

    asian = training_data[training_data['race'] == '3. Asian']
    non_asian = training_data[training_data['race'] != '3. Asian']
    asian_upsampled = resample(asian, replace=True, random_state=117, n_samples=50)

    training_data = pd.concat([non_black, black_upsampled, asian_upsampled])

    for_label = ['health', 'health_ins', 'race', 'education', 'jobclass', 'maritl']

    x_labels = [
        'year',
        'age',
        'health',
        'health_ins',
        'wage',
        'education',
        'jobclass',
        'maritl',
    ]

    y_labels = ['race']

    pp = Preprocessor(training_data=training_data, test_data=test_data, for_label=for_label, x_labels=x_labels,
                      y_labels=y_labels, scaler=StandardScaler())

    x_train, y_train, x_test, y_test = pp.sets()

    performer = Performer(pca=KernelPCA(kernel='poly', degree=14, n_components=5, random_state=2),
                          classifier=RandomForestClassifier(max_depth=6, n_estimators=5, random_state=243))

    performer.fit(x_train, y_train)
    y_pred = performer.predict(x_test)

    score = f1_score(y_test, y_pred, average='micro')

    print(score)
