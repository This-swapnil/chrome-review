import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import confusion_matrix

import pickle


class Train:
    def __init__(self, data) -> None:
        self.data = data

    # data preprocessing
    def preprocessing(self):
        """Data Preprocessing

        Returns:
            Preprocessed data
        """
        data = self.data[['Text', 'Star']]
        data.dropna(inplace=True)
        print("data: \n", data.head())
        data['sentiments'] = data.Star.apply(lambda x: 0 if x in [1, 2] else 1)
        return data

    # split the data into train and test
    def split_data(self, data):
        """Spliting the data into dependent and independent featuare and then split into train and test

        Args:
            data (_type_): data to preprocess
        Returns:
            X_train: 80% of the data from independent feature
            X_test: 20% of the data from independent feature
            y_train: 80% of the data from dependent feature
            y_test: 20% of the data from dependent feature
        """
        X = data['Text']
        y = data['sentiments']
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=42)
        return X_train, X_test, y_train, y_test

    # vectorizing the data
    def vectorize(self, X_train, X_test):
        """ Vectorizing the data
        Args:
            X_train (_type_): 80% of the data from independent feature
            X_test (_type_): 20% of the data from independent feature
        Returns:
            X_train_vectorized: Vectorized data from 80% of the data from dependent feature
            X_test_vectorized: Vectorized data from 20% of the data from dependent feature
        """
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test_vectorized = vectorizer.transform(X_test)
        pickle.dump(vectorizer, open('vector.pkl', 'wb'))
        return X_train_vectorized, X_test_vectorized

    # training and testing the model
    def train_model(self, X_train_vectorized, y_train):
        """Training the model

        Args:
            X_train_vectorized (_type_): Vectorized data
            y_train (_type_): Dependent feature
        Returns:
            fitted model
            pickling the model
        """
        svcl = svm.SVC(kernel='rbf')
        svcl.fit(X_train_vectorized, y_train)
        pickle.dump(svcl, open('svcl.pkl', 'wb'))

    def test_model(self, X_test_vectorized, y_test):
        """ Testing the model

        Args:
            X_test_vectorized (_type_): Vectorized data
            y_test (_type_): Dependent feature (unseen data)
        """
        saved_model = pickle.load(open('svcl.pkl', 'rb'))
        svcl_score = saved_model.score(X_test_vectorized, y_test)
        print("Results for Support Vector Machine with tfidf: ", svcl_score)

        y_pred_sv = saved_model.predict(X_test_vectorized)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_sv).ravel()

        print("tn: ", tn, "fp:", fp, "fn: ", fn, "tp:", tp)

        tpr_sv = round(tp / (tp + fn), 4)
        tnr_sv = round(tn / (tn + fp), 4)
        print("tpr_sv: ", tpr_sv, "tnr_sv: ", tnr_sv)


if __name__ == "__main__":
    pd = pd.read_csv('chrome_reviews.csv')
    obj = Train(pd)
    data = obj.preprocessing()
    X_train, X_test, y_train, y_test = obj.split_data(data=data)
    X_train_vectorized, X_test_vectorized = obj.vectorize(X_train, X_test)
    obj.train_model(X_train_vectorized, y_train)
    obj.test_model(X_test_vectorized, y_test)