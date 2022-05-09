import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import confusion_matrix
import streamlit as st


# reading the data
def read_data(file_name):
    data = pd.read_csv(file_name)
    return data


# data preprocessing
def preprocessing(data):
    data = data[['Text', 'Star']]
    data.dropna(inplace=True)
    print("data: \n", data.head())
    data['sentiments'] = data.Star.apply(lambda x: 0 if x in [1, 2] else 1)
    return data


# split the data into train and test
def split_data(data):
    X = data['Text']
    y = data['sentiments']
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


# vectorizing the data
def vectorize(X_train, X_test):
    #tfidf vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    return X_train_vectorized, X_test_vectorized


# training and testing the model
def train_and_test_model(X_train_vectorized, y_train, X_test_vectorized,
                         y_test):
    svcl = svm.SVC(kernel='rbf')
    #clf_sv = GridSearchCV(svcl, params)
    svcl.fit(X_train_vectorized, y_train)

    svcl_score = svcl.score(X_test_vectorized, y_test)
    print("Results for Support Vector Machine with tfidf: ", svcl_score)

    train_and_test_model.y_pred_sv = svcl.predict(X_test_vectorized)

    tn, fp, fn, tp = confusion_matrix(y_test,
                                      train_and_test_model.y_pred_sv).ravel()

    print("tn: ", tn, "fp:", fp, "fn: ", fn, "tp:", tp)

    tpr_sv = round(tp / (tp + fn), 4)
    tnr_sv = round(tn / (tn + fp), 4)
    print("tpr_sv: ", tpr_sv, "tnr_sv: ", tnr_sv)


st.title("Chrome Reviews")

uploaded_file = st.file_uploader(
    "Choose a file for checking review/rating discrepancy")

try:
    if uploaded_file is not None:
        data = read_data(uploaded_file)

        data = preprocessing(data)

        X_train, X_test, y_train, y_test = split_data(data)
        X_train_vectorized, X_test_vectorized = vectorize(X_train, X_test)

        train_and_test_model(X_train_vectorized, y_train, X_test_vectorized,
                             y_test)

        st.write(
            "The list of reviews where the reviews and ratings probably don't match are as below"
        )
        for i in range(0, len(data)):
            if (train_and_test_model.y_pred_sv[i] == 1 and y_test[i] == 0):
                st.write(data['Text'][i], data['Star'][i])
except Exception as e:
    st.write(e)
    raise e