import pickle

from numpy import vectorize
import streamlit as st
import pandas as pd

saved_model = pickle.load(open('svcl.pkl', 'rb'))
vectorize = pickle.load(open('vector.pkl', 'rb'))

st.title("Chrome Reviews")
uploaded_file = st.file_uploader(
    "Choose a file for checking review/rating discrepancy")
try:
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, usecols=['Text', 'Star'])
        data.dropna(inplace=True)
        x = data['Text']
        print("x: ", x)
        x_vec = vectorize.transform(x)
        data['sentiments'] = data.Star.apply(lambda x: 0 if x in [1, 2] else 1)
        y = data['sentiments']

        y_pred_sv = saved_model.predict(x_vec)
        st.write(
            "The list of reviews where the reviews and ratings probably don't match are as below"
        )
        for i in range(0, len(data)):
            if (y_pred_sv[i] == 1 and y.iloc[i] == 0):
                st.write(data['Text'][i], "\t", data['Star'][i])
except Exception as e:
    st.write(e)
    raise e
