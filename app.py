import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.title("Aplikasi Prediksi Penyakit Jantung")

data_path = "heart.csv"
data = pd.read_csv(data_path)

st.subheader("Data Awal")
st.write(data.head())

# Pra-pemrosesan data
data = data.dropna()

# Pemisahan fitur dan target
X = data.drop(columns=['target'])  # Asumsikan 'target' adalah kolom label
y = data['target']

# Pemisahan data pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pelatihan model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.subheader("Akurasi Model")
st.write(f"Akurasi: {accuracy * 100:.2f}%")

# Prediksi berdasarkan input pengguna
st.subheader("Prediksi Penyakit Jantung")
input_data = []
for column in X.columns:
    value = st.number_input(f"Masukkan nilai untuk {column}", value=float(X[column].mean()))
    input_data.append(value)

input_data = np.array(input_data).reshape(1, -1)
prediction = model.predict(input_data)[0]

if st.button("Prediksi"):
    if prediction == 1:
        st.write("Model memprediksi bahwa Anda berisiko terkena penyakit jantung.")
    else:
        st.write("Model memprediksi bahwa Anda tidak berisiko terkena penyakit jantung.")
