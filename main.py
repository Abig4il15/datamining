import streamlit as st
import numpy as np

st.title('Analisis Sentimen Masyarakt terhadap Program MBG')
st.write("""Metode Support Vector Machine""")
dataset=st.sidebar.selectbox('pilih dataset', ['Dataset 1', 'Dataset 2', 'Dataset 3'])
if dataset == 'Dataset 1':
    st.write("You selected Dataset 1")
    data = np.random.rand(10, 2)
    st.line_chart(data)
