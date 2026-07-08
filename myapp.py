import streamlit as st
import pickle
import numpy as np

#load model
with open('iris1_model.pkl','rb') as f:
    model=pickle.load(f)
st.set_page_config(page_title="Iris Flower Classifier")
st.title("Iris Floer Classifier")
st.write("Enter the measurements below:")
#input sliders
sepal_length=st.slider("Sepal Length (cm)" , 4.0,8.0,5.8)
sepal_width=st.slider("Sepal Length (cm)" , 2.0,4.5,3.0)
petal_length=st.slider("Sepal Length (cm)" , 1.0,7.0,4.0)
petal_width=st.slider("Sepal Length (cm)" , 0.1,2.5,1.2)
#prediction
features = np.array([[sepal_length,sepal_width,petal_length,petal_width]])

if st.button("Predict"):
    prediction = model.predict(features)
    class_names=['Setosa','Versilolor','Virginica']
    st.success(f"The predicted Iris Specie is: {class_names[prediction[0]]}")
