from altair.vegalite.v4.schema.channels import X
import streamlit as st
#importing necessary libraries
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sb

st.title("Iris Flower Predictor")
st.header("Enter Your Data: ")

def user_input():
    sepal_length =st.slider('Sepal Length',0.0,10.0,5.0)
    sepal_width = st.slider('Sepal Width',0.0,10.0,5.0)
    petal_length =st.slider('Petal Length',0.0,10.0,5.0)
    petal_width =st.slider('Petal Width',0.0,10.0,5.0)
    data = {"Sepal Length":sepal_length,
    "Sepal Width": sepal_width,
    "Petal Length": petal_length,
    "Petal Width": petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features
df = user_input()

st.subheader("User Input")
st.write(df)   

flowers = load_iris()
X = flowers.data
Y = flowers.target
model = LogisticRegression()
model.fit(X,Y)
predicted_flower = model.predict(df)
pred_values = model.predict(X)
score = str(model.score(X,Y)*100)


st.subheader("Predicted Flower:")
st.write(pd.DataFrame(flowers.target_names[predicted_flower]))

st.subheader("Model Accuracy:")
st.write(score + "%")


st.set_option('deprecation.showPyplotGlobalUse', False)

st.subheader("Confustion Matrix:")
cm = confusion_matrix(Y,pred_values)
sb.heatmap(cm,annot=True)
plt.xlabel("Predicted")
plt.ylabel("True")
st.pyplot()

