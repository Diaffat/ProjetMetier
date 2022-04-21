import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
#Classe
#selection p, t pour classe
data=pd.read_csv("data Projet Metier IA.csv") 
data

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

data.shape

x=data.iloc[:,0:9]
y=data.iloc[:,10] 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=0)
x_train.shape

from sklearn import svm

model=svm.SVC()
model.fit(x_train,y_train)
model.score(x_test,y_test)

from sklearn.model_selection import GridSearchCV 

svc = svm.SVC(C= 0.1, gamma=1, kernel='linear')
svc.fit(x_train, y_train)
svc.score(x_test,y_test)
from sklearn import tree
model1 = tree.DecisionTreeClassifier(ccp_alpha=0.1, max_depth=3, min_samples_split=2, min_weight_fraction_leaf=0.1)
model1.fit(x_train, y_train)
model1.score(x_test,y_test)

from sklearn.neighbors import KNeighborsClassifier
model_3 =  KNeighborsClassifier(n_neighbors=1)

model_3.fit(x_train, y_train)

from sklearn.ensemble import VotingClassifier,BaggingClassifier # VotingRegressor
from sklearn.linear_model import SGDClassifier # Model 1
from sklearn.tree import DecisionTreeClassifier # Model 2
from sklearn.neighbors import KNeighborsClassifier # Model 3
model_final = VotingClassifier([('KNN',model_3),('SVC',svc),('Tree',model1)],voting='hard')

for m in (model_3,svc,model1,model_final):
  m.fit(x_train,y_train)
  print(m.__class__.__name__ , m.score(x_test,y_test))
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
cm=confusion_matrix(y,model_final.predict(x),labels=model_final.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model_final.classes_)
disp.plot()
import pickle
 
# Save the trained model as a pickle string.
OURS = pickle.dumps(model_3)

# Load the pickled model
modelload = pickle.loads(OURS)
 
# Use the loaded pickled model to make predictions
modelload.predict(x_test,y_test)

import plotly.graph_objects as go

st.title("Classification de la qualité des eaux de l'oued MOULOUYA")
nav = st.sidebar.radio("Menu", ["Home", "Prediction"])

if nav == "Home":
    st.image("l-eau.jpg", width=300)
    site = pd.DataFrame({"nom site":["Ait boulmane", "Ait OhaOhaki", "Source Arbalou", "Krouchene=Irhdis","Boumia", "Zaïda","AnzarOufounas", "Aval AnzarOufounas", "Anzegmir avant barrage", "Anzegmir Amont", "Tamdafelt", "Missour", "Outat Al Haj", "Tindint", "Moulouya Amont Melloulou", "Moulouya Aval Melloulou", "Moulouya Amont Za", "Moulouya aval Za", "Sebra","Safsaf", "Pont Hassan II","Pré-Estuaire"],
                         "lat": [506388.899, 508378.875, 510270.048, 521267.473, 528157.842, 541119.868, 522710.665, 523517.135, 529513.51, 545371.311, 608563.842, 632589.028, 657696.741, 667765.779, 688527.58, 691373.449, 716153.504, 717261.981, 750073.689, 752451.688, 770940.249, 774425.807],
                         "lon": [ 223813.71, 225419.701, 231109.507, 239017.267, 235764.271, 246884.595, 203771.489, 209213.863, 213616.77, 238557.759, 254198.713, 273722.851, 304862.154, 340891.631, 403762.245, 406608.844, 442463.459, 442636.19, 479499.227, 481921.171, 498404.754, 503613.795]})
    st.map(site)
    if st.checkbox("Show Training data"):
        st.write(data)

    graph = st.selectbox("What kind of graph?", ["PH", "T", "CE", "O2", "NH", "NO", "SO", "PO", "DBO5"])
    if graph == "PH":
        lt = go.Layout(xaxis=dict(range=[0, 66]), yaxis=dict(range=[1,4]))

        fig = go.Figure(data=go.Scatter(x=np.arange(0, 65), y=data["PH"], mode="markers"), layout=lt)
        st.plotly_chart(fig)

    if graph=="T":
        fig, ax = plt.subplots()
        plt.scatter(np.arange(66), data["T"])
        plt.ylabel("T")
        st.pyplot(fig)

    if graph=="CE":
        fig, ax = plt.subplots()
        plt.scatter(np.arange(66), data["CE"])
        plt.ylabel("CE")
        st.pyplot(fig)
    if graph=="O2":
        fig, ax = plt.subplots()
        plt.scatter(np.arange(66), data["O2"])
        plt.ylabel("O2")
        st.pyplot(fig)
    if graph=="NH":
        fig, ax = plt.subplots()
        plt.scatter(np.arange(66), data["NH"])
        plt.ylabel("NH")
        st.pyplot(fig)
    if graph=="NO":
        fig, ax = plt.subplots()
        plt.scatter(np.arange(66), data["NO"])
        plt.ylabel("NO")
        st.pyplot(fig)
    if graph=="SO":
        fig, ax = plt.subplots()
        plt.scatter(np.arange(66), data["SO"])
        plt.ylabel("SO")
        st.pyplot(fig)
    if graph=="PO":
        fig, ax = plt.subplots()
        plt.scatter(np.arange(66), data["PO"])
        plt.ylabel("PO")
        st.pyplot(fig)
    if graph=="DBO5":
        fig, ax = plt.subplots()
        plt.scatter(np.arange(66), data["DBO5"])
        plt.ylabel("DBO5")
        st.pyplot(fig)


if nav == "Prediction":

    st.header(" Saisir les donner d'entrées")
    PH = st.number_input("PH")
    T = st.number_input("T")
    CE = st.number_input("CE")
    O2 = st.number_input("O2")
    NH = st.number_input("NH")
    NO = st.number_input("NO")
    SO = st.number_input("SO")
    PO = st.number_input("PO")
    DBO5 = st.number_input("DBO5")
    val = [PH, T, CE,	O2,	NH,	NO,	SO,	PO,	DBO5]
    ypred = modelload.predict([val])
    if st.button("Prédire"):
        st.write(val)
        if ypred[0]==1:
            classe = "Excellente"
        if ypred[0]==2:
            classe = "Bonne"
        if ypred[0]==3:
            classe = "Peu Polluée"
        if ypred[0]==4:
            classe = "Mauvaise"
        if ypred[0]==5:
            classe = "Très mauvaise"
        st.success(f"La qualité de l'eau est: {classe}")

