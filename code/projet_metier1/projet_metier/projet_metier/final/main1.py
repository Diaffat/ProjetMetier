from audioop import add
import email
from email.policy import default
import string
import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from streamlit_option_menu import option_menu
from PIL import Image
#from streamlit_aggrid import Agrid
#from raceplotly.plots import barplot
from collections import deque
from streamlit.delta_generator import DeltaGenerator as _DeltaGenerator
#import leafmap.kepler as leafmap
from streamlit.elements.arrow import ArrowMixin
from pandas.io.formats.style import Styler
import pyrebase
#from datatime import datatime
import json
import requests  # pip install requests
import streamlit as st  # pip install streamlit
from streamlit_lottie import st_lottie
from torch import classes  # pip install streamlit-lottie
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


#def load_lottieurl(url: str):
   # r = requests.get(url)
   # if r.status_code != 200:
    #    return None
    #return r.json()
c="",
def loti(c) :
    lottie_coding = load_lottiefile(c)  # replace link to local lottie file
          
    st_lottie(
              lottie_coding,
              speed=1,
              reverse=False,
              loop=True,
              quality="high", # medium ; high
         #renderer="svg", # canvas
              height=None,
              width=200,
              key=None,
               )  


def principal():
     data = pd.read_csv("projet_metier\Classeur11.csv") 
     x = data.iloc[:,0:9]
     y= data.iloc[:,10]
     choose = option_menu("Main Menu",["Home","Dataset","Map","Regression","Prediction","About"],
    icons=['house','file','pin-map','graph-up','pie-chart-fill','person lines fill'],
     menu_icon = "list", default_index=0,
    styles={
        "container": {"padding": "5!important", "background-color": ""},
        "icon": {"color": "orange", "font-size": "18px"}, 
        "nav-link": {"font-size": "10px", "text-align": "left", "margin":"0px", "--hover-color": ""},
        "nav-link-selected": {"background-color": ""},
    },orientation = "horizontal"
    )


       #*********************************************Import dataset**************************************************************

     if choose == "Dataset":
            loti("projet_metier\projet_metier\data-analysis.json")
         
            st.subheader("Import Dataset")
            datafile = st.file_uploader("Upload CSV",type=["CSV"])
            if datafile is not None:
               st.write(type(datafile))
               file_details = {"filename":datafile.name,"filetype":datafile.type,"filesize":datafile.size}
               st.write(file_details)
        
               data = pd.read_csv(datafile)
               st.dataframe(data)
         #datafile = st.file_uploader("Upload CSV",type=["CSV"])        
                          

      #*********************************************Regression**************************************************************    
     elif choose == "Regression":
       # st.image("l-eau.jpg", width=300)
    #site = pd.DataFrame({"nom site":["Ait boulmane", "Ait OhaOhaki", "Source Arbalou", "Krouchene=Irhdis","Boumia", "Zaïda","AnzarOufounas", "Aval AnzarOufounas", "Anzegmir avant barrage", "Anzegmir Amont", "Tamdafelt", "Missour", "Outat Al Haj", "Tindint", "Moulouya Amont Melloulou", "Moulouya Aval Melloulou", "Moulouya Amont Za", "Moulouya aval Za", "Sebra","Safsaf", "Pont Hassan II","Pré-Estuaire"],
                         #"lat": [506388.899, 508378.875, 510270.048, 521267.473, 528157.842, 541119.868, 522710.665, 523517.135, 529513.51, 545371.311, 608563.842, 632589.028, 657696.741, 667765.779, 688527.58, 691373.449, 716153.504, 717261.981, 750073.689, 752451.688, 770940.249, 774425.807],
                         #"lon": [ 223813.71, 225419.701, 231109.507, 239017.267, 235764.271, 246884.595, 203771.489, 209213.863, 213616.77, 238557.759, 254198.713, 273722.851, 304862.154, 340891.631, 403762.245, 406608.844, 442463.459, 442636.19, 479499.227, 481921.171, 498404.754, 503613.795]})
    #st.map(site)
    #if st.checkbox("Show Table"):
        #st.write(data)
        loti("projet_metier\projet_metier\smooth-chart.json")
    

        graph = st.selectbox("What kind of graph?", ["PH", "T", "CE", "O2", "NH", "NO", "SO", "PO", "DBO5"])
       
        if graph == "PH":
          fig, ax = plt.subplots()
          plt.scatter(np.arange(66), data["PH"])
          plt.ylabel("PH")
          st.pyplot(fig)
          from sklearn.linear_model import LinearRegression
          regPH = LinearRegression()
          regPH.fit(x,y)
          ypredPH = regPH.predict(x)
          st.write(ypredPH.shape)
          plt.plot(np.arange(66),c=ypredPH[0])
          plt.show()
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
  #*********************************************Prediction**************************************************************
     elif choose == "Prediction":
         loti("projet_metier\98831-pie-chart.json")
         st.title("Prediction")
         coll1, coll2= st.columns(2)
         kind = coll1.selectbox("Kind of Predict",["One Predict","Whith File"])
         model = coll2.selectbox("Model",["SVC","Tree","KNN","Voting","Bagging","Perceptron","MLPClassifier","RidgeClassifier"])
        
    
    
    
         if kind =="One Predict":
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
        
         elif kind == "Whith File":
              st.subheader("Import Dataset")
              datafile = st.file_uploader("Upload CSV",type=["CSV"])
              if datafile is not None:
                  st.write(type(datafile))
                  file_details = {"filename":datafile.name,"filetype":datafile.type,"filesize":datafile.size}
                  st.write(file_details)
        
                  data = pd.read_csv(datafile)
           
                  st.dataframe(data)
        
         if model =="SVC":
        
          
           x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
           svc = svm.SVC(C= 0.1, gamma=1, kernel='linear')
           svc.fit(x_train, y_train)
           st.header("SVC Params")
           namec,firstc,midlec=st.columns(3)
           Min_value_C = firstc.number_input("Min_value C")
           Max_value_C = midlec.number_input("Max_value C")
           namec.header("C:")
           namegamma,firstgamma,midlegamma=st.columns(3)
           Min_value_g = firstgamma.number_input("Min_value Gamma")
           Max_value_g = midlegamma.number_input("Max_value Gamma")
           namegamma.header("GAMMA:")
            
           name,last=st.columns(2)
           name.header("Kermel :")
           kernel = last.multiselect("Choose Kernel",["sigmoid","poly","rbf"])
           st.write(kernel)
           col3,col4 = st.columns(2)
           gridsearch = col3.button("GridSearch")
           randomsearch = col4.button("RandomSearch")   
         
           #st.write(firstc)
        
         elif model =="Tree":
            
             x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
             mtree = tree.DecisionTreeClassifier(ccp_alpha=0.1, max_depth=3, min_samples_split=2, min_weight_fraction_leaf=0.1)
             mtree.fit(x_train, y_train)

             st.header("Tree Params")
             tcl1,tcl2,tcl3=st.columns(3)
              
             tcl1.header("min_impurity")
             t=tcl2.number_input("choose min value decrease")
             tmax=tcl3.number_input("choose max  value decrease")
             trl1,trcl2,trcl3=st.columns(3)
             trl1.header("min-leaf ")
             trs= trcl2.number_input("choose min value ")
             tres= trcl3.number_input("choose max value ")
             trls1,trcls2,trcls3=st.columns(3)
             trls1.header("min_split")
             tr= trcls2.number_input("choose min value split ")
             tre= trcls3.number_input("choose max value split ")
             tr1,tr2,tr3=st.columns(3)
             tr1.header("ccp alpha")
             trf= tr2.number_input("choose min value ccp ")
             trf2= tr3.number_input("choose max value ccp")
             tri1,tri2=st.columns(2)
             tri1.header("max features")
             trfi= tri2.multiselect("Choose one ",["auto", "sqrt", "log2","None"] )
             col3,col4 = st.columns(2)
             gridsearch = col3.button("GridSearch")
             randomsearch = col4.button("RandomSearch")   
         

         elif model =="KNN":
            
              x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
              mknn =  KNeighborsClassifier(n_neighbors=1)
              mknn.fit(x_train, y_train)
              st.header("KNN Params")
              kcl1,kcl2,kcl3=st.columns(3)
              
              kcl1.header("n_neighbors")
              k=kcl2.number_input("choose min neighbors")
              kmax=kcl3.number_input("choose max neighbors")
              wcl1,wcl2=st.columns(2)
              wcl1.header("weight")
              l= wcl2.multiselect("Choose weight",["uniform","distance"])
              acl1,acl2=st.columns(2)
              
              acl1.header("    Algorithm")
              ac=acl2.multiselect("Choose algo",["ball_tree","auto","kd_tree"])
              col3,col4 = st.columns(2)
              gridsearch = col3.button("GridSearch")
              randomsearch = col4.button("RandomSearch")   
         elif model =="Perceptron":
         
              x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
              percep = Perceptron()
              percep.fit(x_train, y_train)
              percep.score(x_test,y_test)
              st.header("Perceptron Params")
              kcl1,kcl2,kcl3=st.columns(3)
              
              kcl1.header("n_neighbors")
              k=kcl2.number_input("choose min neighbers")
              kmax=kcl3.number_input("choose max neighbers")
              wcl1,wcl2=st.columns(2)
              wcl1.header("weight")
              l= wcl2.multiselect("Choose weight",["uniform","distance"])
              acl1,acl2=st.columns(2)
              
              acl1.header("    Algorithm")
              ac=acl2.multiselect("Choose algo",["ball_tree","auto","kd_tree"])
              col3,col4 = st.columns(2)
              gridsearch = col3.button("GridSearch")
              randomsearch = col4.button("RandomSearch") 
        
         elif model =="MLPClassifier":
         
              x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
              mlp = MLPClassifier()
              mlp.fit(x_train, y_train)
              mlp.score(x_test,y_test)
              st.header("MLPClassifier Params")
              ml1,ml2,ml3=st.columns(3)
              
              ml1.header("alpha")
              ml=ml2.number_input("choose min alpha")
              mll=ml3.number_input("choose max alpha")
              mll1,mll2=st.columns(2)
              mll1.header("Max iter",)
              mlll= mll2.selectbox("Choose max iter",[200,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000 ])
              s1,s2=st.columns(2)
              s1.header("Solver",)
              s= s2.multiselect("Choose solver",["lbfgs", "sgd", "adam"])
              mlp1,mlp2,mlp3=st.columns(3)
              
              mlp1.header("hidden layers")
              mlp22=mlp2.number_input("Choose min hidden",min_value=1, max_value=10, value=1)
              mlp23=mlp3.number_input("Choose max hidden",min_value=1, max_value=50, value=1)
              #st.write(mlp.score(x_test,y_test))
              col3,col4 = st.columns(2)
              gridsearch = col3.button("GridSearch")
              randomsearch = col4.button("RandomSearch")  
         elif model =="RidgeClassifier":
         
              x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
              ridge = RidgeClassifier()
              ridge.fit(x_train, y_train)
              ridge.score(x_test,y_test)
              st.header("RidgeClassifier Params")
              ri1,ri2,ri3=st.columns(3)
              
              ri1.header("alpha")
              r1=ri2.number_input("choose mini alpha")
              r2=ri3.number_input("choose maxi alpha")
              rid1,rid2=st.columns(2)
              rid1.header("Max iter",)
              rid= rid2.selectbox("Choose max iter",[200,500,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,3000,5000 ])
              ridg1,ridg2=st.columns(2)
              ridg1.header("Solver",)
              ridg= ridg2.multiselect("Choose solver(s)",['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'])
              #st.write(mlp.score(x_test,y_test))
              col3,col4 = st.columns(2)
              gridsearch = col3.button("GridSearch")
              randomsearch = col4.button("RandomSearch") 

              from sklearn.ensemble import AdaBoostClassifier 

         elif model =="AdaBoostClassifier ":
              x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
              ada = AdaBoostClassifier ()
              ada.fit(x_train, y_train)
              ada.score(x_test,y_test)
              st.header("AdaBoostClassifier  Params")
              ad1,ad2,ad3=st.columns(3)
              
              ad1.header("Learning_rate")
              ad=ad2.number_input("choose min rate",min_value=1, max_value=10, value=1)
              a=ad3.number_input("choose max rate",min_value=1, max_value=20, value=1)
              add1,add2=st.columns(2)
              add1.header("Numbers estimators",)
              addd= add2.selectbox("Choose one number",[10,15,20,50,80,100,110,120,130,140,150,160,170,180,190,200,300,400,500])
              adb1,adb2=st.columns(2)
              adb1.header("Algorithm",)
              adb= adb2.multiselect("Choose algorithm(s)",['SAMME', 'SAMME.R'])
             
              #st.write(ada.score(x_test,y_test))
              col3,col4 = st.columns(2)
              gridsearch = col3.button("GridSearch")
              randomsearch = col4.button("RandomSearch")  
              from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
         elif model =="QuadraticDiscriminantAnalysis":
         
              x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
              qua = QuadraticDiscriminantAnalysis()
              qua.fit(x_train, y_train)
              qua.score(x_test,y_test)
              st.header("QuadraticDiscriminantAnalysis Params")
              q1,q2,q3=st.columns(3)
              
              q1.header("tol")
              ql=q2.number_input("choose min",min_value=0.0001, max_value=0.1, value=0.0001)
              qll=q3.number_input("choose max",min_value=0.0001, max_value=2.0, value=0.0001)
              qu1,qu2=st.columns(2)
              qu1.header("store_covariance")
              qu= qu2.selectbox("Choose one",["True", "False"])
              qlp1,qlp2,qlp3=st.columns(3)
              
              qlp1.header("reg_param")
              qlp22=qlp2.number_input("Choose min reg",min_value=0, max_value=10, value=1)
              qlp23=qlp3.number_input("Choose max reg",min_value=1, max_value=50, value=1)
              #st.write(mlp.score(x_test,y_test))
              col3,col4 = st.columns(2)
              gridsearch = col3.button("GridSearch")
              randomsearch = col4.button("RandomSearch")  
              from sklearn.mixture import GaussianMixture

         elif model =="GaussianMixture":
         
              x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
              gau = GaussianMixture()
              gau.fit(x_train, y_train)
              gau.score(x_test,y_test)
              st.header("GaussianMixture Params")
              ga1,ga2,ga3=st.columns(3)
              
              ga1.header('n_components')
              ga=ga2.number_input("choose min components")
              g=ga3.number_input("choose max components")

              gll1,gll2=st.columns(2)
              gll1.header("Max iteration",)
              glll= gll2.selectbox("Choose maxi iter",[20,100,110,120,130,140,150,160,170,180,190,200 ])
              g11,g33,g44=st.columns(2)
              g11.header("tol")
              g3=g33.number_input("choose min",min_value=0.0001, max_value=0.1, value=0.0001)
              g4l=g44.number_input("choose max",min_value=0.0001, max_value=2.0, value=0.0001)

              gu1,gu2=st.columns(2)
              gu1.header("init_params ")
              gu= gu2.selectbox("Choose one",["kmeans", "random"])

              glp1,glp2,glp3=st.columns(3)
              glp1.header("reg_covar")
              glp22=glp2.number_input("Choose min reg",min_value=0, max_value=10, value=1)
              glp23=glp3.number_input("Choose max reg",min_value=1, max_value=50, value=1)

              g1,g2=st.columns(2)
              g1.header("covariance_type",)
              g22= g2.multiselect("Choose covariance",["spherical", "tied", "diag", "full"])
              gaus1,gaus2,gaus3=st.columns(3)
              
              gaus1.header("cv")
              gaus=gaus2.number_input("Choose min cv",min_value=1, max_value=2, value=1)
              gaus23=gaus3.number_input("Choose max cv",min_value=1, max_value=10, value=1)
              #st.write(mlp.score(x_test,y_test))
              col3,col4 = st.columns(2)
              gridsearch = col3.button("GridSearch")
              randomsearch = col4.button("RandomSearch")  
              from sklearn.multiclass import OneVsOneClassifier

         elif model =="OneVsOneClassifier":
         
              x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
              one= OneVsOneClassifier()
              one.fit(x_train, y_train)
              one.score(x_test,y_test)
              st.header("OneVsOneClassifier Params")
              on1,on2,on3=st.columns(3)
              
              on1.header("n_jobs")
              one1=on2.number_input("choose min alpha")
              one2=on3.number_input("choose max alpha")
              oe1,oe2=st.columns(2)
              oe1.header("Estimator",)
              oee= oe2.multiselect("Choose estimator(s)",["GaussianMixture", "SVC", "ada"])
              #st.write(mlp.score(x_test,y_test))
              col3,col4 = st.columns(2)
              gridsearch = col3.button("GridSearch")
              randomsearch = col4.button("RandomSearch") 
              from sklearn.semi_supervised import LabelPropagation 

         elif model =="LabelPropagation ":
         
              x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
              llp = LabelPropagation ()
              llp.fit(x_train, y_train)
              llp.score(x_test,y_test)
              st.header("LabelPropagation Params")
              ll1,ll2,ll3=st.columns(3)
              
              ll1.header("gamma")
              ll=ll2.number_input("choose min gamma")
              lll=ll3.number_input("choose max gamma")
              lll1,lll2=st.columns(2)
              lll1.header("Maxi iter",)
              llll= lll2.selectbox("Choose max itera",[200,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000 ])
              l1,l2=st.columns(2)
              l1.header("kernel",)
              LL= l2.multiselect("Choose kernel",["knn", "rbf"])
              llp1,llp2,llp3=st.columns(3)
              
              llp1.header("n_NEIGHBORS")
              llp22=llp2.number_input("Choose min ",min_value=1, max_value=5, value=1)
              llp23=llp3.number_input("Choose max ",min_value=1, max_value=50, value=1)
              #st.write(mlp.score(x_test,y_test))
              col3,col4 = st.columns(2)
              gridsearch = col3.button("GridSearch")
              randomsearch = col4.button("RandomSearch")  
              from sklearn.mixture import GMM 

         elif model =="GMM ":
         
              x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
              gmm = GMM ()
              gmm.fit(x_train, y_train)
              gmm.score(x_test,y_test)
              st.header("GMM Params")
              gml1,gml2,gml3=st.columns(3)
              
              gml1.header("n_components")
              gml=gml2.number_input("choose minimum")
              gmll=gml3.number_input("choose maximum")
              g111,g222=st.columns(2)
              g111.header("covariance_type",)
              gm= g222.multiselect("Choose type(s)",[ "spherical", "tied", "diag", "full"])
              gmlp1,gmlp2,gmlp3=st.columns(3)
              
              gmlp1.header("n_iter")
              gmlp22=mlp2.number_input("Choose min n",min_value=1, max_value=10, value=1)
              gmlp23=mlp3.number_input("Choose max n",min_value=1, max_value=50000, value=1)
              #st.write(mlp.score(x_test,y_test))
              col3,col4 = st.columns(2)
              gridsearch = col3.button("GridSearch")
              randomsearch = col4.button("RandomSearch")  


         elif model=="Voting":
              voting = VotingClassifier([('KNN', mknn), ('SVC', svc), ('Tree', mtree)], voting='hard')

              for m in (mknn, svc, mtree, voting):
                   m.fit(x_train, y_train)
                   print(m.__class__.__name__ , m.score(x_test, y_test))
        
         
         
         if gridsearch:

           if model == "SVC":
             params = {"C":np.arange(Min_value_C,Max_value_C,0.1),"gamma":np.arange(Min_value_g,Max_value_g,0.1),"kernel":kernel}
             grid = GridSearchCV(svc,params)
             grid.fit(x_train,y_train)
             grid.best_params_
             svc = grid.best_estimator_
           if model == "KNN":
             params = {"n_neighbors":np.arange(int(k),int(kmax),1),"weights":l,"algorithm":ac}
             grid = GridSearchCV(mknn,params)
             grid.fit(x_train,y_train)
             grid.best_params_
             knn = grid.best_estimator_
           if model == "Tree":
             params = {"min_impurity_decrease":np.arange(t,tmax,0.1),"min_samples_split":np.arange(tr,tre,0.1),"min_samples_leaf":np.arange(trs,tres,0.1),"ccp_alpha":np.arange(trf,trf2,0.1),"max_features":trfi}
             grid = GridSearchCV(mtree,params)
             grid.fit(x_train,y_train)
             grid.best_params_
             mtree = grid.best_estimator_
           if model == "MLPClassifier":
             params = {'solver': [s], 'max_iter': [mlll ], 'alpha': np.arange(ml, mll), 'hidden_layer_sizes':np.arange(mlp22, mlp23), 'random_state':[0,1,2,3,4,5,6,7,8,9]}
             grid = GridSearchCV(mlp, params)
             grid.fit(x_train,y_train)
             grid.best_params_
             mlp = grid.best_estimator_
           if model == "RidgeClassifier":
             params = {'solver': [rid], 'max_iter': [ridg], 'alpha': np.arange(r1, r2), 'random_state':[0,1,2,3,4,5,6,7,8,9]}
             grid = GridSearchCV(ridge, params)
             grid.fit(x_train,y_train)
             grid.best_params_
             ridge = grid.best_estimator_
           if model == "AdaBoostClassifier":
             params = {'algorithm': [adb], 'n_estimators': [add], 'learning_rate': np.arange(ad, a), 'random_state':[0,1,2,3,4,5,6,7,8,9]}
             grid = GridSearchCV(ada, params)
             grid.fit(x_train,y_train)
             grid.best_params_
             ada= grid.best_estimator_
           if model == "QuadraticDiscriminantAnalysis":
             params = {'store_covariance': [qu], 'tol': np.arange(ql, qll), 'reg_param': np.arange(qlp22, qlp23)}
             grid = GridSearchCV(qua, params)
             grid.fit(x_train,y_train)
             grid.best_params_
             qua= grid.best_estimator_
           if model == "GaussianMixture":
             params = {'n_components': np.arange(ga, g), 'cv': np.arange(gaus, gaus23),'covariance_type': [g22],'max_iter': [glll], 'tol': np.arange(g3, g4l),'init_params': [gu], 'reg_covar': np.arange(glp22, glp23)}
             grid = GridSearchCV(gau, params)
             grid.fit(x_train,y_train)
             grid.best_params_
             gau= grid.best_estimator_
           if model == "OneVsOneClassifier":
             params = {'estimator': [oee], 'n_jobs': np.arange(one1, one2)}
             grid = GridSearchCV(ridge, params)
             grid.fit(x_train,y_train)
             grid.best_params_
             ridge = grid.best_estimator_
           if model == "LabelPropagation":
             params = {"n_neighbors":np.arange(llp22,llp23,1),"gamma":np.arange(ll,lll,0.1),"kernel":[LL],"max_iter":[llll]}
             grid = GridSearchCV(llp,params)
             grid.fit(x_train,y_train)
             grid.best_params_
             llp = grid.best_estimator_
           if model == "GMM":
             params = {'n_components': np.arange(gml, gmll),'covariance_type': [gm],'n_iter': np.arange(gmlp22, gmlp23)}
             grid = GridSearchCV(gmm, params)
             grid.fit(x_train,y_train)
             grid.best_params_
             gmm= grid.best_estimator_


         if randomsearch:
            if model == "SVC":
             params = {"C":np.arange(Min_value_C,Max_value_C,0.1),"gamma":np.arange(Min_value_g,Max_value_g,0.1),"kernel":kernel}
             grid = RandomizedSearchCV(svc,params)
             grid.fit(x_train,y_train)
             grid.best_params_
             svc = grid.best_estimator_
            if model == "KNN":
             params = {"n_neighbors":np.arange(int(k),int(kmax),1),"weights":l,"algorithm":ac}
             grid = RandomizedSearchCV(mknn,params)
             grid.fit(x_train,y_train)
             grid.best_params_
             knn = grid.best_estimator_
            if model == "Tree":
             params = {"min_impurity_decrease":np.arange(t,tmax,0.1),"min_samples_split":np.arange(tr,tre,0.1),"min_samples_leaf":np.arange(trs,tres,0.1),"ccp_alpha":np.arange(trf,trf2,0.1),"max_features":trfi}
             grid = RandomizedSearchCV(mtree,params)
             grid.fit(x_train,y_train)
             grid.best_params_
             mtree = grid.best_estimator_
            if model == "MLPClassifier":
             params = {'solver': [s], 'max_iter': mlll, 'alpha': 10.0 ** -np.arange(ml, mll), 'hidden_layer_sizes':np.arange(mlp22, mlp23), 'random_state':[0,1,2,3,4,5,6,7,8,9]}
             grid =  RandomizedSearchCV(MLPClassifier(), params, n_jobs=-1)
             grid.best_params_
             mlp = grid.best_estimator_
            if model == "RidgeClassifier":
             params = {'solver': [rid], 'max_iter': [ridg], 'alpha': np.arange(r1, r2), 'random_state':[0,1,2,3,4,5,6,7,8,9]}
             grid = RandomizedSearchCV(ridge, params)
             grid.fit(x_train,y_train)
             grid.best_params_
             ridge = grid.best_estimator_
            if model == "AdaBoostClassifier":
             params = {'algorithm': [adb], 'n_estimators': [addd], 'learning_rate': np.arange(ad, a), 'random_state':[0,1,2,3,4,5,6,7,8,9]}
             grid =  RandomizedSearchCV(ada, params)
             grid.fit(x_train,y_train)
             grid.best_params_
             ada= grid.best_estimator_
            if model == "QuadraticDiscriminantAnalysis":
             params = {'store_covariance': [qu], 'tol': np.arange(ql, qll), 'reg_param': np.arange(qlp22, qlp23)}
             grid = RandomizedSearchCV(qua, params)
             grid.fit(x_train,y_train)
             grid.best_params_
             qua= grid.best_estimator_
            if model == "GaussianMixture":
             params = {'n_components': np.arange(ga, g), 'cv': np.arange(gaus, gaus23),'covariance_type': [g22],'max_iter': [glll], 'tol': np.arange(g3, g4l),'init_params': [gu], 'reg_covar': np.arange(glp22, glp23)}
             grid = RandomizedSearchCV(gau, params)
             grid.fit(x_train,y_train)
             grid.best_params_
             gau= grid.best_estimator_
            if model == "OneVsOneClassifier":
             params = {'estimator': [oee], 'n_jobs': np.arange(one1, one2)}
             grid = RandomizedSearchCV(one, params)
             grid.fit(x_train,y_train)
             grid.best_params_
             one = grid.best_estimator_
            if model == "LabelPropagation":
             params = {"n_neighbors":np.arange(llp22,llp23,1),"gamma":np.arange(ll,lll,0.1),"kernel":[LL],"max_iter":[llll]}
             grid = RandomizedSearchCV(llp,params)
             grid.fit(x_train,y_train)
             grid.best_params_
             llp = grid.best_estimator_
            if model == "GMM":
             params = {'n_components': np.arange(gml, gmll),'covariance_type': [gm],'n_iter': np.arange(gmlp22, gmlp23)}
             grid = RandomizedSearchCV(gmm, params)
             grid.fit(x_train,y_train)
             grid.best_params_
             gmm= grid.best_estimator_


         result = st.button("Results")
         if result:
           rcol1,rcol2 = st.columns(2)
           rcol1.header("Score")
           rcol1.write(mtree.score(x_test,y_test))
           rcol2.header("Curve")
           rchoose = rcol2.selectbox("Choose curve",["Confusion Matrix","ROC Curve","Precision Recall Curve"])
           if rchoose =="Confusion Matrix":
             from sklearn.metrics import plot_confusion_matrix
             rcol2.subheader("Confusion Matrix") 
             plot_confusion_matrix(mtree, x_test, y_test, display_labels = mtree.classes_)
             rcol2.pyplot()

           if rchoose =="ROC Curve":
             st.subheader("ROC Curve") 
             if model =="SVC":
                plot_roc_curve(svc, x_test, y_test)
                st.pyplot()
             if model =="KNN":
                plot_roc_curve(knn, x_test, y_test)
                st.pyplot()
             if model =="Tree":
                plot_roc_curve(mtree, x_test, y_test)
                st.pyplot()
             if model =="Voting":
                plot_roc_curve(voting, x_test, y_test)
                st.pyplot()
             if model =="Bagging":
                plot_roc_curve(svc, x_test, y_test)
                st.pyplot()

           if rchoose =="Precision Recall Curve":
             st.subheader("Precision-Recall Curve")
             if model =="SVC":
                plot_precision_recall_curve(svc, x_test, y_test)
                st.pyplot()
             if model =="KNN":
                plot_precision_recall_curve(knn, x_test, y_test)
                st.pyplot()
             if model =="Tree":
                plot_precision_recall_curve(tree, x_test, y_test)
                st.pyplot()
             if model =="Voting":
                plot_precision_recall_curve(voting, x_test, y_test)
                st.pyplot()
             if model =="Bagging":
                plot_precision_recall_curve(svc, x_test, y_test)
                st.pyplot()
                

           
           


      # **************Home***************************************************
     elif choose=="Home":
       #st.image("projet_metier/umi.jpg",width=350)
       #first,last = st.columns(2)
       #first.image("projet_metier/fs.png")
       #last.image("projet_metier/ensam.jpg",width=270)
       
       loti("projet_metier\projet_metier\loading-animation.json")
     elif choose =="Map":
        site = pd.DataFrame({"nom site":["Ait boulmane", "Ait OhaOhaki", "Source Arbalou", "Krouchene=Irhdis","Boumia", "Zaïda","AnzarOufounas", "Aval AnzarOufounas", "Anzegmir avant barrage", "Anzegmir Amont", "Tamdafelt", "Missour", "Outat Al Haj", "Tindint", "Moulouya Amont Melloulou", "Moulouya Aval Melloulou", "Moulouya Amont Za", "Moulouya aval Za", "Sebra","Safsaf", "Pont Hassan II","Pré-Estuaire"],
                         "lat": [506388.899, 508378.875, 510270.048, 521267.473, 528157.842, 541119.868, 522710.665, 523517.135, 529513.51, 545371.311, 608563.842, 632589.028, 657696.741, 667765.779, 688527.58, 691373.449, 716153.504, 717261.981, 750073.689, 752451.688, 770940.249, 774425.807],
                         "lon": [ 223813.71, 225419.701, 231109.507, 239017.267, 235764.271, 246884.595, 203771.489, 209213.863, 213616.77, 238557.759, 254198.713, 273722.851, 304862.154, 340891.631, 403762.245, 406608.844, 442463.459, 442636.19, 479499.227, 481921.171, 498404.754, 503613.795]})
        st.map(site)









#///////////////////////////////////////////////////////////////////////////////////


firebaseConfig = {
  "apiKey": "AIzaSyCdTgMVxd0iUr2kxmI9lN2FuvAHTSrGr80",
  "authDomain": "streamlitsign.firebaseapp.com",
  "projectId": "streamlitsign",
  "databaseURL":"gs://streamlitsign.appspot.com",
  "storageBucket": "streamlitsign.appspot.com",
  "messagingSenderId": "217936996069",
 " appId": "1:217936996069:web:58d0de12de4e1d635a560c",
  "measurementId": "G-ZE7R381DQ9"
}
# Firebase authenticatio
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

# database
db = firebase.database()
storage = firebase.storage()
# authentication

choise = st.sidebar.selectbox("Login/Logup",["Sin Up","LogIn"])
mail = st.sidebar.text_input("Please enter your email Adress")
password = st.sidebar.text_input("Please enter your password",type="password")

if choise =="Sin Up":
    handle = st.sidebar.text_input("Please enter your handle name",value="Default")
    submit = st.sidebar.button("Create my account")
    if submit:
        user = auth.create_user_with_email_and_password(mail,password)
        st.success("Your account is created succesfully!")
        st.balloons()
        user = auth.sign_in_with_email_and_password(mail,password)
        db.child(user["LocalID"]).child("Handle").set(handle)
        db.child(user["LocalID"]).child("ID").set(user["LocalID"])
        st.title("Welcome" + handle)
if choise =="LogIn":
    login = st.sidebar.button("LogIn")
    if login:
        user = auth.sign_in_with_email_and_password(mail,password)
        #st.success("good")
        principal()
        
            
        
    









    
#********************************tree*********************************************

    
#***************************************************


    

#************************************************

    
#*************************************************
    
    
    #if st.button("Prédire"):
       # st.write(val)
       # if ypred[0]==1:
           # classe = "Excellente"
       # if ypred[0]==2:
        #    classe = "Bonne"
       # if ypred[0]==3:
        #    classe = "Peu Polluée"
       # if ypred[0]==4:
        #    classe = "Mauvaise"
        #if ypred[0]==5:
        #    classe = "Très mauvaise"
       # st.success(f"La prediction est: {classe}")







    




