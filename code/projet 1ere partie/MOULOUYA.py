import streamlit as st
from streamlit_observable import observable
import pandas as pd
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np
import cv2
from st_aggrid import AgGrid
from raceplotly.plots import barplot
from collections import deque
import plotly.express as px
import io 
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
data = pd.read_excel("data3.xlsx")
x = data.iloc[:,1:10]
y = data.iloc[:,10]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=0)

# **************le premier modele SVC***************************************************
from sklearn.model_selection import GridSearchCV 
svc = svm.SVC(C=0.9470000000000007,gamma=0.01,kernel='linear')
svc.fit(x_train,y_train)
#sorted(clf.cv_results_.keys())

#********************************tree*********************************************

model1=tree.DecisionTreeClassifier(ccp_alpha=0.1,max_depth=3,min_weight_fraction_leaf=0.1,min_samples_split=8)
model1.fit(x_train, y_train)
#***************************************************
model_3=  KNeighborsClassifier(n_neighbors=1)
model_3.fit(x_train,y_train)
model_3.score(x_test,y_test)
#************************************************

model_final = VotingClassifier([('KNN', model_3), ('SVC', svc), ('Tree', model1)], voting='hard')

for m in (model_3, svc, model1, model_final):
    m.fit(x_train, y_train)
    print(m.__class__.__name__ , m.score(x_test, y_test))
#*************************************************



st.title("Classification de la qualité des eaux de l'oued MOULOUYA")
with st.sidebar:
    choose = option_menu("App Gallery", ["Home", "Prédiction","Nos différents sites","Nos données d'entrainement","Contact","Map"],
                         icons=['house', 'bi bi-moisture', 'bi bi-diagram-3-fill', 'bi bi-activity','person lines fill',"bi bi-map"],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )
if choose == "Home":
    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Classification de la qualité des eaux </p>', unsafe_allow_html=True)    
    with col2:               # To display brand log
        st.image("l-eau.jpg", width=130 )
    
    st.write("La Moulouya (en rifain : Melwacht, en latin Mulucha1, en arabe : ملوية Malwiyyah) est un fleuve du Maroc qui prend naissance à la jonction du massif du Moyen et du Haut Atlas dans la région d'Almssid dans la province de Midelt. Il est long de 600 km[réf. nécessaire] et se jette dans la mer Méditerranée, dans la région du Rif, dans les plaines de Kebdana, à l’extrême nord-est du Maroc. Son embouchure est située à 14 km de la frontière algéro-marocaine.Dans l'Antiquité, la Moulouya a un temps marqué la limite entre la Maurétanie et le royaume de Numidie.\n\nTo read Sharone's data science posts, please visit her Medium blog at: https://medium.com/@insightsbees")    
    st.image("l-eau.jpg", width=700 )
elif choose == "Contact":
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Contact Form</p>', unsafe_allow_html=True)
    with st.form(key='columns_in_form2',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
        #st.write('Please help us improve!')
        Name=st.text_input(label='Please Enter Your Name') #Collect user feedback
        Email=st.text_input(label='Please Enter Email') #Collect user feedback
        Message=st.text_input(label='Please Enter Your Message') #Collect user feedback
        submitted = st.form_submit_button('Submit')
        if submitted:
            st.write('Thanks for your contacting us. We will respond to your questions or inquiries as soon as possible!')

if  choose  == "Home":
    st.image("l-eau.jpg", width=1400)
if  choose  == "Nos données d'entrainement":
    if st.checkbox("Show Training data"):
       st.write(data)

    graph = st.selectbox("What kind of graph?", ["pH", "T", "CE", "O2", "NH", "NO", "SO", "PO", "DBO5"])
    if graph=="pH":
        lt = go.Layout(xaxis=dict(range=[0, 66]), yaxis=dict(range=[1,4]))

        fig = go.Figure(data=go.Scatter(x=np.arange(0, 65), y=data["pH"], mode="markers"), layout=lt)
        st.plotly_chart(fig)

    if graph=="T":
        fig, ax = plt.subplots()
        plt.scatter(np.arange(66), data["Temp"])
        plt.ylabel("T")
        st.pyplot(fig)

    if graph=="CE":
        fig, ax = plt.subplots()
        plt.scatter(np.arange(66), data["Con"])
        plt.ylabel("CE")
        st.pyplot(fig)
    if graph=="O2":
        fig, ax = plt.subplots()
        plt.scatter(np.arange(66), data["O_diss"])
        plt.ylabel("O2")
        st.pyplot(fig)
    if graph=="NH":
        fig, ax = plt.subplots()
        plt.scatter(np.arange(66), data["N_NH"])
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
    raceplot = barplot(data, item_column='Classe', value_column='Altitude en m', time_column='Stationscode') 
    raceplot.plot(item_label = 'LES CLASSES', value_label = 'Altitude', frame_duration = 800)
    st.write('---')
    st.markdown('<p class="font">VISUALISATIONS DES PARAMETRES...</p>', unsafe_allow_html=True)
    column_list=list(data)
    column_list = deque(column_list)
    column_list.appendleft('-')
    with st.form(key='columns_in_form'):
            text_style = '<p style="font-family:sans-serif; color:red; font-size: 15px;">***Veuillez choisir les inputs***</p>'
            st.markdown(text_style, unsafe_allow_html=True)
            col1, col2, col3 = st.columns( [1, 1, 1])
            with col1:
                item_column=st.selectbox('Bar column:',column_list, index=0, help='Choose the column in your data that represents the station, temp, O_diss, DBO5, etc.') 
            with col2:    
                value_column=st.selectbox('Metric column:',column_list, index=0, help='Choose the column in your data that represents the value/metric of each bar, e.g., population, gdp, etc.') 
            with col3:    
                time_column=st.selectbox('Time column:',column_list, index=0, help='Choose the column in your data that represents the time series, e.g., year, month, etc.')   

            text_style = '<p style="font-family:sans-serif; color:blue; font-size: 15px;">***Customize and fine-tune your plot (optional)***</p>'
            st.markdown(text_style, unsafe_allow_html=True)
            col4, col5, col6 = st.columns( [1, 1, 1])
            with col4:
                direction=st.selectbox('Choose plot orientation:',['-','Horizontal','Vertical'], index=0, help='Specify whether you want the bar chart race to be plotted horizontally or vertically. The default is horizontal' ) 
                if direction=='Horizontal'or direction=='-':
                    orientation='horizontal'
                elif  direction=='Vertical':   
                    orientation='vertical'
            with col5:
                item_label=st.text_input('Add a label for bar column:', help='For example: Top 10 countries in the world by 2020 GDP')  
            with col6:
                value_label=st.text_input('add a label for metric column', help='For example: GDP from 1965 - 2020') 

            col7, col8, col9 = st.columns( [1, 1, 1])
            with col7:
                num_items=st.number_input('Choose how many bars to show:', min_value=5, max_value=50, value=10, step=1,help='Enter a number to choose how many bars ranked by the metric column. The default is top 10 items.')
            with col8:
                format=st.selectbox('Show by Year or Month:',['-','By Year','By Month'], index=0, help='Choose to show the time series by year or month')
                if format=='By Year' or format=='-':
                    date_format='%Y'
                elif format=='By Month':
                    date_format='%x'   
            with col9:
                chart_title=st.text_input('Add a chart title', help='Add a chart title to your plot')    
            
            col10, col11, col12 = st.columns( [1, 1, 1])
            with col10:
                speed=st.slider('Animation Speed',10,500,100, step=10, help='Adjust the speed of animation')
                frame_duration=500-speed  
            with col11:
                chart_width=st.slider('Chart Width',500,1000,500, step=20, help='Adjust the width of the chart')
            with col12:    
                chart_height=st.slider('Chart Height',500,1000,600, step=20, help='Adjust the height of the chart')
        
            submitted = st.form_submit_button('Submit')
            st.write('---')
            if submitted:        
                if item_column=='-'or value_column=='-'or time_column=='-':
                  st.warning("Vous devez completer la selection des inputs")
                else: 
                 st.markdown('<p class="font">Generating your bar chart race plot... And Done!</p>', unsafe_allow_html=True)   
                from datetime import date
                data['time_column'] =pd.to_datetime(data[time_column])
                datast['value_column'] = data[value_column].astype(float)
     
                raceplot = barplot(data,  item_column=item_column, value_column=value_column, time_column=time_column,top_entries=num_items)
                fig=raceplot.plot(item_label = item_label, value_label = value_label, frame_duration = frame_duration, date_format=date_format,orientation=orientation)
                fig.update_layout(
                title=chart_title,
                autosize=False,
                width=chart_width,
                height=chart_height,
                paper_bgcolor="lightgray",
                )
                st.plotly_chart(fig, use_container_width=True)
        
if  choose  == "Prediction":
     st.header(" Saisir les donner d'entrées")
     PH = st.number_input("pH")
     T = st.number_input("Temp")
     CE = st.number_input("Con")
     O2 = st.number_input("O_diss")
     NH = st.number_input("N_NH")
     NO = st.number_input("NO")
     SO = st.number_input("SO")
     PO = st.number_input("PO")
     DBO5 = st.number_input("DBO5")
     val = [PH, T, CE,	O2,	NH,	NO,	SO,	PO,	DBO5]
     ypred = svc.predict([val])

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

if  choose == "Nos differents sites":

 if st.checkbox("NOS SITES"):
        st.write(data["Stations"])

if  choose  == "Map":

 st.line_chart(data.iloc[:,14])
 st.line_chart(data.iloc[:,12])
 st.line_chart(data.iloc[:,13])
 from bokeh.plotting import figure

 x = data.iloc[:,12]
 y = data.iloc[:,13]

 p = figure(
     title='latitude en fonction de la longitude',
     x_axis_label='longitude',
     y_axis_label='latitude')

 p.line(x, y, legend_label='Trend', line_width=2)

 st.bokeh_chart(p, use_container_width=True)
 z = data.iloc[:,12]
 y = data.iloc[:,14]

 k= figure(
     title='altitude en fonction de la longitude',
     x_axis_label='longitude',
     y_axis_label='altitude')

 k.line(z, y, legend_label='altitude', line_width=2)

 st.bokeh_chart(k, use_container_width=True)
 m = data.iloc[:,14]
 y = data.iloc[:,13]

 n= figure(
     title='altitude en fonction de la latitude',
     x_axis_label='latitude',
     y_axis_label='altitude')

 n.line(m, y, legend_label='altitude', line_width=2)

 st.bokeh_chart(n, use_container_width=True)
