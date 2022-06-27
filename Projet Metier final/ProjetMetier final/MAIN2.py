import pickle,pyodbc,time,datetime,base64,json,seaborn as sns,folium,requests,geopy,geocoder,numpy as np,pandas as pd,ipywidgets,streamlit as st ,matplotlib.pyplot as plt ,plotly.express as px,plotly.graph_objects as go
from fileinput import filename
from lib2to3.pytree import convert
from collections import Counter
from os import times
from statistics import mode
from unittest import result
from itsdangerous import TimestampSigner
from sklearn import metrics
from folium import plugins
from streamlit_folium import folium_static
from streamlit.type_util import Key
from streamlit_option_menu import option_menu
from streamlit_tags import st_tags
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap, MDS, SpectralEmbedding
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from streamlit_lottie import st_lottie,st_lottie_spinner
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn import svm, datasets,tree
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  plot_roc_curve, plot_precision_recall_curve,confusion_matrix,roc_curve,roc_auc_score,auc,precision_score,recall_score,confusion_matrix,ConfusionMatrixDisplay,classification_report,f1_score,accuracy_score
from sklearn.metrics import RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier ,AdaBoostClassifier,VotingClassifier 
from sklearn.multiclass import OneVsOneClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.linear_model import RidgeClassifier
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sympy import N

def report():
    st.title("Metrics Description")
    st.header('Precision:')
    st.write("Indicates the proportion of positive identifications (model predicted class 1) which were actually correct. A model which produces no false positives has a precision of 1.0.")
    st.header('Recall :')
    st.write("Indicates the proportion of actual positives which were correctly classified. A model which produces no false negatives has a recall of 1.0.")
    st.header('F1 score :')
    st.write("A combination of precision and recall. A perfect model achieves an F1 score of 1.0.")
    st.header('Support :')
    st.write("The number of samples each metric was calculated on.")
    st.header(' Accuracy :')
    st.write("The accuracy of the model in decimal form. Perfect accuracy is equal to 1.0")
    st.header('Macro avg :')
    st.write("Short for macro average, the average precision, recall and F1 score between classes. Macro avg doesn't class imbalance into effort, so if you do have class imbalances, pay attention to this metric.")
    st.header('Weighted avg :')
    st.write("Short for weighted average, the weighted average precision, recall and F1 score between classes. Weighted means each metric is calculated with respect to how many samples there are in each class. This metric will favour the majority class (e.g. will give a high vile when one class out performs another due to having more samples).")


def predict_input(model):
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
    ypred = model.predict([val])

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
    st.success(f"La qualité de ton eau est: {classe}")
#***************************insertion des donnees********************
connection = pyodbc.connect(r'DRIVER={ODBC Driver 17 for SQL Server};'
    r'SERVER=DESKTOP-PTQ7CTJ\SQLEXPRESS;'
    r'DATABASE=DataProjetMetier;'
    r'Trusted_Connection=yes;') 
cursor = connection.cursor()
def insert_data(nomModele,hyperpara,PRECISION,RECALL_SCORE,F1_SCORE,ACCURACY,TIMESTAMPS):
    DB_table_name = 'DUsers'
    insert_sql = "insert into " + DB_table_name + """
    values (?,?,?,?,?,?,?)"""
    rec_values = (nomModele,hyperpara,PRECISION,RECALL_SCORE,F1_SCORE,ACCURACY,TIMESTAMPS)
    cursor.execute(insert_sql, rec_values)
    connection.commit()
def get_table_download_link(df,filename,text):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    # href = f'<a href="data:file/csv;base64,{b64}">Download Report</a>'
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
def clean_data(x):  # enlever les symboles d'une colonne
    if isinstance(x, str):
        return x.replace('$', '').replace(',', '').replace('€', '').replace('£', '')
    return x
def map(pca):
        (fig, ax) = plt.subplots(figsize=(8, 8))
        for i in range(0, pca.components_.shape[1]):
             ax.arrow(0,
             0, # Start the arrow at the origin
             pca.components_[0, i], #0 for PC1
             pca.components_[1, i], #1 for PC2
             head_width=0.05,
             head_length=0.05,color='black')
         #plt.text(pca.components_[0, i] + 0.05, pca.components_[1, i] + 0.05, data.feature_names[i])
        an = np.linspace(0, 2 * np.pi, 100)
        plt.plot(np.cos(an), np.sin(an)) # Add a unit circle for scale
        plt.axis('equal')
        ax.set_title('Variable factor map')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        plt.show()
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
###### Load data #######
def load_data():
    try:
        if 'csv' in st.session_state.file_details['FileName']:
            if st.session_state.separateur != "":
                st.session_state.data = pd.read_csv(uploaded_file, sep=st.session_state.separateur, engine='python')
            else:
                st.session_state.data = pd.read_csv(uploaded_file)
        else:
            if st.session_state.separateur != "":
                st.session_state.data = pd.read_excel(uploaded_file, sep=st.session_state.separateur, engine='python')
            else:
                st.session_state.data = pd.read_excel(uploaded_file)
    except:
        pass
        
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


choose = option_menu("Main Menu",["Home","Data Preprocessing","Map","Regression","Classification","Rapport"],
    icons=['house','file','pin-map','graph-up','pie-chart-fill'],
     menu_icon = "list", default_index=0,
    styles={
        "container": {"padding": "5!important", "background-color": ""},
        "icon": {"color": "orange", "font-size": "18px"}, 
        "nav-link": {"font-size": "8px", "text-align": "left", "margin":"0px", "--hover-color": ""},
        "nav-link-selected": {"background-color": ""},
    },orientation = "horizontal"
    )
#*************rapport********
if choose =="Rapport":
    st.success("Welcome")
    # Display Data
    cursor.execute('''SELECT*FROM DUsers''')
    data = cursor.fetchall()
    st.header("**User's👨‍💻 Data**")
    df = pd.read_sql_query('''SELECT*FROM DUsers''',connection)
    st.dataframe(df)
    st.markdown(get_table_download_link(df,'MODELES.csv','Download Report'), unsafe_allow_html=True)
    cursor.execute('''SELECT*FROM DUsers''')
    data = cursor.fetchall()
    query = 'select * from DUsers;'
    plot_data = pd.read_sql(query, connection)
    ## Pie chart for predicted field recommendations
    labels = plot_data.ACCURACY.unique()
    print(labels)
    values = plot_data.ACCURACY.value_counts()
    print(values)
    st.subheader("📈 **ACCCURACY**")
    fig = px.pie(df, values=values, names=labels, title='Accuracy')
    st.plotly_chart(fig)
    labels = plot_data.PRECISION.unique()
    values = plot_data.PRECISION.value_counts()
    st.subheader("📈 ** Pie-Chart for modele's👨‍💻 precision**")
    fig = px.pie(df, values=values, names=labels, title="Pie-Chart📈 for modele's👨‍💻 precision")
    st.plotly_chart(fig)
    ### Pie chart for User's👨‍💻 Experienced Level
    labels = plot_data.hyperpara.unique()
    values = plot_data.hyperpara.value_counts()
    st.subheader("📈 ** Pie-Chart for modele's👨‍💻 hyperparameters**")
    fig = px.pie(df, values=values, names=labels, title="Pie-Chart📈 for modele's👨‍💻 hyperparametres")
    st.plotly_chart(fig)
# Session
if "col_to_time" not in st.session_state:
    st.session_state.col_to_time = ""
if "drop_col" not in st.session_state:
    st.session_state.drop_col = ""
if "col_to_float_money" not in st.session_state:
    st.session_state.col_to_float_money = ""
if "col_to_float_coma" not in st.session_state:
    st.session_state.col_to_float_coma = ""
if "separateur" not in st.session_state:
    st.session_state.separateur = ""
if "slider_col" not in st.session_state:
    st.session_state.slider_col = ""
if "degres" not in st.session_state:
    st.session_state.degres = ""
if "file_details" not in st.session_state:
    st.session_state.file_details = ""

#************************************************************************************************************
if choose == "Data Preprocessing":
    # Pages principales
    PAGES = ["Accueil", "Dataset", "Analyse des colonnes", "Matrice de corrélations", "Section graphiques",
            "Régressions", "Classifications", "Réduction de dimension"]
    with st.sidebar:
        st_lottie(load_lottieurl('https://assets8.lottiefiles.com/packages/lf20_jjojhxyb.json'), height=150)
    choix_page = st.sidebar.radio(label="", options=PAGES)
    ############# Page 2 #############
    if choix_page == 'Dataset':
        st.markdown('<p class="grand_titre">Chargement du dataset</p>', unsafe_allow_html=True)
        loti("C:/Users/DIASSANA/Desktop/ProjetMetier/Projet Metier final/ProjetMetier final/datanalysis.json")
        st.write('##')
        col1_1, b_1, col2_1 = st.columns((1, 0.1, 1))
        col1, b, col2 = st.columns((2.7, 0.2, 1))
        with col2_1:
            st_lottie(load_lottieurl('https://assets9.lottiefiles.com/packages/lf20_zidar9jm.json'), height=200)
        with col1_1:
            dataset_choix = st.selectbox("Dataset",
                                        ["-- Choisissez une option --",
                                        "Choisir un dataset personnel","RIEN"] )
            message_ = st.empty()

        if 'choix_dataset' in st.session_state:
            with col1_1:
                message_.success(st.session_state.choix_dataset)

        if dataset_choix == "-- Choisissez une option --":
            if 'choix_dataset' in st.session_state:
                if st.session_state.choix_dataset == "Vous avez choisi de selectionner votre dataset" and 'data' in st.session_state:
                    with col1:
                        st.write("##")
                        st.markdown('<p class="section">Aperçu</p>', unsafe_allow_html=True)
                        st.write(st.session_state.data.head(50))
                        st.write("##")

                    with col2:
                        st.write("##")
                        st.markdown('<p class="section">Caractéristiques</p>', unsafe_allow_html=True)
                        st.write(' - Taille:', st.session_state.data.shape)
                        st.write(' - Nombre de valeurs:', st.session_state.data.shape[0] * st.session_state.data.shape[1])
                        st.write(' - Type des colonnes:', st.session_state.data.dtypes.astype(str).value_counts())
                        st.write(' - Pourcentage de valeurs manquantes:', round(
                            sum(pd.DataFrame(st.session_state.data).isnull().sum(axis=1).tolist()) * 100 / (
                                    st.session_state.data.shape[0] * st.session_state.data.shape[1]), 2),
                                ' % (', sum(pd.DataFrame(st.session_state.data).isnull().sum(axis=1).tolist()), ')')

                    col1_modif, col2_modif = st.columns((2, 1))
                    with col1_modif:
                        st.write("##")
                        st.info("Rechargez votre dataset pour le modifier, en cliquant 2 FOIS sur le bouton ci-dessous")
                    if st.button("Modifier le dataset"):
                        dataset_choix = ""
                        st.session_state.choix_dataset = ""
                        st.session_state.clear()


        if dataset_choix == "Choisir un dataset personnel":

            with col1_1:
                uploaded_file = st.file_uploader("", type=['csv', 'xls'])
                # uploaded_file = 0
                if uploaded_file is not None:
                    st.session_state.file_details = {"FileName": uploaded_file.name,
                                                    "FileType": uploaded_file.type,
                                                    "FileSize": uploaded_file.size}
                    st.success('Fichier ' + st.session_state.file_details['FileName'] + ' chargé avec succès !')

            if 'data' in st.session_state:
                del st.session_state.data

            if uploaded_file == None:
                with col1_1:
                    st.info("Veuillez charger un dataset")

            if "data" not in st.session_state:
                load_data()

            if "data" in st.session_state:
                my_expander = st.expander(label="Options de preprocessing")
                with my_expander:
                    with col1_1:
                        st.write("##")
                        st.write("##")
                        st.write("##")
                        st.session_state.separateur = st.text_input("Séparateur (optionnel): ")
                    st.write("##")

                    load_data()

                    st.markdown("<p class='petite_section'>Modifications du dataset : </p>", unsafe_allow_html=True)
                    col1_1, b_1, col2_1, c_1, col3_1 = st.columns((1, 0.2, 1, 0.2, 1))  # pour time series
                    st.write("##")
                    option_col_update = st.session_state.data.columns.tolist()

                    with col1_1:
                        st.session_state.col_to_time = st.multiselect(label='Conversion Time Series',
                                                                    options=option_col_update,
                                                                    )
                    with col2_1:
                        st.session_state.col_to_float_money = st.multiselect('Conversion Monnaies',
                                                                            options=option_col_update,
                                                                            )
                    with col3_1:
                        st.session_state.col_to_float_coma = st.multiselect('Conversion string avec virgules vers float',
                                                                            options=option_col_update,
                                                                            )
                    with col1_1:
                        st.session_state.drop_col = st.multiselect(label='Drop columns',
                                                                options=option_col_update,
                                                                )

                    with col1_1:
                        for col in st.session_state["col_to_time"]:
                            try:
                                st.session_state.data[col] = pd.to_datetime(st.session_state.data[col])
                                st.success("Transformation de " + col + " effectuée !")
                            except:
                                st.error("Transformation impossible ou déjà effectuée")
                    with col2_1:
                        for col in st.session_state.col_to_float_money:
                            try:
                                st.session_state.data[col] = st.session_state.data[col].apply(clean_data).astype('float')
                                st.success("Transformation de " + col + " effectuée !")
                            except:
                                st.error("Transformation impossible ou déjà effectuée")
                    with col3_1:
                        for col in st.session_state.col_to_float_coma:
                            try:
                                st.session_state.data[col] = st.session_state.data[col].apply(
                                    lambda x: float(str(x).replace(',', '.')))
                                st.success("Transformation de " + col + " effectuée !")
                            except:
                                st.error("Transformation impossible ou déjà effectuée")
                    with col1_1:
                        for col in st.session_state["drop_col"]:
                            try:
                                st.session_state.data = st.session_state.data.drop(columns=col, axis=1)
                                st.success("Colonnes " + col + " supprimée !")
                            except:
                                st.error("Transformation impossible ou déjà effectuée")
                    with col1:
                        st.write("##")
                        st.markdown('<p class="section">Aperçu</p>', unsafe_allow_html=True)
                        st.write(st.session_state.data.head(50))
                        st.write("##")
                    with col2:
                        st.write("##")
                        st.markdown('<p class="section">Caractéristiques</p>', unsafe_allow_html=True)
                        st.write(' - Taille:', st.session_state.data.shape)
                        st.write(' - Nombre de valeurs:', st.session_state.data.shape[0] * st.session_state.data.shape[1])
                        st.write(' - Type des colonnes:', st.session_state.data.dtypes.astype(str).value_counts())
                        st.write(' - Pourcentage de valeurs manquantes:', round(
                            sum(pd.DataFrame(st.session_state.data).isnull().sum(axis=1).tolist()) * 100 / (
                                    st.session_state.data.shape[0] * st.session_state.data.shape[1]), 2),
                                ' % (', sum(pd.DataFrame(st.session_state.data).isnull().sum(axis=1).tolist()), ')')
                    st.download_button(data=st.session_state.data.to_csv(), label="Télécharger le dataset modifié",
                                    file_name='dataset.csv')
            st.session_state.choix_dataset = "Vous avez choisi de selectionner votre dataset"
            with col1_1:
                message_.success(st.session_state.choix_dataset)
    ############# Page 3 #############
    elif choix_page == "Analyse des colonnes":
        st.markdown('<p class="grand_titre">Analyse des colonnes</p>', unsafe_allow_html=True)
        st.write('##')
        if 'data' in st.session_state:
            options = st.session_state.data.columns.to_list()
            st.session_state.slider_col = st.multiselect(
                'Selectionner une ou plusieurs colonnes',
                options, help="Choisissez les colonnes à analyser"
            )
            if st.session_state.slider_col:
                col1, b, col2, c = st.columns((1.1, 0.1, 1.1, 0.3))
                with col1:
                    st.write('##')
                    st.markdown('<p class="section">Aperçu</p>', unsafe_allow_html=True)
                with col2:
                    st.write('##')
                    st.markdown('<p class="section">Caractéristiques</p>', unsafe_allow_html=True)
                for col in st.session_state.slider_col:
                    ### Données ###
                    data_col = st.session_state.data[col].copy()
                    n_data = st.session_state.data[col].to_numpy()

                    st.write('##')
                    col1, b, col2, c = st.columns((1, 1, 2, 0.5))
                    with col1:
                        st.markdown('<p class="nom_colonne_page3">' + col + '</p>', unsafe_allow_html=True)
                        st.write(data_col.head(20))
                    with col2:
                        st.write('##')
                        st.write(' ● type de la colonne :', type(data_col))
                        st.write(' ● type des valeurs :', type(data_col.iloc[1]))
                        if n_data.dtype == float:
                            moyenne = data_col.mean()
                            variance = data_col.std()
                            max = data_col.max()
                            min = data_col.min()
                            st.write(' ● Moyenne :', round(moyenne, 3))

                            st.write(' ● Variance :', round(variance, 3))

                            st.write(' ● Maximum :', max)

                            st.write(' ● Minimum :', min)

                        st.write(' ● Valeurs les plus présentes:', (Counter(n_data).most_common()[0])[0], 'apparait',
                                (Counter(n_data).most_common()[0])[1], 'fois', ', ', (Counter(n_data).most_common()[1])[0],
                                'apparait',
                                (Counter(n_data).most_common()[1])[1], 'fois')

                        st.write(' ● Nombre de valeurs manquantes:',
                                sum(pd.DataFrame(n_data).isnull().sum(axis=1).tolist()))

                        st.write(' ● Longueur:', n_data.shape[0])

                        st.write(' ● Nombre de valeurs différentes non NaN:',
                                abs(len(Counter(n_data)) - sum(pd.DataFrame(n_data).isnull().sum(axis=1).tolist())))
                        ### Fin section données ###
                    st.write('##')

        else:
            st.info("Veuillez charger vos données dans la section Dataset")
            st.write("##")
            st_lottie(load_lottieurl('https://assets5.lottiefiles.com/packages/lf20_inuxiflu.json'), height=200)
    ############# Page 3 #############

    ############# Page 4 #############
    elif choix_page == "Matrice de corrélations":
        st.markdown('<p class="grand_titre">Matrice de corrélations</p>', unsafe_allow_html=True)
        st.write("##")
        if 'data' in st.session_state:
            col1, b, col2 = st.columns((1, 1, 2))
            df_sans_NaN = st.session_state.data
            with col1:
                st.session_state.couleur_corr = st.selectbox('Couleur',
                                                            [
                                                                '-- Selectionner une colonne --'] + df_sans_NaN.columns.tolist(),
                                                            help="Choisissez la variable catégorielle pour la coloration des classes"
                                                            )
                st.write("##")
            st.session_state.select_columns_corr = st.multiselect("Choisir au moins deux colonnes",
                                                                ["Toutes les colonnes"] + col_numeric(df_sans_NaN),
                                                                help="Choisissez vos features"
                                                                )
            if len(st.session_state.select_columns_corr) > 1 and "Toutes les colonnes" not in st.session_state.select_columns_corr and st.session_state.couleur_corr != "-- Selectionner une colonne --":
                df_sans_NaN = pd.concat([st.session_state.data[col] for col in st.session_state.select_columns_corr],
                                        axis=1).dropna()
                if len(df_sans_NaN) == 0:
                    st.write("##")
                    st.warning('Le dataset avec suppression des NaN suivant les lignes est vide!')
                else:
                    if st.session_state.couleur_corr != '-- Selectionner une colonne --':
                        fig = px.scatter_matrix(st.session_state.data,
                                                dimensions=col_numeric(df_sans_NaN[st.session_state.select_columns_corr]),
                                                color=st.session_state.couleur_corr, color_continuous_scale='Bluered_r')
                    else:
                        fig = px.scatter_matrix(df_sans_NaN,
                                                dimensions=col_numeric(df_sans_NaN[st.session_state.select_columns_corr]))
                    fig.update_layout(width=1000, height=700, margin=dict(l=40, r=50, b=40, t=40), font=dict(size=10))
                    fig.update_layout({"xaxis" + str(i + 1): dict(showticklabels=False) for i in
                                    range(len(col_numeric(df_sans_NaN[st.session_state.select_columns_corr])))})
                    fig.update_layout({"yaxis" + str(i + 1): dict(showticklabels=False) for i in
                                    range(len(col_numeric(df_sans_NaN[st.session_state.select_columns_corr])))})
                    fig.update_traces(marker=dict(size=7))
                    fig.update_traces(diagonal_visible=False)
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig)
            elif st.session_state.select_columns_corr == [
                "Toutes les colonnes"] and st.session_state.couleur_corr != "-- Selectionner une colonne --":
                df_sans_NaN = st.session_state.data.dropna()
                if len(df_sans_NaN) == 0:
                    st.write("##")
                    st.warning('Le dataset avec suppression des NaN suivant les lignes est vide!')
                else:
                    if st.session_state.couleur_corr != 'Selectionner une colonne':
                        fig = px.scatter_matrix(df_sans_NaN, dimensions=col_numeric(df_sans_NaN),
                                                color=st.session_state.couleur_corr)
                    else:
                        fig = px.scatter_matrix(df_sans_NaN, dimensions=col_numeric(df_sans_NaN))
                    fig.update_layout(
                        {"xaxis" + str(i + 1): dict(showticklabels=False) for i in range(len(col_numeric(df_sans_NaN)))})
                    fig.update_layout(
                        {"yaxis" + str(i + 1): dict(showticklabels=False) for i in range(len(col_numeric(df_sans_NaN)))})
                    fig.update_traces(marker=dict(size=2))
                    fig.update_layout(width=1000, height=700, margin=dict(l=40, r=50, b=40, t=40), font=dict(size=10))
                    fig.update_traces(marker=dict(size=7))
                    fig.update_traces(diagonal_visible=False)
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig)
            elif len(
                    st.session_state.select_columns_corr) > 1 and "Toutes les colonnes" in st.session_state.select_columns_corr:
                st.error("Erreur de saisi !")
            else:
                pass
        else:
            st.info("Veuillez charger vos données dans la section Dataset")
            st.write("##")
            st_lottie(load_lottieurl('https://assets5.lottiefiles.com/packages/lf20_inuxiflu.json'), height=200)
    ############# Page 4 #############              
               
if choose == "Map":
           
           loti("C:/Users/DIASSANA/Desktop/ProjetMetier/Projet Metier final/ProjetMetier final/99797-data-management.json")
           st.title("Ci-dessous une vue des differentes stations du fleuve MOULOUYA")
           markers_dict = {"Ait boulmane": [ 31.0 , -7.1], 
                            "Source Arbalou": [31.2911, -9.2391], 
                            "Boumia": [32.7253, -5.1018], 
                            "Zaïda": [32.8026, -4.8709], 
                            "AnzarOufounas": [33.1633, -5.1638],
                            "Tamdafelt":[32.8722, -4.2564], 
                            "Missour":[33.0471, -3.9926],  
                            "Sebra":[34.826,-1.5303],
                            "Safsaf":[34.9139,-2.6225], 
                            "Pont Hassan II":[34.0254,-6.8222],
                            "Moulouya":[35.1092,-2.3578]}
           # create map
           map_cities = folium.Map(location=[31.7, -11.6],zoom_start=4)
           # plot locations
           for i in markers_dict.items():
               folium.Marker(location=i[1], popup=i[0]).add_to(map_cities)
               print(i)
            # display map
           folium_static(map_cities)           
  
                
if choose == "Regression":
    loti("C:/Users/DIASSANA/Desktop/ProjetMetier/Projet Metier final/ProjetMetier final/98831-pie-chart.json")
    df = pd.read_csv("C:/Users/DIASSANA/Desktop/ProjetMetier/Projet Metier final/ProjetMetier final/dataset.csv")
    #y=df.iloc[0:len(df),1] 
    x=np.arange(0,66)
    
    with st.sidebar:
        st.title("Data Visualization")
        rd = st.radio("Choose",["Visualiszation","Regression"])
    
    if rd =="Visualiszation":
        rd1,rd2 = st.columns(2)
        a = rd1.selectbox("choose features?", ["PH", "T", "CE", "O2", "NH", "NO", "SO", "PO", "DBO5"])
        g = rd2.selectbox("Choose graph",["boxplot","distplot"])
        if a=="PH":
            if g=="boxplot":
                fig, ax = plt.subplots()
                sns.boxplot(x="Classe", y="PH", data=df)
                st.pyplot(fig)
            if g=="distplot":
                fig, ax = plt.subplots()
                sns.distplot(df["PH"])
                st.pyplot(fig)

        if a=="T":
            if g=="boxplot":
                fig, ax = plt.subplots()
                sns.boxplot(x="Classe", y="T", data=df)
                st.pyplot(fig)
            if g=="distplot":
                fig, ax = plt.subplots()
                sns.distplot(df["T"])
                st.pyplot(fig)

        if a=="CE":
            if g=="boxplot":
                fig, ax = plt.subplots()
                sns.boxplot(x="Classe", y="CE", data=df)
                st.pyplot(fig)
            if g=="distplot":
                fig, ax = plt.subplots()
                sns.distplot(df["CE"])
                st.pyplot(fig)
        if a=="O2":
            if g=="boxplot":
                fig, ax = plt.subplots()
                sns.boxplot(x="Classe", y="O2", data=df)
                st.pyplot(fig)
            if g=="distplot":
                fig, ax = plt.subplots()
                sns.distplot(df["o2"])
                st.pyplot(fig)
        if a=="NH":
            if g=="boxplot":
                fig, ax = plt.subplots()
                sns.boxplot(x="Classe", y="NH", data=df)
                st.pyplot(fig)
            if g=="distplot":
                fig, ax = plt.subplots()
                sns.distplot(df["NH"])
                st.pyplot(fig)

        if a=="NO":
            if g=="boxplot":
                fig, ax = plt.subplots()
                sns.boxplot(x="Classe", y="NO", data=df)
                st.pyplot(fig)
            if g=="distplot":
                fig, ax = plt.subplots()
                sns.distplot(df["NO"])
                st.pyplot(fig)

        if a=="SO":
            if g=="boxplot":
                fig, ax = plt.subplots()
                sns.boxplot(x="Classe", y="SO", data=df)
                st.pyplot(fig)
            if g=="distplot":
                fig, ax = plt.subplots()
                sns.distplot(df["SO"])
                st.pyplot(fig)
        if a=="PO":
            if g=="boxplot":
                fig, ax = plt.subplots()
                sns.boxplot(x="Classe", y="PO", data=df)
                st.pyplot(fig)
            if g=="distplot":
                fig, ax = plt.subplots()
                sns.distplot(df["PO"])
                st.pyplot(fig)
        if a=="DBO5":
            if g=="boxplot":
                fig, ax = plt.subplots()
                sns.boxplot(x="Classe", y="DBO5", data=df)
                st.pyplot(fig)
            if g=="distplot":
                fig, ax = plt.subplots()
                sns.distplot(df["DBO5"])
                st.pyplot(fig)



    def regression (y):
            polynomial_features= PolynomialFeatures(degree=6) # polynomial degree
            x=np.arange(0,66)
            x=x.reshape(-1, 1)
            x_poly = polynomial_features.fit_transform(x)
            model = LinearRegression()
            model.fit(x_poly, y)  # apprentissage
            ypred=model.predict(x_poly)
            fig, ax = plt.subplots()
            plt.plot(x, ypred, color='red', ls="-")
            plt.scatter(x, y,  marker ="*", color ="blue",s=50)
            plt.show()
            st.pyplot(fig)
            
            rmse = np.sqrt(mean_squared_error(y,ypred))
            r2 = r2_score(y,ypred)
            colrmse1,colrmse2=st.columns(2)
            colrmse1.title("RMSE")
            colrmse2.title(rmse)
            colr1,colr2=st.columns(2)
            colr1.title("R2")
            colr2.title(r2)
    if rd =="Regression":
        graph1 = st.selectbox("What kind of graph?", ["PH", "T", "CE", "O2", "NH", "NO", "SO", "PO", "DBO5"])
        if graph1=="PH":
            regression (df.iloc[0:len(df),1])       
        if graph1=="T":
            regression (df.iloc[0:len(df),2])
        if graph1=="CE":
            regression (df.iloc[0:len(df),3])       
        if graph1=="O2":
            regression (df.iloc[0:len(df),4])
        if graph1=="NH":
            regression (df.iloc[0:len(df),5])       
        if graph1=="NO":
            regression (df.iloc[0:len(df),6])
        if graph1=="SO":
            regression (df.iloc[0:len(df),7])       
        if graph1=="PO":
            regression (df.iloc[0:len(df),8])
        if graph1=="DBO5":
            regression (df.iloc[0:len(df),9])
acc="";recall="";precision="";nomModele="";hyperpara="";f1=""
if choose == "Classification":
    loti("C:/Users/DIASSANA/Desktop/ProjetMetier/Projet Metier final/ProjetMetier final/98831-pie-chart.json")
    with st.sidebar:
        st.title("Classification")
        model = st.selectbox("Choose Model",["SVC","RandomForest","Tree","KNN","Voting","Bagging","Perceptron","MLPClassifier","RidgeClassifier"])
        # col3,col4= st.columns(2)
        # gridsearch = col3.checkbox("GridSearch")
        # randomsearch = col4.checkbox("RandomSearch")
        with st.sidebar:
            
             option = st.radio("Navigation",["GridSearch","RandomSearch","Predict"])

        # st.title("Prediction")
        # kind = st.checkbox("One Predict")
    data = pd.read_csv("C:/Users/DIASSANA/Desktop/ProjetMetier/Projet Metier final/ProjetMetier final/dataset.csv")
    x = data.iloc[:,1:10]
    y= data.iloc[:,11]
    x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
    if option=="GridSearch":

            if model == "SVC":
                global svc
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
                name.header("Kernel :")
                kernel = last.multiselect("Choose Kernel(s)",["sigmoid","poly","rbf"])
                #st.write(kernel)
                v1,v2 = st.columns(2)
                valider = v1.checkbox("Valider")        
                save = v2.button("Save model")       
                if valider:
                    params = {"C":np.arange(Min_value_C,Max_value_C,0.1),"gamma":np.arange(Min_value_g,Max_value_g,0.1),"kernel":kernel}
                    grid = GridSearchCV(svc,params)
                    grid.fit(x_train,y_train)
                    hyper=grid.best_params_
                    st.write(hyperpara)
                    svc = grid.best_estimator_
                    result = st.selectbox("choose",["Metrics","CURVE","precision_score","SAVE"])
                    if result=="Metrics":
                        st.text('Model Report:\n ' + classification_report(y_test, svc.predict(x_test)))
                        precision=precision_score(y_test, svc.predict(x_test) , average='macro')
                        recall=recall_score(y_test, svc.predict(x_test) , average='weighted')
                        acc=accuracy_score(y_test, svc.predict(x_test))
                        f1=f1_score(y_test, svc.predict(x_test) , average='weighted')
                        report()
                    if result=="precision_score":
                        st.text('Model precisiion:\n'+ str(precision))
                    if result=="CURVE": 
                        
                        y_test = y_test.to_numpy()
                        st.subheader("Confusion Matrix")  
                        predictions = svc.predict(x_test)
                        cm = confusion_matrix(y_test, predictions, labels=svc.classes_)
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=svc.classes_)
                            #plot_confusion_matrix(svc, x_test, y_test, display_labels=class_names)
                        disp.plot()
                        plt.show()
                        st.pyplot()
                    if result=="SAVE":
                        filename = "svc"
                        pickle.dump(svc,open(filename,"wb"))
                        st.write("model saved")
                        ts = time.time()
                        cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                        cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                        timestamp = str(cur_date+'heure'+cur_time)
                        nomModele=filename + timestamp
                        hyperpara=str(hyper)
                        insert_data(nomModele,hyperpara,str(precision),str(recall),str(f1),str(acc),timestamp)
                        st.balloons()        
                if save:
                    st.write(hyperpara)
                    filename = "svc"
                    pickle.dump(svc,open(filename,"wb"))
                    st.write("model saved")
                    ts = time.time()
                    cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    timestamp = str(cur_date+'##'+cur_time)
                    metri=classification_report(y_test, svc.predict(x_test))
                    nomModele= filename + timestamp
                    hyperpara=str(hyper)
                    insert_data(nomModele,hyperpara,str(precision),str(recall),str(f1),str(acc),timestamp)
                    st.balloons()
                # with st.sidebar:
                    
            if model == "KNN":
                global mknn
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
                v1,v2 = st.columns(2)

                valider = v1.checkbox("Valider")
                
                save = v2.button("Save model")
                if valider:
                    params = {"n_neighbors":np.arange(int(k),int(kmax),1),"weights":l,"algorithm":ac}
                    grid = GridSearchCV(mknn,params)
                    grid.fit(x_train,y_train)
                    grid.best_params_
                    mknn = grid.best_estimator_
                    result = st.selectbox("choose",["Metrics","CURVE"])
                    if result=="Metrics":
                        st.text('Model Report:\n ' + classification_report(y_test, mknn.predict(x_test)))
                        report()
                    if result=="CURVE": 
                        
                        y_test = y_test.to_numpy()
                        st.subheader("Confusion Matrix") 
                        predictions = mknn.predict(x_test)
                        cm = confusion_matrix(y_test, predictions, labels=mknn.classes_)
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=mknn.classes_)
                        #plot_confusion_matrix(svc, x_test, y_test, display_labels=class_names)
                        disp.plot()
                        plt.show()
                        st.pyplot()
                if save:
                    filename = "mknn"
                    pickle.dump(mknn,open(filename,"wb"))
                    st.balloons()
                    st.write("model saved")

            if model == "Tree":
                global mtree
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
                v1,v2 = st.columns(2)

                valider = v1.checkbox("Valider")
                
                save = v2.button("Save model")
                if valider:
                    params = {"min_impurity_decrease":np.arange(t,tmax,0.1),"min_samples_split":np.arange(tr,tre,0.1),"min_samples_leaf":np.arange(trs,tres,0.1),"ccp_alpha":np.arange(trf,trf2,0.1),"max_features":trfi}
                    grid = GridSearchCV(mtree,params)
                    grid.fit(x_train,y_train)
                    grid.best_params_
                    mtree = grid.best_estimator_
                    result = st.selectbox("choose",["Metrics","CURVE"])
                    if result=="Metrics":
                        st.text('Model Report:\n ' + classification_report(y_test, mtree.predict(x_test)))
                        report()

                    if result=="CURVE": 
                        
                        y_test = y_test.to_numpy()
                        st.subheader("Confusion Matrix") 
                        predictions = mtree.predict(x_test)
                        cm = confusion_matrix(y_test, predictions, labels=mtree.classes_)
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=mtree.classes_)
                        #plot_confusion_matrix(svc, x_test, y_test, display_labels=class_names)
                        disp.plot()
                        plt.show()
                        st.pyplot()
                if save:
                    filename = "mtree"
                    pickle.dump(mtree,open(filename,"wb"))
                    st.write("model saved")
                    st.balloons()
            if model == "RandomForest":
                global raf
                x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
                raf = RandomForestClassifier(n_estimators=100)
                raf.fit(x_train, y_train)

                st.header("RandomForest hyperParams")
                tcl1,tcl2,tcl3=st.columns(3)
                
                tcl1.header("n_estimators")
                ta=tcl2.number_input("choose min value ",min_value=10, max_value=100, value=10)
                tmaxa=tcl3.number_input("choose max  value ",min_value=10, max_value=100, value=20)
                trl1,trcl2,trcl3=st.columns(3)
                # trl1.header("max_depth")
                # trsa= trcl2.number_input("choose mini value ")
                # tresa= trcl3.number_input("choose maxi value ")
                trls1,trcls2,trcls3=st.columns(3)
                trls1.header("min_split")
                tra= trcls2.number_input("choose min value split ")
                trea= trcls3.number_input("choose max value split ")
                tr1,tr2,tr3=st.columns(3)
                tr1.header("ccp alpha")
                trfa= tr2.number_input("choose min value ccp ")
                trf2a= tr3.number_input("choose max value ccp")
                tri1,tri2=st.columns(2)
                tri1.header("max features")
                trfia= tri2.multiselect("Choose one ",["auto", "sqrt", "log2","None"] )
                v1,v2 = st.columns(2)

                valider = v1.checkbox("Valider")
                
                save = v2.button("Save model")
                if valider:
                    params = {"n_estimators":np.arange(int(ta),int(tmaxa)),"min_samples_leaf":np.arange(tra,trea,0.1),"ccp_alpha":np.arange(trfa,trf2a,0.1),"max_features":trfia}
                    grid = GridSearchCV(raf,params)
                    grid.fit(x_train,y_train)
                    grid.best_params_
                    raf = grid.best_estimator_
                    result = st.selectbox("choose",["Metrics","CURVE"])
                    if result=="Metrics":
                        st.text('Model Report:\n ' + classification_report(y_test, raf.predict(x_test)))
                        report()

                    if result=="CURVE": 
                        
                        y_test = y_test.to_numpy()
                        st.subheader("Confusion Matrix")  
                        predictions = raf.predict(x_test)
                        cm = confusion_matrix(y_test, predictions, labels=raf.classes_)
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=raf.classes_)
                    #plot_confusion_matrix(svc, x_test, y_test, display_labels=class_names)
                        disp.plot()
                        plt.show()
                        st.pyplot()
                if save:
                    filename = "raf"
                    pickle.dump(raf,open(filename,"wb"))
                    st.balloons()
                    st.write("model saved")
    

            if model == "Perceptron":
                x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
                percep = Perceptron()
                percep.fit(x_train, y_train)
                percep.score(x_test,y_test)
                st.header("Perceptron Params")
                kcl1,kcl2,kcl3=st.columns(3)
            
                kcl1.header("n_iter_no_change")
                k=kcl2.number_input("choose min n_iter")
                kmax=kcl3.number_input("choose max n_iter")
                acl1,acl2=st.columns(2)
                v1,v2 = st.columns(2)

                valider = v1.checkbox("Valider")
                
                save = v2.button("Save model")
                if valider:
                    params = {"n_iter_no_change":np.arange(k,kmax,1)}
                    grid = GridSearchCV(percep,params)
                    grid.fit(x_train,y_train)
                    grid.best_params_
                    percep = grid.best_estimator_
                    result = st.selectbox("choose",["Metrics","CURVE"])
                    if result=="Metrics":
                        st.text('Model Report:\n ' + classification_report(y_test, percep.predict(x_test)))
                    if result=="CURVE": 
                    
                        y_test = y_test.to_numpy()
                        st.subheader("Confusion Matrix")  
                        predictions = percep.predict(x_test)
                        cm = confusion_matrix(y_test, predictions, labels=percep.classes_)
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=percep.classes_)
                    #plot_confusion_matrix(svc, x_test, y_test, display_labels=class_names)
                        disp.plot()
                        plt.show()
                        st.pyplot()
                if save:
                    filename = "percep"
                    pickle.dump(percep,open(filename,"wb"))
                    st.write("model saved")  
                    st.balloons() 


            if model == "MLPClassifier":
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
                 v1,v2 = st.columns(2)

                 valider = v1.checkbox("Valider")
                 save = v2.button("Save model")
                 if valider:

                     params = {'max_iter': [mlll ], 'alpha': np.arange(ml, mll), 'hidden_layer_sizes':np.arange(mlp22, mlp23), 'random_state':[0,1,2,3,4,5,6,7,8,9]}
                     grid = GridSearchCV(mlp, params)
                     grid.fit(x_train,y_train)
                     grid.best_params_
                     mlp = grid.best_estimator_
                     #rcol1 = st.columns(1)
                     #rcol1.header("Score")
                     #st.write(mlp.score(x_test,y_test))
                     result = st.selectbox("choose",["Metrics","CURVE"])
                     if result=="Metrics":
                         st.text('Model Report:\n ' + classification_report(y_test, mlp.predict(x_test)))
                     if result=="CURVE": 
                        
                         y_test = y_test.to_numpy()
                         st.subheader("Confusion Matrix")  
                         predictions = mlp.predict(x_test)
                         cm = confusion_matrix(y_test, predictions, labels=mlp.classes_)
                         disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=mlp.classes_)
                        #plot_confusion_matrix(svc, x_test, y_test, display_labels=class_names)
                         disp.plot()
                         plt.show()
                         st.pyplot()
                 if save:
                    filename = "mlp"
                    pickle.dump(mlp,open(filename,"wb"))
                    st.balloons()
                    st.write("model saved") 
            if model =="RidgeClassifier":
         
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
                 v1,v2 = st.columns(2)

                 valider = v1.checkbox("Valider")
                 save = v2.button("Save model")
                 if valider:

                     params = {'max_iter': [ridg], 'alpha': np.arange(r1, r2), 'random_state':[0,1,2,3,4,5,6,7,8,9]}
                     grid = GridSearchCV(ridge, params)
                     grid.fit(x_train,y_train)
                     grid.best_params_
                     ridge = grid.best_estimator_
                
                     st.write(ridge.score(x_test,y_test))
                     result = st.selectbox("choose",["Metrics","CURVE"])
                     if result=="Metrics":
                         st.text('Model Report:\n ' + classification_report(y_test, ridge.predict(x_test)))
                     if result=="CURVE": 
                        
                         y_test = y_test.to_numpy()
                         st.subheader("Confusion Matrix")  
                         predictions = ridge.predict(x_test)
                         cm = confusion_matrix(y_test, predictions, labels=ridge.classes_)
                         disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=ridge.classes_)
                        #plot_confusion_matrix(svc, x_test, y_test, display_labels=class_names)
                         disp.plot()
                         plt.show()
                         st.pyplot()
                     if save:
                         filename = "ridge"
                         pickle.dump(ridge,open(filename,"wb"))
                         st.balloons()   
            if model =="AdaBoostClassifier ":
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
                 v1,v2 = st.columns(2)
                 valider = v1.checkbox("Valider")
                 save = v2.button("Save model")
                 if valider:

                     params = {'algorithm': [adb], 'Numbers estimators': [addd], 'learning_rate': np.arange(ad, a), 'random_state':[0,1,2,3,4,5,6,7,8,9]}
                     grid = GridSearchCV(ada, params)
                     grid.fit(x_train,y_train)
                     grid.best_params_
                     ada= grid.best_estimator_
                     st.write(ada.score(x_test,y_test))
                     result = st.selectbox("choose",["Metrics","CURVE"])
                     if result=="Metrics":
                         st.text('Model Report:\n ' + classification_report(y_test,ada.predict(x_test)))
                     if result=="CURVE": 
                        
                         y_test = y_test.to_numpy()
                         st.subheader("Confusion Matrix")  
                         predictions =ada.predict(x_test)
                         cm = confusion_matrix(y_test, predictions, labels=ada.classes_)
                         disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=ada.classes_)
                        #plot_confusion_matrix(svc, x_test, y_test, display_labels=class_names)
                         disp.plot()
                         plt.show()
                         st.pyplot()
                     if save:
                         filename = "adaboost"
                         pickle.dump(ada,open(filename,"wb"))
                         st.write("model saved") 
            if model =="QuadraticDiscriminantAnalysis":
         
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
                 v1,v2 = st.columns(2)
                 valider = v1.checkbox("Valider")
                 save = v2.button("Save model")
                 if valider:

                     params = {'store_covariance': [qu], 'tol': np.arange(ql, qll), 'reg_param': np.arange(qlp22, qlp23)}
                     grid = GridSearchCV(qua, params)
                     grid.fit(x_train,y_train)
                     grid.best_params_
                     qua= grid.best_estimator_
                     st.write(qua.score(x_test,y_test))
                     result = st.selectbox("choose",["Metrics","CURVE"])
                     if result=="Metrics":
                         st.text('Model Report:\n ' + classification_report(y_test,qua.predict(x_test)))
                     if result=="CURVE": 
                        
                         y_test = y_test.to_numpy()
                         st.subheader("Confusion Matrix")  
                         predictions =qua.predict(x_test)
                         cm = confusion_matrix(y_test, predictions, labels=qua.classes_)
                         disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=qua.classes_)
                        #plot_confusion_matrix(svc, x_test, y_test, display_labels=class_names)
                         disp.plot()
                         plt.show()
                         st.pyplot()
                     if save:
                         filename = "QuadraticDiscriminantAnalysis"
                         pickle.dump(qua,open(filename,"wb"))
                         st.write("model saved") 
            if model =="GaussianMixture":
            
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
                 g11,g33,g44=st.columns(3)
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
                 v1,v2 = st.columns(2)
                 valider = v1.checkbox("Valider")
                 save = v2.button("Save model")
                 if valider:

                     params = {'max_iter': [glll], 'tol': np.arange(g3, g4l),'init_params': [gu]}
                     grid = GridSearchCV(gau, params)
                     grid.fit(x_train,y_train)
                     grid.best_params_
                     gau= grid.best_estimator_
                     st.write(gau.score(x_test,y_test))
                     result = st.selectbox("choose",["Metrics","CURVE"])
                     if result=="Metrics":
                         st.text('Model Report:\n ' + classification_report(y_test,gau.predict(x_test)))
                     if result=="CURVE": 
                        
                         y_test = y_test.to_numpy()
                         st.subheader("Confusion Matrix")  
                         predictions =gau.predict(x_test)
                         cm = confusion_matrix(y_test, predictions, labels=gau.classes_)
                         disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=gau.classes_)
                        #plot_confusion_matrix(svc, x_test, y_test, display_labels=class_names)
                         disp.plot()
                         plt.show()
                         st.pyplot()
                     if save:
                         filename = "GaussianMixture"
                         pickle.dump(gau,open(filename,"wb"))
                         st.write("model saved") 
            if model =="OneVsOneClassifier":
         
                 x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
                 svc = svm.SVC(C= 0.1, gamma=1, kernel='linear')
                 one= OneVsOneClassifier(svc)
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
                 v1,v2 = st.columns(2)
                 valider = v1.checkbox("Valider")
                 save = v2.button("Save model")
                 if valider:

                     params = {'estimator': [oee], 'n_jobs': np.arange(one1, one2)}
                     grid = GridSearchCV(one, params)
                     grid.fit(x_train,y_train)
                     grid.best_params_
                     one = grid.best_estimator_
                     st.write(one.score(x_test,y_test))
                     result = st.selectbox("choose",["Metrics","CURVE"])
                     if result=="Metrics":
                         st.text('Model Report:\n ' + classification_report(y_test,qua.predict(x_test)))
                     if result=="CURVE": 
                        
                         y_test = y_test.to_numpy()
                         st.subheader("Confusion Matrix")  
                         predictions =one.predict(x_test)
                         cm = confusion_matrix(y_test, predictions, labels=one.classes_)
                         disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=one.classes_)
                        #plot_confusion_matrix(svc, x_test, y_test, display_labels=class_names)
                         disp.plot()
                         plt.show()
                         st.pyplot()
                     if save:
                         filename = "one"
                         pickle.dump(one,open(filename,"wb"))
                         st.write("model saved")
            if model =="LabelPropagation ":
         
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
                 v1,v2 = st.columns(2)
                 valider = v1.checkbox("Valider")
                 save = v2.button("Save model")
                 if valider:

                     params = {"n_neighbors":np.arange(llp22,llp23,1),"gamma":np.arange(ll,lll,0.1),"kernel":[LL],"max_iter":[llll]}
                     grid = GridSearchCV(llp,params)
                     grid.fit(x_train,y_train)
                     grid.best_params_
                     llp = grid.best_estimator_
                     st.write(llp.score(x_test,y_test))
                     result = st.selectbox("choose",["Metrics","CURVE"])
                     if result=="Metrics":
                         st.text('Model Report:\n ' + classification_report(y_test,llp.predict(x_test)))
                     if result=="CURVE": 
                        
                         y_test = y_test.to_numpy()
                         st.subheader("Confusion Matrix")  
                         predictions =llp.predict(x_test)
                         cm = confusion_matrix(y_test, predictions, labels=llp.classes_)
                         disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=llp.classes_)
                        #plot_confusion_matrix(svc, x_test, y_test, display_labels=class_names)
                         disp.plot()
                         plt.show()
                         st.pyplot()
                     if save:
                         filename = "LabelPropagation"
                         pickle.dump(llp,open(filename,"wb"))
                         st.write("model saved")
            if model =="BayesianGaussianMixture ":
                 x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
                 gmm =BayesianGaussianMixture()
                 gmm.fit(x_train, y_train)
                 gmm.score(x_test,y_test)
                 st.header("BayesianGaussianMixture Params")
                 gml1,gml2,gml3=st.columns(3)
                
                 gml1.header("n_components")
                 gml=gml2.number_input("choose minimum")
                 gmll=gml3.number_input("choose maximum")
                 g111,g222=st.columns(2)
                 g111.header("covariance_type",)
                 gm= g222.multiselect("Choose type(s)",[ "spherical", "tied", "diag", "full"])
                 gmlp1,gmlp2,gmlp3=st.columns(3)
                
                 gmlp1.header("reg_covar")
                 gmlp22=gmlp2.number_input("Choose min n",min_value=1, max_value=10, value=1)
                 gmlp23=gmlp3.number_input("Choose max n",min_value=1, max_value=50000, value=1)
                 v1,v2 = st.columns(2)
                 valider = v1.checkbox("Valider")
                 save = v2.button("Save model")
                 if valider:

                     params = {'n_components': np.arange(gml, gmll),'covariance_type': [gm],'reg_covar': np.arange(gmlp22, gmlp23)}
                     grid =GridSearchCV(gmm, params)
                     grid.fit(x_train,y_train)
                     grid.best_params_
                     gmm= grid.best_estimator_
                     st.write(gmm.score(x_test,y_test))
                     result = st.selectbox("choose",["Metrics","CURVE"])
                     if result=="Metrics":
                         st.text('Model Report:\n ' + classification_report(y_test,gmm.predict(x_test)))
                     if result=="CURVE":                     
                         y_test = y_test.to_numpy()
                         st.subheader("Confusion Matrix")  
                         predictions =gmm.predict(x_test)
                         cm = confusion_matrix(y_test, predictions, labels=gmm.classes_)
                         disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=gmm.classes_)
                        #plot_confusion_matrix(svc, x_test, y_test, display_labels=class_names)
                         disp.plot()
                         plt.show()
                         st.pyplot()
                     if save:
                         filename = "BayesianGaussianMixture"
                         pickle.dump(gmm,open(filename,"wb"))
                         st.write("model saved") 

            if model == "Voting":
                svc = pickle.load(open("svc","rb"))
                mtree = pickle.load(open("mtree","rb"))
                mknn = pickle.load(open("mknn","rb"))

                model_final = VotingClassifier([('SVC',svc),('KNN',mknn),('TREE',mtree)])
                for model in (svc,mknn,mtree,model_final):
                    model.fit(x_train,y_train)
                    st.write(model.__class__.__name__ , model.score(x_test,y_test))   

                if st.button("save"):
                    filename = "model_final"
                    pickle.dump(model_final,open(filename,"wb"))
                    st.write("model saved")
                    st.balloons()
#************************RANDOMIZEDSEARCH****************************************************
    if option=="RandomSearch":
            if model == "SVC":
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
                name.header("Kernel :")
                kernel = last.multiselect("Choose Kernel(s)",["sigmoid","poly","rbf"])
                #st.write(kernel)
                v1,v2 = st.columns(2)

                valider = v1.checkbox("Valider")
                
                save = v2.button("Save model")
                
                if valider:
                    params = {"C":np.arange(Min_value_C,Max_value_C,0.1),"gamma":np.arange(Min_value_g,Max_value_g,0.1),"kernel":kernel}
                    grid = RandomizedSearchCV(svc,params)
                    grid.fit(x_train,y_train)
                    grid.best_params_
                    svc = grid.best_estimator_
                    result = st.selectbox("choose",["Metrics","CURVE"])
                    if result=="Metrics":
                        st.text('Model Report:\n ' + classification_report(y_test, svc.predict(x_test)))
                        report()
                    if result=="CURVE": 
                        
                        y_test = y_test.to_numpy()
                        st.subheader("Confusion Matrix")  
                        predictions = svc.predict(x_test)
                        cm = confusion_matrix(y_test, predictions, labels=svc.classes_)
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=svc.classes_)
                            #plot_confusion_matrix(svc, x_test, y_test, display_labels=class_names)
                        disp.plot()
                        plt.show()
                        st.pyplot()
                if save:
                    filename = "SVC"
                    pickle.dump(svc,open(filename,"wb"))
                    st.write("model saved")
                    st.balloons()
            if model == "RandomForest":
                x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
                raf = RandomForestClassifier(n_estimators=100)
                raf.fit(x_train, y_train)

                st.header("RandomForest hyperParams")
                tcl1,tcl2,tcl3=st.columns(3)
                
                tcl1.header("n_estimators")
                ta=tcl2.number_input("choose min value ",min_value=10, max_value=100, value=10)
                tmaxa=tcl3.number_input("choose max  value ",min_value=10, max_value=100, value=20)
                trl1,trcl2,trcl3=st.columns(3)
                # trl1.header("max_depth")
                # trsa= trcl2.number_input("choose mini value ")
                # tresa= trcl3.number_input("choose maxi value ")
                trls1,trcls2,trcls3=st.columns(3)
                trls1.header("min_split")
                tra= trcls2.number_input("choose min value split ")
                trea= trcls3.number_input("choose max value split ")
                tr1,tr2,tr3=st.columns(3)
                tr1.header("ccp alpha")
                trfa= tr2.number_input("choose min value ccp ")
                trf2a= tr3.number_input("choose max value ccp")
                tri1,tri2=st.columns(2)
                tri1.header("max features")
                trfia= tri2.multiselect("Choose one ",["auto", "sqrt", "log2","None"] )
                v1,v2 = st.columns(2)

                valider = v1.checkbox("Valider")
                
                save = v2.button("Save model")
                if valider:
                    params = {"n_estimators":np.arange(int(ta),int(tmaxa)),"min_samples_leaf":np.arange(tra,trea,0.1),"ccp_alpha":np.arange(trfa,trf2a,0.1),"max_features":trfia}
                    grid = RandomizedSearchCV(raf,params)
                    grid.fit(x_train,y_train)
                    grid.best_params_
                    raf = grid.best_estimator_
                    result = st.selectbox("choose",["Metrics","CURVE"])
                    if result=="Metrics":
                        st.text('Model Report:\n ' + classification_report(y_test, raf.predict(x_test)))
                        report()

                    if result=="CURVE": 
                        
                        y_test = y_test.to_numpy()
                        st.subheader("Confusion Matrix") 
                        fig, ax = plt.subplots()
                        cm=confusion_matrix(y_test,raf.predict(x_test),labels=raf.classes_)
                        st.write(cm)
                        ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=raf.classes_)
                        st.pyplot(fig)
                        st.subheader("TREEPLOT")
                        apprent = raf.fit(x_train,y_train)
                        fig, ax = plt.subplots()
                        tree.plot_tree(apprent, filled=True)
                        st.pyplot(fig)
                if save:
                    filename = "raf"
                    pickle.dump(raf,open(filename,"wb"))
                    st.balloons()
                    st.write("model saved")

            if model == "KNN":
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
                v1,v2 = st.columns(2)

                valider = v1.checkbox("Valider")
                
                save = v2.button("Save model")
                if valider:
                    params = {"n_neighbors":np.arange(int(k),int(kmax),1),"weights":l,"algorithm":ac}
                    grid = RandomizedSearchCV(mknn,params)
                    grid.fit(x_train,y_train)
                    grid.best_params_
                    mknn = grid.best_estimator_
                    result = st.selectbox("choose",["Metrics","CURVE"])
                    if result=="Metrics":
                        st.text('Model Report:\n ' + classification_report(y_test, mknn.predict(x_test)))
                        report()
                    if result=="CURVE": 
                        
                        y_test = y_test.to_numpy()
                        st.subheader("Confusion Matrix") 
                        predictions = mknn.predict(x_test)
                        cm = confusion_matrix(y_test, predictions, labels=mknn.classes_)
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=mknn.classes_)
                        #plot_confusion_matrix(svc, x_test, y_test, display_labels=class_names)
                        disp.plot()
                        plt.show()
                        st.pyplot()
                    if save:
                        filename = "mknn"
                        pickle.dump(mknn,open(filename,"wb"))
                        st.balloons()
            if model == "Tree":
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
                v1,v2 = st.columns(2)

                valider = v1.checkbox("Valider")
                
                save = v2.button("Save model")
                if valider:
                    params = {"min_impurity_decrease":np.arange(t,tmax,0.1),"min_samples_split":np.arange(tr,tre,0.1),"min_samples_leaf":np.arange(trs,tres,0.1),"ccp_alpha":np.arange(trf,trf2,0.1),"max_features":trfi}
                    grid = RandomizedSearchCV(mtree,params)
                    grid.fit(x_train,y_train)
                    grid.best_params_
                    mtree = grid.best_estimator_
                    result = st.selectbox("choose",["Metrics","CURVE"])
                    if result=="Metrics":
                        st.text('Model Report:\n ' + classification_report(y_test, mtree.predict(x_test)))
                        report()

                    if result=="CURVE": 
                        
                        y_test = y_test.to_numpy()
                        st.subheader("Confusion Matrix") 
                        predictions = mtree.predict(x_test)
                        cm = confusion_matrix(y_test, predictions, labels=mtree.classes_)
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=mtree.classes_)
                        #plot_confusion_matrix(svc, x_test, y_test, display_labels=class_names)
                        disp.plot()
                        plt.show()
                        st.pyplot()
                    if save:
                        filename = "mtree"
                        pickle.dump(mtree,open(filename,"wb"))
                        st.balloons()
            if model == "RandomForest":
                x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
                raf = RandomForestClassifier(n_estimators=100)
                raf.fit(x_train, y_train)

                st.header("RandomForest hyperParams")
                tcl1,tcl2,tcl3=st.columns(3)
                tcl1.header("n_estimators")
                ta=tcl2.number_input("choose min value ",min_value=10, max_value=100, value=10)
                tmaxa=tcl3.number_input("choose max  value ",min_value=10, max_value=1000, value=20)
                trls1,trcls2,trcls3=st.columns(3)
                trls1.header("min_split")
                tra= trcls2.number_input("choose min value split ")
                trea= trcls3.number_input("choose max value split ")
                tr1,tr2,tr3=st.columns(3)
                tr1.header("ccp alpha")
                trfa= tr2.number_input("choose min value ccp ")
                trf2a= tr3.number_input("choose max value ccp")
                tri1,tri2=st.columns(2)
                tri1.header("max features")
                trfia= tri2.multiselect("Choose one ",["auto", "sqrt", "log2","None"] )
                v1,v2 = st.columns(2)

                valider = v1.checkbox("Valider")
                
                save = v2.button("Save model")
                if valider:
                    params = {"n_estimators":np.arange(int(ta),int(tmaxa)),"min_samples_leaf":np.arange(tra,trea,0.1),"ccp_alpha":np.arange(trfa,trf2a,0.1),"max_features":trfia}
                    grid = RandomizedSearchCV(raf,params)
                    grid.fit(x_train,y_train)
                    grid.best_params_
                    raf = grid.best_estimator_
                    result = st.selectbox("choose",["Metrics","CURVE"])
                    if result=="Metrics":
                        st.text('Model Report:\n ' + classification_report(y_test, raf.predict(x_test)))
                        report()
                    if result=="CURVE": 
                        
                        y_test = y_test.to_numpy()
                        st.subheader("Confusion Matrix")  
                        predictions = raf.predict(x_test)
                        cm = confusion_matrix(y_test, predictions, labels=raf.classes_)
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=raf.classes_)
                    #plot_confusion_matrix(svc, x_test, y_test, display_labels=class_names)
                        disp.plot()
                        plt.show()
                        st.pyplot()
                if save:
                    filename = "raf"
                    pickle.dump(raf,open(filename,"wb"))
                    st.balloons()
        

            if model == "Perceptron":
                x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
                percep = Perceptron()
                percep.fit(x_train, y_train)
                percep.score(x_test,y_test)
                st.header("Perceptron Params")
                kcl1,kcl2,kcl3=st.columns(3)
            
                kcl1.header("n_iter_no_change")
                k=kcl2.number_input("choose min n_iter")
                kmax=kcl3.number_input("choose max n_iter")
                acl1,acl2=st.columns(2)
                v1,v2 = st.columns(2)

                valider = v1.checkbox("Valider")
                
                save = v2.button("Save model")
                if valider:
                    params = {"n_iter_no_change":np.arange(k,kmax,1)}
                    grid = RandomizedSearchCV(percep,params)
                    grid.fit(x_train,y_train)
                    grid.best_params_
                    percep = grid.best_estimator_
                    result = st.selectbox("choose",["Metrics","CURVE"])
                    if result=="Metrics":
                        st.text('Model Report:\n ' + classification_report(y_test, percep.predict(x_test)))
                    if result=="CURVE": 
                    
                        y_test = y_test.to_numpy()
                        st.subheader("Confusion Matrix")  
                        predictions = percep.predict(x_test)
                        cm = confusion_matrix(y_test, predictions, labels=percep.classes_)
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=percep.classes_)
                    #plot_confusion_matrix(svc, x_test, y_test, display_labels=class_names)
                        disp.plot()
                        plt.show()
                        st.pyplot()
                    if save:
                        filename = "percep"
                        pickle.dump(percep,open(filename,"wb"))
                        st.balloons()   


            if model == "MLPClassifier":
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
                 v1,v2 = st.columns(2)

                 valider = v1.checkbox("Valider")
                 save = v2.button("Save model")
                 if valider:

                     params = {'max_iter': [mlll ], 'alpha': np.arange(ml, mll), 'hidden_layer_sizes':np.arange(mlp22, mlp23), 'random_state':[0,1,2,3,4,5,6,7,8,9]}
                     grid = RandomizedSearchCV(mlp, params)
                     grid.fit(x_train,y_train)
                     grid.best_params_
                     mlp = grid.best_estimator_
                     #rcol1 = st.columns(1)
                     #rcol1.header("Score")
                     #st.write(mlp.score(x_test,y_test))
                     result = st.selectbox("choose",["Metrics","CURVE"])
                     if result=="Metrics":
                         st.text('Model Report:\n ' + classification_report(y_test, mlp.predict(x_test)))
                     if result=="CURVE": 
                        
                         y_test = y_test.to_numpy()
                         st.subheader("Confusion Matrix")  
                         predictions = mlp.predict(x_test)
                         cm = confusion_matrix(y_test, predictions, labels=mlp.classes_)
                         disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=mlp.classes_)
                        #plot_confusion_matrix(svc, x_test, y_test, display_labels=class_names)
                         disp.plot()
                         plt.show()
                         st.pyplot()
                     if save:
                         filename = "mlp"
                         pickle.dump(mlp,open(filename,"wb")) 
            if model =="RidgeClassifier":
         
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
                 v1,v2 = st.columns(2)

                 valider = v1.checkbox("Valider")
                 save = v2.button("Save model")
                 if valider:

                     params = {'max_iter': [ridg], 'alpha': np.arange(r1, r2), 'random_state':[0,1,2,3,4,5,6,7,8,9]}
                     grid = RandomizedSearchCV(ridge, params)
                     grid.fit(x_train,y_train)
                     grid.best_params_
                     ridge = grid.best_estimator_
                     st.write(ridge.score(x_test,y_test))
                     result = st.selectbox("choose",["Metrics","CURVE"])
                     if result=="Metrics":
                         st.text('Model Report:\n ' + classification_report(y_test, ridge.predict(x_test)))
                     if result=="CURVE": 
                        
                         y_test = y_test.to_numpy()
                         st.subheader("Confusion Matrix")  
                         predictions = ridge.predict(x_test)
                         cm = confusion_matrix(y_test, predictions, labels=ridge.classes_)
                         disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=ridge.classes_)
                        #plot_confusion_matrix(svc, x_test, y_test, display_labels=class_names)
                         disp.plot()
                         plt.show()
                         st.pyplot()
                     if save:
                         filename = "ridge"
                         pickle.dump(ridge,open(filename,"wb"))
                         st.balloons()   
            if model =="AdaBoostClassifier ":
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
                 v1,v2 = st.columns(2)
                 valider = v1.checkbox("Valider")
                 save = v2.button("Save model")
                 if valider:

                     params = {'algorithm': [adb], 'Numbers estimators': [addd], 'learning_rate': np.arange(ad, a), 'random_state':[0,1,2,3,4,5,6,7,8,9]}
                     grid = RandomizedSearchCV(ada, params)
                     grid.fit(x_train,y_train)
                     grid.best_params_
                     ada= grid.best_estimator_
                     st.write(ada.score(x_test,y_test))
                     result = st.selectbox("choose",["Metrics","CURVE"])
                     if result=="Metrics":
                         st.text('Model Report:\n ' + classification_report(y_test,ada.predict(x_test)))
                     if result=="CURVE": 
                        
                         y_test = y_test.to_numpy()
                         st.subheader("Confusion Matrix")  
                         predictions =ada.predict(x_test)
                         cm = confusion_matrix(y_test, predictions, labels=ada.classes_)
                         disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=ada.classes_)
                        #plot_confusion_matrix(svc, x_test, y_test, display_labels=class_names)
                         disp.plot()
                         plt.show()
                         st.pyplot()
                     if save:
                         filename = "adaboost"
                         pickle.dump(ada,open(filename,"wb"))
                         st.write("model saved") 
            if model =="QuadraticDiscriminantAnalysis":
         
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
                 v1,v2 = st.columns(2)
                 valider = v1.checkbox("Valider")
                 save = v2.button("Save model")
                 if valider:

                     params = {'store_covariance': [qu], 'tol': np.arange(ql, qll), 'reg_param': np.arange(qlp22, qlp23)}
                     grid = RandomizedSearchCV(qua, params)
                     grid.fit(x_train,y_train)
                     grid.best_params_
                     qua= grid.best_estimator_
                     st.write(qua.score(x_test,y_test))
                     result = st.selectbox("choose",["Metrics","CURVE"])
                     if result=="Metrics":
                         st.text('Model Report:\n ' + classification_report(y_test,qua.predict(x_test)))
                     if result=="CURVE": 
                        
                         y_test = y_test.to_numpy()
                         st.subheader("Confusion Matrix")  
                         predictions =qua.predict(x_test)
                         cm = confusion_matrix(y_test, predictions, labels=qua.classes_)
                         disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=qua.classes_)
                        #plot_confusion_matrix(svc, x_test, y_test, display_labels=class_names)
                         disp.plot()
                         plt.show()
                         st.pyplot()
                     if save:
                         filename = "QuadraticDiscriminantAnalysis"
                         pickle.dump(qua,open(filename,"wb"))
                         st.write("model saved") 
            if model =="GaussianMixture":
            
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
                 g11,g33,g44=st.columns(3)
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
                 v1,v2 = st.columns(2)
                 valider = v1.checkbox("Valider")
                 save = v2.button("Save model")
                 if valider:

                     params = {'n_components': np.arange(ga, g), 'cv': np.arange(gaus, gaus23),'covariance_type': [g22],'max_iter': [glll], 'tol': np.arange(g3, g4l),'init_params': [gu], 'reg_covar': np.arange(glp22, glp23)}
                     grid = RandomizedSearchCV(gau, params)
                     grid.fit(x_train,y_train)
                     grid.best_params_
                     gau= grid.best_estimator_
                     st.write(gau.score(x_test,y_test))
                     result = st.selectbox("choose",["Metrics","CURVE"])
                     if result=="Metrics":
                         st.text('Model Report:\n ' + classification_report(y_test,gau.predict(x_test)))
                     if result=="CURVE": 
                        
                         y_test = y_test.to_numpy()
                         st.subheader("Confusion Matrix")  
                         predictions =gau.predict(x_test)
                         cm = confusion_matrix(y_test, predictions, labels=gau.classes_)
                         disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=gau.classes_)
                        #plot_confusion_matrix(svc, x_test, y_test, display_labels=class_names)
                         disp.plot()
                         plt.show()
                         st.pyplot()
                     if save:
                         filename = "GaussianMixture"
                         pickle.dump(gau,open(filename,"wb"))
                         st.write("model saved") 
            if model =="OneVsOneClassifier":
                 x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
                 svc = svm.SVC(C= 0.1, gamma=1, kernel='linear')
                 one= OneVsOneClassifier(svc)
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
                 v1,v2 = st.columns(2)
                 valider = v1.checkbox("Valider")
                 save = v2.button("Save model")
                 if valider:

                     params = {'estimator': [oee], 'n_jobs': np.arange(one1, one2)}
                     grid = RandomizedSearchCV(one, params)
                     grid.fit(x_train,y_train)
                     grid.best_params_
                     one = grid.best_estimator_
                     st.write(one.score(x_test,y_test))
                     result = st.selectbox("choose",["Metrics","CURVE"])
                     if result=="Metrics":
                         st.text('Model Report:\n ' + classification_report(y_test,qua.predict(x_test)))
                     if result=="CURVE": 
                        
                         y_test = y_test.to_numpy()
                         st.subheader("Confusion Matrix")  
                         predictions =one.predict(x_test)
                         cm = confusion_matrix(y_test, predictions, labels=one.classes_)
                         disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=one.classes_)
                        #plot_confusion_matrix(svc, x_test, y_test, display_labels=class_names)
                         disp.plot()
                         plt.show()
                         st.pyplot()
                     if save:
                         filename = "one"
                         pickle.dump(one,open(filename,"wb"))
                         st.write("model saved")
            if model =="LabelPropagation ":
         
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
                 v1,v2 = st.columns(2)
                 valider = v1.checkbox("Valider")
                 save = v2.button("Save model")
                 if valider:

                     params = {"n_neighbors":np.arange(llp22,llp23,1),"gamma":np.arange(ll,lll,0.1),"kernel":[LL],"max_iter":[llll]}
                     grid = RandomizedSearchCV(llp,params)
                     grid.fit(x_train,y_train)
                     grid.best_params_
                     llp = grid.best_estimator_
                     st.write(llp.score(x_test,y_test))
                     result = st.selectbox("choose",["Metrics","CURVE"])
                     if result=="Metrics":
                         st.text('Model Report:\n ' + classification_report(y_test,llp.predict(x_test)))
                     if result=="CURVE": 
                        
                         y_test = y_test.to_numpy()
                         st.subheader("Confusion Matrix")  
                         predictions =llp.predict(x_test)
                         cm = confusion_matrix(y_test, predictions, labels=llp.classes_)
                         disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=llp.classes_)
                        #plot_confusion_matrix(svc, x_test, y_test, display_labels=class_names)
                         disp.plot()
                         plt.show()
                         st.pyplot()
                 if save:
                         filename = "llp"
                         pickle.dump(llp,open(filename,"wb"))
                         st.ballons()
            if model =="BayesianGaussianMixture ":
                 x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
                 gmm =BayesianGaussianMixture()
                 gmm.fit(x_train, y_train)
                 gmm.score(x_test,y_test)
                 st.header("BayesianGaussianMixture Params")
                 gml1,gml2,gml3=st.columns(3)
                
                 gml1.header("n_components")
                 gml=gml2.number_input("choose minimum")
                 gmll=gml3.number_input("choose maximum")
                 g111,g222=st.columns(2)
                 g111.header("covariance_type",)
                 gm= g222.multiselect("Choose type(s)",[ "spherical", "tied", "diag", "full"])
                 gmlp1,gmlp2,gmlp3=st.columns(3)
                
                 gmlp1.header("reg_covar")
                 gmlp22=gmlp2.number_input("Choose min n",min_value=1, max_value=10, value=1)
                 gmlp23=gmlp3.number_input("Choose max n",min_value=1, max_value=50000, value=1)
                 v1,v2 = st.columns(2)
                 valider = v1.checkbox("Valider")
                 save = v2.button("Save model")
                 if valider:

                     params = {'n_components': np.arange(gml, gmll),'covariance_type': [gm],'reg_covar': np.arange(gmlp22, gmlp23)}
                     grid =RandomizedSearchCV(gmm, params)
                     grid.fit(x_train,y_train)
                     grid.best_params_
                     gmm= grid.best_estimator_
                     st.write(gmm.score(x_test,y_test))
                     result = st.selectbox("choose",["Metrics","CURVE"])
                     if result=="Metrics":
                         st.text('Model Report:\n ' + classification_report(y_test,gmm.predict(x_test)))
                     if result=="CURVE": 
                        
                         y_test = y_test.to_numpy()
                         st.subheader("Confusion Matrix")  
                         predictions =gmm.predict(x_test)
                         cm = confusion_matrix(y_test, predictions, labels=gmm.classes_)
                         disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=gmm.classes_)
                        #plot_confusion_matrix(svc, x_test, y_test, display_labels=class_names)
                         disp.plot()
                         plt.show()
                         st.pyplot()
                     if save:
                         filename = "BayesianGaussianMixture"
                         pickle.dump(gmm,open(filename,"wb"))
                         st.write("model saved") 
            if model == "Voting":
                svc = pickle.load(open("svc","rb"))
                mtree = pickle.load(open("mtree","rb"))
                mknn = pickle.load(open("mknn","rb"))


                model_final = VotingClassifier([('SVC',svc),('KNN',mknn),('TREE',mtree)])
                for model in (svc,mknn,mtree,model_final):
                    model.fit(x_train,y_train)
                    st.write(model.__class__.__name__ , model.score(x_test,y_test))
                

                if st.button("save"):
                    filename = "model_final"
                    pickle.dump(model_final,open(filename,"wb"))
                    st.balloons()
            
    if option=="Predict":
        modelpredict = st.selectbox("Model of prediction",["SVC","RandomForest","Tree","KNN","Voting","Bagging","Perceptron","MLPClassifier","RidgeClassifier"])
        if modelpredict=="SVC":
            svc = pickle.load(open("svc","rb"))
            predict_input(svc)

        if modelpredict=="KNN":
            mknn = pickle.load(open("mknn","rb"))
            predict_input(mknn)
        if modelpredict=="RandomForest":
            raf = pickle.load(open("raf","rb"))
            predict_input(raf)
        
        if modelpredict=="Tree":
            mtree = pickle.load(open("mtree","rb"))
            predict_input(mtree)
        if modelpredict=="Bagging":
            svc = pickle.load(open("svc","rb"))
            predict_input(svc)

        if modelpredict=="Perceptron":
            percep = pickle.load(open("percep","rb"))
            predict_input(percep)
        if modelpredict=="MLPClassifier":
            mlp = pickle.load(open("mlp","rb"))
            predict_input(mlp)
        if modelpredict=="RidgeClassifier":
            ridge = pickle.load(open("ridge","rb"))
            predict_input(ridge)

        if modelpredict=="BayesianGaussianMixture":
            gmm = pickle.load(open("gmm","rb"))
            predict_input(gmm)
        if modelpredict=="LabelPropagation":
            llp = pickle.load(open("llp","rb"))
            predict_input(llp)
        if modelpredict=="QuadraticDiscriminantAnalysis":
            qua = pickle.load(open("qua","rb"))
            predict_input(qua)

        if modelpredict=="OneVsOneClassifier":
            one = pickle.load(open("one","rb"))
            predict_input(one)
        if modelpredict=="GaussianMixture":
            gau = pickle.load(open("gau","rb"))
            predict_input(gau)
        if modelpredict=="AdaBoostClassifier":
            ada = pickle.load(open("ada","rb"))
            predict_input(ada)
        
        if modelpredict=="Voting":
            model_final = pickle.load(open("model_final","rb"))
            predict_input(model_final)


    if option=="Compare models":
        svc = pickle.load(open("svc","rb"))
        mtree = pickle.load(open("mtree","rb"))
        mknn = pickle.load(open("mknn","rb"))
        from sklearn.preprocessing import label_binarize
        y_test_binarized=label_binarize(y_test,classes=np.unique(y_test))
        # roc curve for classes
        fpr = {}
        tpr = {}
        thresh ={}
        roc_auc = dict()
        n_class = y.shape[0]
        for i in range(n_class-3):    
            fpr[i], tpr[i], thresh[i] = roc_curve(y_test_binarized[:,i], mtree.predict_proba(x_test)[:,i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            # plotting    
            plt.plot(fpr[i], tpr[i], linestyle='--', 
                    label='%s vs Rest (AUC=%0.2f)'%(y[i],roc_auc[i]))

        plt.plot([0,1],[0,1],'b--')
        plt.xlim([0,1])
        plt.ylim([0,1.05])
        plt.title('Multiclass ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')
        plt.legend(loc='lower right')
        # fpr, tpr, thresholds = roc_curve(y_test, mtree.predict_proba(x_test)[:,1])
        # sns.set(style='white',font_scale=1.2)
        # fig = plt.figure(figsize=[10,7])
        # plt.plot(fpr, tpr,lw=2,label='RandomForest (AUC={:.3f})'.format(roc_auc_score(y_test, mtree.predict_proba(x_test)[:,1])))
        # plt.plot([0,1],[0,1],c='violet',ls='--')
        # plt.xlim([-0.05,1.05])
        # plt.ylim([-0.05,1.05])
        # plt.legend(loc="lower right",fontsize=15)

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Receiver Operating Characteristic (ROC) Curve',weight='bold',fontsize=15);
sns.despine()
plt.savefig('ROCplot.png',transparent=True, bbox_inches='tight')

if choose == "Home":
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
                        width=400,
                        key=None,
                        )  
        st.title("BIENVENUE SUR NOTRE APPLICATION WEB DE PREDICTION ")

        loti("C:/Users/DIASSANA/Desktop/ProjetMetier/Projet Metier final/ProjetMetier final/artificial-intelligence.json")
        st.write("Veuillez parcourir les autres options de l'application")
####### html/css config ########
#st.set_page_config(layout="wide", page_title="AI students project", menu_items={
    #'About': "AI without code - réalisé par FATOUMATA ET HAROUNA"
#})

st.markdown("""
<style>
.first_titre {
    font-size:75px !important;
    font-weight: bold;
    box-sizing: border-box;
    text-align: center;
    width: 100%;
}
.intro{
    text-align: justify;
    font-size:20px !important;
}
.grand_titre {
    font-size:30px !important;
    font-weight: bold;
    text-align: center;
    text-decoration: underline;
    text-decoration-color: #4976E4;
    text-decoration-thickness: 5px;
}
.section{
    font-size:20px !important;
    font-weight: bold;
    text-align: center;
    text-decoration: underline;
    text-decoration-color: #111111;
    text-decoration-thickness: 3px;
}
.petite_section{
    font-size:16px !important;
    font-weight: bold;
}
.nom_colonne_page3{
    font-size:17px !important;
    text-decoration: underline;
    text-decoration-color: #000;
    text-decoration-thickness: 1px;
}
</style>
""", unsafe_allow_html=True)
