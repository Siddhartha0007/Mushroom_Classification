# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 09:38:56 2022

@author: Siddhartha-PC
"""
# =============================================================================
    
###  Import Libreries  
import streamlit as st
from streamlit_option_menu import option_menu
st.set_option('deprecation.showPyplotGlobalUse', False)
import contractions 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image 
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
from collections import  Counter
import inflect
from bs4 import BeautifulSoup
import streamlit.components.v1 as components
import os
#for model-building
from sklearn.model_selection import train_test_split
import string
from tqdm import tqdm
#for model accuracy
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import absolute
from numpy import sqrt
#for visualization
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import plotly.tools as tls
init_notebook_mode(connected=True)
import plotly.express as px
import plotly.graph_objects as go
import pickle
from joblib import dump, load
import joblib
# Utils
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
import sys
from sklearn.metrics import r2_score,accuracy_score,mean_absolute_error,mean_squared_error,mean_absolute_percentage_error 
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from plotly import tools
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,KFold
from sklearn.preprocessing import StandardScaler,LabelEncoder
import plotly.figure_factory as ff
import cufflinks as cf
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.metrics import roc_auc_score
import matplotlib.gridspec as gridspec
import scipy
import re
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
sns.set_palette("hls")
plt.style.use('fivethirtyeight')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats.mstats import winsorize
from scipy.stats import boxcox, probplot, norm
from scipy.special import inv_boxcox
import random
import datetime
import math
sns.set_palette("hls")
plt.style.use('fivethirtyeight')
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import silhouette_score
## Hyperopt modules
#from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
#from functools import partial
from sklearn.metrics import roc_auc_score
import matplotlib.gridspec as gridspec
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
sns.set_palette("hls")
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import scipy
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
#preprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#ML libraries
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
from yellowbrick.classifier import PrecisionRecallCurve
#preprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#ML libraries
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import plot_importance, to_graphviz
from xgboost import plot_tree
#For Model Acuracy
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from xgboost import plot_importance, to_graphviz #
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,roc_auc_score,accuracy_score,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
import plotly.graph_objs as go
import plotly.tools as tls
import cufflinks as cf
import plotly.figure_factory as ff
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=True)
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import silhouette_score
from sklearn.metrics import roc_auc_score
import matplotlib.gridspec as gridspec
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
#preprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#ML libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
#preprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#ML libraries
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
import re
import sys
#preprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#ML libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import plot_importance, to_graphviz
from xgboost import plot_tree
#For Model Acuracy
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from xgboost import plot_importance, to_graphviz #
from sklearn.metrics import classification_report
from yellowbrick.classifier import PrecisionRecallCurve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,roc_auc_score,accuracy_score,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import log_loss
# EDA Pkgs
import pandas as pd 
import codecs
from pandas_profiling import ProfileReport 
# Components Pkgs
import streamlit.components.v1 as components
from streamlit_pandas_profiling import st_profile_report

# Custome Component Fxn
import sweetviz as sv 
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

#lottie animations
import time
import requests
import json
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner

le_encoder=LabelEncoder()
###############################################Data Processing###########################
# Importing Data and Pickle file
mushroom_data=pd.read_csv("mushroom_data.csv")
loaded_model=pickle.load(open("Random_Forest_model_intelligence.pkl","rb"))

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
lottie_url_hello = "https://assets5.lottiefiles.com/packages/lf20_ev1cfn9h.json"
lottie_url_download = "https://assets4.lottiefiles.com/private_files/lf30_t26law.json"
lottie_hello = load_lottieurl(lottie_url_hello)
project_url_1="https://assets9.lottiefiles.com/packages/lf20_bzgbs6lx.json"
project_url_2="https://assets6.lottiefiles.com/packages/lf20_eeuhulsy.json"
report_url="https://assets9.lottiefiles.com/packages/lf20_zrqthn6o.json"
about_url="https://assets2.lottiefiles.com/packages/lf20_k86wxpgr.json"

about_1=load_lottieurl(about_url)
report_1=load_lottieurl(report_url)
project_1=load_lottieurl(project_url_1)
project_2=load_lottieurl(project_url_2)

lottie_download = load_lottieurl(lottie_url_download)

#st_lottie(lottie_hello, key="hello")


def st_display_sweetviz(report_html,width=1000,height=500):
	report_file = codecs.open(report_html,'r')
	page = report_file.read()
	components.html(page,width=width,height=height,scrolling=True)


def user_input_features():
    
    cap_shape = st.sidebar.selectbox("cap-shape",mushroom_data["cap-shape"].unique())
    cap_surface = st.sidebar.selectbox("cap-surface",mushroom_data["cap-surface"].unique())
    cap_color = st.sidebar.selectbox("cap-color",mushroom_data["cap-color"].unique())
    bruises = st.sidebar.selectbox("bruises",mushroom_data["bruises"].unique())
    odor = st.sidebar.selectbox("odor",mushroom_data["odor"].unique())
    gill_attachment = st.sidebar.selectbox("gill-attachment",mushroom_data["gill-attachment"].unique())
    gill_spacing = st.sidebar.selectbox("gill-spacing",mushroom_data["gill-spacing"].unique())
    gill_size = st.sidebar.selectbox("gill-size",mushroom_data["gill-size"].unique())
    gill_color = st.sidebar.selectbox("gill-color",mushroom_data["gill-color"].unique())
    stalk_shape = st.sidebar.selectbox("stalk-shape",mushroom_data["stalk-shape"].unique())
    stalk_root = st.sidebar.selectbox("stalk-root",mushroom_data["stalk-root"].unique())
    stalk_surface_above_ring = st.sidebar.selectbox("stalk-surface-above-ring",mushroom_data["stalk-surface-above-ring"].unique())
    stalk_surface_below_ring = st.sidebar.selectbox("stalk-surface-below-ring",mushroom_data["stalk-surface-below-ring"].unique())
    stalk_color_above_ring = st.sidebar.selectbox("stalk-color-above-ring",mushroom_data["stalk-color-above-ring"].unique())
    stalk_color_below_ring = st.sidebar.selectbox("stalk-color-below-ring",mushroom_data["stalk-color-below-ring"].unique())
    veil_type = st.sidebar.selectbox("veil-type",mushroom_data["veil-type"].unique())
    veil_color = st.sidebar.selectbox("veil-color",mushroom_data["veil-color"].unique())
    ring_number = st.sidebar.selectbox("ring-number",mushroom_data["ring-number"].unique())
    ring_type = st.sidebar.selectbox("ring-type",mushroom_data["ring-type"].unique())
    spore_print_color = st.sidebar.selectbox("spore-print-color",mushroom_data["spore-print-color"].unique())
    population = st.sidebar.selectbox("population",mushroom_data["population"].unique())
    habitat = st.sidebar.selectbox("habitat",mushroom_data["habitat"].unique())
    
   
    
    
    
    data = {'cap-shape':cap_shape,
            'cap-surface':cap_surface,
            'cap-color':cap_color,
            'bruises':bruises,
            'odor':odor,
            'gill-attachment':gill_attachment,
            'gill-spacing':gill_spacing,
            'gill-size':gill_size,
            'gill-color':gill_color,
            'stalk-shape':stalk_shape,
            'stalk-root':stalk_root,
            'stalk-surface-above-ring':stalk_surface_above_ring,
            'stalk-surface-below-ring':stalk_surface_below_ring,
            'stalk-color-above-ring':stalk_color_above_ring,
            'stalk-color-below-ring':stalk_color_below_ring,
            'veil-type':veil_type,
            'veil-color':veil_color,
            'ring-number':ring_number,
            'ring-type':ring_type,
            'spore-print-color':spore_print_color,
            'population':population,
            'habitat':habitat,
            
            }

    features = pd.DataFrame(data,index = [0])
    
    return features
        


###############################################Exploratory Data Analysis###############################################

#For Label Analysis
def label_analysis():
    st.title("Basic Plots of the Features")
    
    image1= Image.open("basic.png")
    st.image(image1,use_column_width=True)
    image1= Image.open("basic1.png")
    st.image(image1,use_column_width=True)
    image1= Image.open("basic2.png")
    st.image(image1,use_column_width=True)
    image1= Image.open("basic3.png")
    st.image(image1,use_column_width=True)
    image1= Image.open("basic4.png")
    st.image(image1,use_column_width=True)
    image1= Image.open("basic5.png")
    st.image(image1,use_column_width=True)
    image1= Image.open("basic6.png")
    st.image(image1,use_column_width=True)
    image1= Image.open("basic7.png")
    st.image(image1,use_column_width=True)
    image1= Image.open("basic8.png")
    st.image(image1,use_column_width=True)
    image1= Image.open("basic9.png")
    st.image(image1,use_column_width=True)
    image1= Image.open("basic10.png")
    st.image(image1,use_column_width=True)
    image1= Image.open("basic11.png")
    st.image(image1,use_column_width=True)
    
    def pie_plot_func(features):
        mushrooms= mushroom_data.copy()
        #Get the population types and its values for Single Pie chart
        populations = mushrooms[features].value_counts()
        pop_labels=  mushrooms[features].unique()
        pop_size = populations.values.tolist() #Provides numerical values
        pop_types = populations.axes[0].tolist() #Converts index labels object to list
        fig = plt.figure(figsize=(13,5))
        # Plot
        plt.title('Mushroom'+' '+str(features)+' '+ 'Percentange Distributions', fontsize=22)
        patches, texts, autotexts = plt.pie(pop_size, labels=pop_labels,
                        autopct='%1.1f%%', shadow=True, startangle=150)
        for text,autotext in zip(texts,autotexts):
            text.set_fontsize(14)
            autotext.set_fontsize(14)

        plt.axis('equal')
        plt.legend()
        plt.show()
    st.write("Pie Plots Of Various Features: ")
    for i in mushroom_data.columns:
        st.pyplot(pie_plot_func(i))
   
def label_analysis1():
    st.title("Multi level Pie & SunBurst Plots")
    
    image1= Image.open("mul.png")
    st.image(image1,use_column_width=True)
    image1= Image.open("mul1.png")
    st.image(image1,use_column_width=True)
    image1= Image.open("mul2.png")
    st.image(image1,use_column_width=True)
    


def label_analysis2():
    st.title("Various Interactive Plots")
    
    image1= Image.open("inter1.png")
    st.image(image1,use_column_width=True)
    image1= Image.open("inter2.png")
    st.image(image1,use_column_width=True)
    image1= Image.open("inter3.png")
    st.image(image1,use_column_width=True)
    image1= Image.open("inter4.png")
    st.image(image1,use_column_width=True)
    image1= Image.open("inter5.png")
    st.image(image1,use_column_width=True)
    image1= Image.open("inter6.png")
    st.image(image1,use_column_width=True)
    image1= Image.open("inter7.png")
    st.image(image1,use_column_width=True)
    image1= Image.open("inter8.png")
    st.image(image1,use_column_width=True)
    image1= Image.open("inter9.png")
    st.image(image1,use_column_width=True)
    image1= Image.open("inter10.png")
    st.image(image1,use_column_width=True)
    

def label_analysis3():
    st.write("Feature importance")
    
    image1= Image.open("feature_imp.png")
    st.image(image1,use_column_width=True)
    
    
    st.write("Feature Selection Using Mutual Information")
    
    image2= Image.open("mutual_feature_imp.png")
    st.image(image2,use_column_width=True)

def label_analysis4():
    st.write("Basic Statistics Of the Data")
    categorical_data=mushroom_data.select_dtypes(include=["object"]).columns.to_list()
    df_sample1 =mushroom_data.describe(include='all').round(2).T
    colorscale = [[0, '#4d004c'],[.5, '#f2e5ff'],[1, '#ffffff']]
    font_colors=[[0,'#ffffff'], [.5,'#000000'], [1,'#000000']]
    fig =  ff.create_table(df_sample1, colorscale=colorscale,index=True,font_colors=['#ffffff', '#000000','#000000'])  
 
    return st.plotly_chart(fig)
    st.write("Data Types are Categorical")
    st.write(categorical_data)
    
    




def label_analysis5():
    def plot13():
        corr_matrix = mushroom_data.corr()
        f,ax = plt.subplots(figsize=(14,6))
        sns.heatmap(corr_matrix,annot=True,linewidths=0.5,linecolor="red", fmt= '.2f',ax=ax,cmap='rainbow')
        plt.show()
         
    p13=plot13()
    st.write("Correlation Matrix")
    st.pyplot(p13)
    

    
    def plot14():
        corr = mushroom_data.corrwith(mushroom_data['class'],method='spearman').reset_index()
        corr.columns = ['Index','Correlations']
        corr = corr.set_index('Index')
        corr = corr.sort_values(by=['Correlations'], ascending = False).head(10)
        plt.figure(figsize=(10, 8))
        fig = sns.heatmap(corr, annot=True, fmt="g", cmap='coolwarm', linewidths=0.4, linecolor='red')
        plt.title("Correlation of Variables with Class", fontsize=20)
        plt.show()
         
    p14=plot14()
    st.write(" Correlation Matrix")
    st.pyplot(p14)
    
    def plot15():
        import seaborn as sns
        corr_matrix = mushroom_data.corr()
        sns.set(rc = {'figure.figsize':(12,7)}) # handle size of thr figure 
        #mask = np.zeros_like(corr_matrix)
        mask = np.triu(np.ones_like(corr_matrix, dtype = bool))
        #mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):
            sns.heatmap(corr_matrix, mask=mask ,annot=True, square=True,linewidths=0.5, 
                  fmt= '.2f',cmap='coolwarm');
            
          
         
    p15=plot15()
    st.write("Correlation Matrix")
    st.pyplot(p15)



def get_data_class():
    mushroom_data=pd.read_csv("mushroom_data.csv")
    from sklearn.preprocessing import LabelEncoder
    labelencoder=LabelEncoder()
    for column in mushroom_data.columns:
        mushroom_data[column] = labelencoder.fit_transform(mushroom_data[column])
    X = mushroom_data.drop(["class"],axis=1)
    y = mushroom_data[["class"]]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=42)
    
    return X_train,X_test,y_train,y_test  
    

############################################### Model Learning ###############################################
#For Precision Recall Curve
def PRCurve(model):
    X_train,X_test,y_train,y_test=get_data_class()
    prc = PrecisionRecallCurve(model)
    prc.fit(X_train, y_train)
    avg_prc = prc.score(X_test, y_test)
    plt.legend(labels = ['Precision Recall Curve',"AP=%.3f"%avg_prc], loc = 'lower right', prop={'size': 14})
    plt.xlabel(xlabel = 'Recall', size = 14)
    plt.ylabel(ylabel = 'Precision', size = 14)
    plt.title(label = 'Precision Recall Curve', size = 16)
    

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    
    """Plot the learning curve for the estinmator."""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(name='Training score - Standard Deviation',
                            x=train_sizes,
                            y=train_scores_mean+train_scores_std,
                            mode='lines',
                            showlegend=False,
                            marker=dict(color='green')))
    fig.add_trace(go.Scatter(name='Training score',
                            x=train_sizes,
                            y=train_scores_mean,
                            fill='tonexty',
                            mode='lines+markers',
                            marker=dict(color='green')))
    fig.add_trace(go.Scatter(name='Training score + Standard Deviation',
                            x=train_sizes,
                            y=train_scores_mean-train_scores_std,
                            mode='lines',
                            fill='tonexty',
                            showlegend=False,
                            marker=dict(color='green')))
    fig.add_trace(go.Scatter(x=train_sizes,
                            y=test_scores_mean+test_scores_std,
                            mode='lines',
                            showlegend=False,
                            marker=dict(color='red')))
    fig.add_trace(go.Scatter(name='Validation Score',
                            x=train_sizes,
                            y=test_scores_mean,
                            mode='lines+markers',
                            fill='tonexty',
                            marker=dict(color='red')))
    fig.add_trace(go.Scatter(x=train_sizes,
                            y=test_scores_mean-test_scores_std,
                            mode='lines',
                            fill='tonexty',
                            showlegend=False,
                            marker=dict(color='red')))

    fig.update_layout(width=700,height=400,template='seaborn',title=title,
                        margin=dict(l=60,r=0,b=0,t=40),legend=dict(orientation='h',x=0.5,y=1),
                        xaxis=dict(title='Training examples',mirror=True,linecolor='black',linewidth=2),
                        yaxis=dict(title='Scores',range=ylim if ylim is not None else None,
                        mirror=True,linecolor='black',linewidth=2))
    return fig    



# st.plotly_chart(p11)

   
#Model Random Forest Classifier
def randomforest_classifier():
    X_train,X_test,y_train,y_test=get_data_class()
    rf =RandomForestClassifier(n_estimators=100,criterion='entropy',max_depth=4,
                            max_features='auto',random_state=None,class_weight="balanced")
    rf.fit(X_train, y_train)
    rf_train_predict=rf.predict(X_train)
    rf_prediction = rf.predict(X_test)
    rf_score = rf.score(X_test, y_test)
    st.write("Random Forest Classification Train Accuracy: {}%".format(round(rf.score(X_train,y_train)*100,2)))
    st.write("Random Forest Classification Test Accuracy: {}%".format(round(rf.score(X_test,y_test)*100,2)))
    rf_cm = confusion_matrix(y_test, rf_prediction)
    st.write("Classification Report:Train data\n")
    plt.figure(figsize = [10,5])
    clf_report3 = classification_report(y_train,rf_train_predict,
                                   labels=None,
                                   target_names=None,
                                   output_dict=True)
    sns.heatmap(pd.DataFrame(clf_report3).T, annot=True,cmap="cividis",fmt='.2f',cbar=True)
    st.pyplot()
    
    #st.write(classification_report(y_test, rf_prediction))
    st.write("Logloss:\n",log_loss(y_test, rf.predict_proba(X_test)))
    st.write('Confusion Matrix \n', rf_cm)
    st.write("Classification Report:Test data\n")
    clf_report4 = classification_report(y_test,rf_prediction,
                                   labels=None,
                                   target_names=None,
                                   output_dict=True)
    sns.heatmap(pd.DataFrame(clf_report4).T, annot=True,cmap="viridis",fmt='.2f',cbar=True)
    st.pyplot()
    #Precision Recall Curve
    st.write("Precision Recall Curve")
    PRCurve(rf)
    st.pyplot()
    st.write("Plotting Confution Matrix")
    plot_confusion_matrix(rf, X_test,y_test)
    st.pyplot()
    st.write("AUC ROC Curve")
    plot_roc_curve(rf, X_test, y_test)
    st.pyplot()
    
    st.write("Feature Importance Learnt by the Model")
    fig = plt.figure(figsize = (14, 9))
    ax = fig.add_subplot(111)
    xgb_classifer= XGBClassifier(n_estimators=200,max_depth=6,booster="gbtree",learning_rate=0.005)
    xgb_classifer.fit(X_train,y_train)
    colours = plt.cm.viridis(np.linspace(0, 1, 22))
    ax = plot_importance(xgb_classifer, height = 1, grid = False, color = colours, \
                     show_values = False, importance_type = 'cover', ax = ax, max_num_features=22);
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
    ax.set_xlabel('importance score', size = 16);
    ax.set_ylabel('features', size = 16);
    ax.set_yticklabels(ax.get_yticklabels(), size = 12);
    ax.set_title('Ordering of features by importance to the model learnt', size = 20);
    st.pyplot()
    st.write("Model Learning Curve")
    p11=plot_learning_curve(rf, 'RandomForestClassifier',X_train, y_train)
    st.plotly_chart(p11)
    
#Model XGBoost  Regressor
def XGboost_classifier():
    X_train,X_test,y_train,y_test=get_data_class()
    xgb_classifer= XGBClassifier(n_estimators=200,max_depth=6,booster="gbtree",learning_rate=0.005)
    xgb_classifer.fit(X_train,y_train)
    xgb_train_predict=xgb_classifer.predict(X_train)
    xgb_prediction = xgb_classifer.predict(X_test)
    xgb_score = xgb_classifer.score(X_test, y_test)
    st.write("XGB Classification Train Accuracy: {}%".format(round(xgb_classifer.score(X_train,y_train)*100,2)))
    st.write("XGB Classification Test Accuracy: {}%".format(round(xgb_classifer.score(X_test,y_test)*100,2)))
    xgb_classifer_cm = confusion_matrix(y_test, xgb_prediction)
    st.write("Classification Report:Train data\n")
    plt.figure(figsize = [10,5])
    clf_report3 = classification_report(y_train,xgb_train_predict,
                                   labels=None,
                                   target_names=None,
                                   output_dict=True)
    sns.heatmap(pd.DataFrame(clf_report3).T, annot=True,cmap="cividis",fmt='.2f',cbar=True)
    st.pyplot()
    st.write("------------------------------------------------------\n")
    st.write("Classification Report:Test data\n")
    clf_report4 = classification_report(y_test,xgb_prediction,
                                   labels=None,
                                   target_names=None,
                                   output_dict=True)
    sns.heatmap(pd.DataFrame(clf_report4).T, annot=True,cmap="viridis",fmt='.2f',cbar=True)
    st.pyplot()
    st.write("Logloss:\n",log_loss(y_test, xgb_classifer.predict_proba(X_test)))
    st.write('Confusion Matrix \n', xgb_classifer_cm)
    #Precision Recall Curve
    st.write("Precision Recall Curve")
    plt.figure(figsize = [10, 8])
    PRCurve(xgb_classifer)
    st.pyplot()
    
    st.write("Plotting Confution Matrix")
    plot_confusion_matrix(xgb_classifer, X_test,y_test)
    st.pyplot()
    
    st.write("AUC ROC Curve")
    plot_roc_curve(xgb_classifer, X_test, y_test)
    st.pyplot()
    
    st.write("Feature Importance Learnt by the Model")
    fig = plt.figure(figsize = (14, 9))
    ax = fig.add_subplot(111)
    xgb_classifer= XGBClassifier(n_estimators=200,max_depth=6,booster="gbtree",learning_rate=0.005)
    xgb_classifer.fit(X_train,y_train)
    colours = plt.cm.viridis(np.linspace(0, 1, 22))
    ax = plot_importance(xgb_classifer, height = 1, grid = False, color = colours, \
                     show_values = False, importance_type = 'cover', ax = ax, max_num_features=22);
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
    ax.set_xlabel('importance score', size = 16);
    ax.set_ylabel('features', size = 16);
    ax.set_yticklabels(ax.get_yticklabels(), size = 12);
    ax.set_title('Ordering of features by importance to the model learnt', size = 20);
    st.pyplot()
    st.write("Model Learning Curve")
    p11=plot_learning_curve(xgb_classifer,'XGboost_classifer', X_train, y_train)
    st.plotly_chart(p11)



def predict_func():
    df=user_input_features()
    st.write(df)
    for column in df.columns:
        df[column] = le_encoder.fit_transform(df[column])

    
    html_temp = """
     <div style="background-color:royalblue;padding:10px;border-radius:10px">
     <h1 style="color:white;text-align:center;">After Entering the inputs press predict Button</h1>
         </div>  """
    components.html(html_temp)
    #st.write("After Entering the inputs press predict Button")
    if st.button("Predict"):
        y_test_pred=loaded_model.predict(df)
        return y_test_pred
        
    
    
        
###############################################Streamlit Main###############################################

def main():
    # set page title
    
    
            
    # 2. horizontal menu with custom style
    selected = option_menu(menu_title=None, options=["Home", "Projects","Report" ,"About"], icons=["house", "book","app-indicator","envelope"],  menu_icon="cast", default_index=0, orientation="horizontal", styles={"container": {"padding": "0!important", "background-color": " #f08080 "},"icon": {"color": "blue", "font-size": "25px"}, "nav-link": {"font-size": "25px","text-align": "left","margin": "0px","--hover-color": "#eee", },           "nav-link-selected": {"background-color": "green"},},)
    
    #horizontal Home selected
    if selected == "Home":
        image= Image.open("home_img.jpg")
        st.image(image,use_column_width=True)
        st.title("Home")   
        #st.sidebar.title("Home")        
        with st.sidebar:
            st_lottie(lottie_hello, key="hello")
            #image= Image.open("Home1.png")
            #add_image=st.image(image,use_column_width=True)
            
            
        
        def header(url):
            st.sidebar.markdown(f'<p style="background-color:royalblue ;color:white;font-size:15px;border-radius:1%;">{url}', unsafe_allow_html=True)    
        html_45=""" A Quick Youtube Video for understanding the Difference between Edible and Poisonous Mushrooms for Educational Purpose ."""
        header(html_45)
        st.sidebar.video("https://www.youtube.com/watch?v=WD0sGwIhJNA")
        with st.sidebar:
            #image= Image.open("Home1.png")
            st.write('This Machine Learning Project Done By Author@ Siddhartha Sarkar')
            st.write('Data Scientist ')
        st.balloons()
        
        html_temp = """
		<div style="background-color:royalblue;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;"> Edible and Poisonous Mushroom Detection Using Machine Learning</h1>
		</div>  """
        
		
        components.html(html_temp)
        def header(url):
            st.markdown(f'<p style="background-color:royalblue ;color:black;font-size:20px;border-radius:2%;">{url}', unsafe_allow_html=True)
        
        html_temp11 = """
		 “Nature alone is antique and the oldest art a mushroom.” ~ Thomas Carlyle

    Mushrooms!! Creamy Mushroom Bruschetta, Mushroom Risotto, Mushroom pizza, Mushrooms in a burger, 
    and what not! Just by hearing the names of these dishes, people be drooling! Their flavor is one 
    reason that takes the dish to the next level!

    But have you ever wondered if the mushroom you eat is healthy for you? From over 50,000 species of 
    mushrooms only in North America, how will you classify the mushroom as edible or poisonous? Poisonous 
    mushrooms can be hard to identify in the wild!
        
		  """
        
		
        header(html_temp11)
        def header(url):
            st.markdown(f'<p style="background-color:#A9A9A9 ;color:black;font-size:20px;border-radius:2%;">{url}', unsafe_allow_html=True)
        html_temp12 = """
		 Introductions:<br>
    Mushrooms are one of the healthiest food on the planet that grow without any efforts. 
    It can grow on/in the ground or on other planets. Mushroom used as an ingredient in most food industry, 
    it has great benefits to our body, where it contains most potent nutrients on the plant such as calcium, 
    phosphorus, vitamins and proteins. Mushroom is used to treat cancer, eradicating viruses, increase immunity 
    system, lose weight, and for good diet programs. Recently, the use of mushroom has been increased by people. 
    However, mushroom can be classified as edible or inedible (poisonous). There are many types of mushrooms and 
    50-100 types cannot be eaten or we can say that most of mushrooms cannot be eaten and eating collected mushroom 
    directly without knowing it’s type is a big mistake and the effect of eating inedible mushroom range from simple 
    symptoms to death. People is looking at the physical characteristic of the mushroom such as shape, neck length, 
    head diameter, size, color and its environment to decide whether its edible or inedible. Huge amount of mushroom’s 
    data collected over years and applications developed to classify mushrooms. Different classification techniques 
    developed and improved to give more accurate decision. The classification algorithms compared for the highest 
    accuracy.    
		  """
        
		
        header(html_temp12)
        
        
        image= Image.open("mushroom_parts.png")
        st.image(image,use_column_width=False)
        def header(url):
            
            st.markdown(f'<p style="background-color:#A9A9A9 ;color:black;font-size:20px;border-radius:2%;">{url}', unsafe_allow_html=True)
        html_temp13 = """
		 Various Parts Of the Mushroom Plant
         Typically, a Mushroom has six different parts. These include:<br>
    -1>Cap: This is the part that gives the fungi its umbrella shape. The cap comes in a variety of colors, 
    including white, brown, and yellow. In the same way that umbrellas protect us from the heat of the sun, rain, 
    and other harsh weather conditions, the mushroom cap protects the pores or gills where mushroom spores are produced. 
    And what are spores? Well, consider them as mushroom “seeds,” although technically speaking mushrooms 
    don’t have seeds.<br>
    -2>Gills, Pores, or Teeth: Have you ever seen a fish’s gills? Mushroom gills look something like that. 
    The gills are also called teeth or pores. The gill is a structure that appears right under the mushroom 
    cap and produces spores.<br>
    -3>Ring: The ring (also known as the annulus) is a partial veil that is left on the stem. 
    It is an extra layer of protection for the spores that grow when the mushroom is still very young. 
    When the cap grows out and breaks through the veil, the remnant is what forms the ring around the stem.<br>
    -4>Stipe or Stem: The stipe or stem is the long, vertical part of the mushroom that holds the cap above the ground. 
    Mushrooms growing in the wild propagate when the wind scatters the spores. For this reason, the cap and gills need 
    to be held high enough from ground level by the stem, so that when the spores drop down, they can be carried away 
    easily by wind.<br>
    ->Note that in some mushrooms, the spores grow right down the sides of the stem. Oyster mushrooms are a good example. 
    Also, the way that a stem is attached to the cap can be an important clue in identifying a mushroom. For example, 
    a morel’s stem will attach to the inside of the hollow cap, whereas a false morel’s stem will attach to the bottom 
    of the cap.<br>
    -5>Volva: Mushrooms are covered in a protective veil as they grow out of the ground. 
    This protective veil is called the volva. The mushroom pushes through the volva as it matures, 
    leaving parts of the veil at the bottom of the stem.<br>
    -6>Mycelium: The mycelium is a collection of thin hair-like strands that grow outward and downward 
    into the soil in search of nutrients. The mycelium of mushrooms acts like the roots in flowering plants and can 
    produce new mushrooms when the conditions are suitable.     
		  """
        
		
        header(html_temp13)
        
        
        
        
        
        
        def plot11():
            
            import plotly.graph_objects as go

            values = [['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat','class'], #1st col
            ["bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s",
            "fibrous=f,grooves=g,scaly=y,smooth=s",
            "brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y",
            "bruises=t,no=f ",
            "  almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s",
            " attached=a,descending=d,free=f,notched=n",
            "  close=c,crowded=w,distant=d ",
            "broad=b,narrow=n",
            " black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y",
            " enlarging=e,tapering=t ",
            "  bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=? ",
            "fibrous=f,scaly=y,silky=k,smooth=s",'fibrous=f,scaly=y,silky=k,smooth=s','brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y',
            " brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y",'partial=p,universal=u','brown=n,orange=o,white=w,yellow=y',
            'none=n,one=o,two=t','cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z ',
            'black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y ','abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y',
            ' grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d','Different Class of Mushrooms']]

            fig = go.Figure(data=[go.Table(
            columnorder = [1,2],
            columnwidth = [80,400],
            header = dict(
            values = [['<b>Columns of<br>  Dataset </b>'],
                  ['<b>Attribute Information</b>']],
            line_color='red',
            fill_color='royalblue',
            align=['left','center'],
            font=dict(color='black', size=12),
            height=40
                  ),
            cells=dict(
            values=values,
            line_color='red',
            fill=dict(color=['pink', 'lightskyblue']),
            font=dict(color='black', size=12),
            align=['left', 'center'],
              font_size=12,
            height=20)
              )
           ])
            return fig

        p11=plot11()
        st.write("About Dataset")
        st.write('The data set composes of 8124 number of rows of data records and 22 attributes. \
                 Each mushroom species is identified as class of edible and poisonous. These rows are distributed as 4208 edible mushrooms and 3916 poisonous mushrooms')
        st.plotly_chart(p11)
        
        def plot12():
            import plotly.figure_factory as ff
            df_sample = mushroom_data.iloc[0:10,0:9]
            colorscale = [[0, '#4d004c'],[.5, '#f2e5ff'],[1, '#ffffff']]
            font_colors=[[0,'#ffffff'], [.5,'#000000'], [1,'#000000']]
            fig =  ff.create_table(df_sample,colorscale=colorscale,index=True,font_colors=['#ffffff', '#000000','#000000'])
            
            return fig
        p12=plot12()
        st.write("Data Table")
        st.plotly_chart(p12)

        def header(url):
            st.markdown(f'<p style="background-color:#A9A9A9 ;color:black;font-size:20px;border-radius:2%;">{url}', unsafe_allow_html=True)
        html_temp111 = """
		About The Project :<br>
        In this project,I will analyze the data and build different machine learning models that will detect
        if the mushroom is edible or poisonous by its specifications like cap shape, cap color, gill color, etc.
        using different classifiers. Its important to get better decision to avoid side effect of eating inedible 
        mushrooms.


        """
        header(html_temp111)
        st.markdown("""
                #### Tasks Perform by the app:
                + App covers the most basic Machine Learning task of  Analysis, Correlation between variables,project report.
                + Machine Learning on different Machine Learning Algorithms, building different models and lastly  prediction.
                
                """)
                
    #Horizontal About selected
    if selected == "About":
        #st.title(f"You have selected {selected}")
        
        st.sidebar.title("About")
        with st.sidebar:
            image= Image.open("About-Us-PNG-Isolated-Photo.png")
            add_image=st.image(image,use_column_width=True)
        
        st_lottie(about_1,key='ab1')
        #image2= Image.open("about.jpg")
        #st.image(image2,use_column_width=True)
        st.sidebar.write("This Youtube Video Shows and Describes Different Kind Of Mushrooms for Learning Purpose ")
        st.sidebar.video('https://www.youtube.com/watch?v=6PNq6paMBXU')
        html_temp = """
		<div style="background-color:royalblue;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">This is a Mushrooms Class Detection Project</h1>
		</div>  """
        
		
        components.html(html_temp)
        def header(url):
            st.markdown(f'<p style="background-color:#A9A9A9 ;color:black;font-size:30px;border-radius:2%;">{url}', unsafe_allow_html=True)
        html_99   =  """  In this Project I tried to build  Machine learning models\
                    that will detect if the mushroom is edible or poisonous by its specifications like cap shape,\
                    cap color, gill color, etc. using different classifiers. Its important to get better decision to avoid side effect of eating inedible mushrooms"""
        header(html_99)
        
        st.markdown("""
                    #### + Project Done By :        
                    #### @Author Mr. Siddhartha Sarkar
                    
        
                    """)
        st.snow()
        
        #st.sidebar.markdown("[ Visit To Github Repositories](.git)")
    #Horizontal Project_Report selected
    if selected == "Report":
        #report_1
        st.title("Profile Report")
        st.sidebar.title("Project_Profile_Report")
        
        with st.sidebar:
            st_lottie(report_1, key="report1")
            #image= Image.open("report_project.png")
            #add_image=st.image(image,use_column_width=True)
        
        st.balloons()    
        html_temp = """
		<div style="background-color:royalblue;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Simple EDA App with Streamlit Components</h1>
		</div>  """
        
		
        components.html(html_temp)
        html_temp1 = """
			<style>
			* {box-sizing: border-box}
			body {font-family: Verdana, sans-serif; margin:0}
			.mySlides {display: none}
			img {vertical-align: middle;}
			/* Slideshow container */
			.slideshow-container {
			  max-width: 1500px;
			  position: relative;
			  margin: auto;
			}
			/* Next & previous buttons */
			.prev, .next {
			  cursor: pointer;
			  position: absolute;
			  top: 50%;
			  width: auto;
			  padding: 16px;
			  margin-top: -22px;
			  color: white;
			  font-weight: bold;
			  font-size: 18px;
			  transition: 0.6s ease;
			  border-radius: 0 3px 3px 0;
			  user-select: none;
			}
			/* Position the "next button" to the right */
			.next {
			  right: 0;
			  border-radius: 3px 0 0 3px;
			}
			/* On hover, add a black background color with a little bit see-through */
			.prev:hover, .next:hover {
			  background-color: rgba(0,0,0,0.8);
			}
			/* Caption text */
			.text {
			  color: #f2f2f2;
			  font-size: 15px;
			  padding: 8px 12px;
			  position: absolute;
			  bottom: 8px;
			  width: 100%;
			  text-align: center;
			}
			/* Number text (1/3 etc) */
			.numbertext {
			  color: #f2f2f2;
			  font-size: 12px;
			  padding: 8px 12px;
			  position: absolute;
			  top: 0;
			}
			/* The dots/bullets/indicators */
			.dot {
			  cursor: pointer;
			  height: 15px;
			  width: 15px;
			  margin: 0 2px;
			  background-color: #bbb;
			  border-radius: 50%;
			  display: inline-block;
			  transition: background-color 0.6s ease;
			}
			.active, .dot:hover {
			  background-color: #717171;
			}
			/* Fading animation */
			.fade {
			  -webkit-animation-name: fade;
			  -webkit-animation-duration: 1.5s;
			  animation-name: fade;
			  animation-duration: 1.5s;
			}
			@-webkit-keyframes fade {
			  from {opacity: .4} 
			  to {opacity: 1}
			}
			@keyframes fade {
			  from {opacity: .4} 
			  to {opacity: 1}
			}
			/* On smaller screens, decrease text size */
			@media only screen and (max-width: 300px) {
			  .prev, .next,.text {font-size: 11px}
			}
			</style>
			</head>
			<body>
			<div class="slideshow-container">
			<div class="mySlides fade">
			  <div class="numbertext">1 / 3</div>
			  <img src="https://www.w3schools.com/howto/img_nature_wide.jpg" style="width:100%">
			  <div class="text">Caption Text</div>
			</div>
			<div class="mySlides fade">
			  <div class="numbertext">2 / 3</div>
			  <img src="https://www.w3schools.com/howto/img_snow_wide.jpg" style="width:100%">
			  <div class="text">Caption Two</div>
			</div>
			<div class="mySlides fade">
			  <div class="numbertext">3 / 3</div>
			  <img src="https://www.w3schools.com/howto/img_mountains_wide.jpg" style="width:100%">
			  <div class="text">Caption Three</div>
			</div>
			<a class="prev" onclick="plusSlides(-1)">&#10094;</a>
			<a class="next" onclick="plusSlides(1)">&#10095;</a>
			</div>
			<br>
			<div style="text-align:center">
			  <span class="dot" onclick="currentSlide(1)"></span> 
			  <span class="dot" onclick="currentSlide(2)"></span> 
			  <span class="dot" onclick="currentSlide(3)"></span> 
			</div>
			<script>
			var slideIndex = 1;
			showSlides(slideIndex);
			function plusSlides(n) {
			  showSlides(slideIndex += n);
			}
			function currentSlide(n) {
			  showSlides(slideIndex = n);
			}
			function showSlides(n) {
			  var i;
			  var slides = document.getElementsByClassName("mySlides");
			  var dots = document.getElementsByClassName("dot");
			  if (n > slides.length) {slideIndex = 1}    
			  if (n < 1) {slideIndex = slides.length}
			  for (i = 0; i < slides.length; i++) {
			      slides[i].style.display = "none";  
			  }
			  for (i = 0; i < dots.length; i++) {
			      dots[i].className = dots[i].className.replace(" active", "");
			  }
			  slides[slideIndex-1].style.display = "block";  
			  dots[slideIndex-1].className += " active";
			}
			</script>
			"""
        components.html(html_temp1)
        st.sidebar.title("Navigation")
        menu = ['None',"Sweetviz","Pandas Profile"]
        choice = st.sidebar.radio("Menu",menu)
        if choice == 'None':
            st.markdown("""
                        #### Kindly select from left Menu.
                       # """)
        elif choice == "Pandas Profile":
            st.subheader("Automated EDA with Pandas Profile")
            #data_file= st.file_uploader("Upload CSV",type=['csv'])
            df = mushroom_data
            st.table(df.head(10))
            if st.button("Generate Profile Report"):
                profile= ProfileReport(df)
                st_profile_report(profile)
            
        elif choice == "Sweetviz":
            st.subheader("Automated EDA with Sweetviz")
            #data_file = st.file_uploader("Upload CSV",type=['csv'])
            df =mushroom_data
            st.dataframe(df.head(10))
            if st.button("Generate Sweetviz Report"):

				# Normal Workflow
                report = sv.analyze(df)
                report.show_html()
                st_display_sweetviz("SWEETVIZ_REPORT.html")  
    			
		       
                
    			
            
		
        
    #Horizontal Project selected
    if selected == "Projects":      
            with st.sidebar:
                st_lottie(project_1, key="pro1")
                #project_1=load_lottieurl(project_url_1)
                #project_2=load_lottieurl(project_url_2)
                #image= Image.open("project_side.jpg")
                #add_image=st.image(image,use_column_width=True)
                #image= Image.open("project_hgff.png")
                #add_image=st.image(image,use_column_width=True)
            import time

                
            st_lottie(project_2, key="pro22")
            st.title("Projects")              
            #image2= Image.open("project11.jpeg")
            #st.image(image2,use_column_width=True)
            my_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1)
            st.sidebar.title("Navigation")
            menu_list1 = ['Exploratory Data Analysis',"Prediction With Machine Learning"]
            menu_Pre_Exp = st.sidebar.radio("Menu For Prediction & Exploratoriy", menu_list1)
            
            #EDA On Document File
            if menu_Pre_Exp == 'Exploratory Data Analysis' and selected == "Projects":
                    st.title('Exploratory Data Analysis')

                    
                    
                    menu_list2 = ['None', 'Basic_Statistics','Basic_Plots','Interactive_plots','Multi_level_SunBurst_plots','Feature Engineering']
                    menu_Exp = st.sidebar.radio("Menu EDA", menu_list2)

                    
                    if menu_Exp == 'None':
                        st.markdown("""
                                    #### Kindly select from left Menu.
                                   # """)
                    
                    elif menu_Exp == 'Basic_Plots':
                        label_analysis()
                    elif menu_Exp == 'Multi_level_SunBurst_plots':
                        label_analysis1()
                    elif menu_Exp == 'Interactive_plots':
                        label_analysis2()
                    elif menu_Exp == 'Feature Engineering':
                        label_analysis3()
                    elif menu_Exp == 'Basic_Statistics':
                        label_analysis4()   
                    

            elif menu_Pre_Exp == "Prediction With Machine Learning" and selected == "Projects":
                    st.title('Prediction With Machine Learning')
                    
                    menu_list3 = ['Checking ML Method And Various Matrices' ,'Prediction' ]
                    menu_Pre = st.radio("Menu Prediction", menu_list3)
                    
                    #Checking ML Method And Accuracy
                    if menu_Pre == 'Checking ML Method And Various Matrices':
                            st.title('Checking Accuracy  And Various Matrices On Different Algorithms')
                            #dataframe=data_func(data)
                            
                            if st.checkbox("View data"):
                                st.write(mushroom_data)
                            model = st.selectbox("ML Method",[ 'XGB Classifier', 'Random Forest Classifier'])
                            #vector= st.selectbox("Vector Method",[ 'CountVectorizer' , 'TF-IDF'])

                            if st.button('Analyze'):
                                #Logistic Regression 
                                #if model=='Logistic Regression':
                                    #logistic_regression(get_data_class(final_data))
                                    #st.write(data)
                                

                                #XGB Classifier
                                if model=='XGB Classifier':
                                    XGboost_classifier()
                                                                           
                                    #st.write(data)
                               
                                
                                
                                #Random Forest Classifier & CountVectorizer
                                elif model=='Random Forest Classifier':
                                    randomforest_classifier()
                                    #st.write(data)
                    #Checking ML Method And Accuracy
                    #elif menu_Pre == 'Checking Regression Method And Accuracy':
                            #st.title('Checking Accuracy On Different Algorithms')
                            #dataframe=data_func(data)
                            
                            #if st.checkbox("View data"):
                                #st.write(data)
                            #model = st.selectbox("ML Method",['XGboost_regressor', 'Random Forest Regressor'])
                            #vector= st.selectbox("Vector Method",[ 'CountVectorizer' , 'TF-IDF'])

                           # if st.button('Analyze'):
                                #Logistic Regression 
                                #if model=='XGboost_regressor':
                                   # XGboost_regressor(get_data_reg(dataframe))
                                    
                                

                                #XGB Classifier & CountVectorizer
                                #elif model=='Random Forest Regressor':
                                    #randomforest_regressor(get_data_reg(dataframe))                                    
                                
                              
                                          
                    elif menu_Pre == 'Prediction':
                        st.title('Prediction')
                        with st.spinner('Wait for it...'):
                            time.sleep(5)
    
                           
                        #df= user_input_features()
                        
                        result_pred = predict_func()
                        if (result_pred==0):
                            st.write("The Mushroom is Edible")
                            image1= Image.open("edible_mush.jpg")
                            st.image(image1,use_column_width=True)
                        elif (result_pred==1):
                            st.write("The Mushroom is Poisonous")
                            image1= Image.open("poison_mush.jpg")
                            st.image(image1,use_column_width=True)
                            
                        
                        st.success('Done!')       
                        #st.success('The Transaction is --> {}'.format(result_pred))
                        
                            
                                

                                                      
if __name__=='__main__':
    main()            
            
            

