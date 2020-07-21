# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 14:51:33 2020

@author: sheik
"""
import streamlit as st
import pandas as pd
import base64
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
import cloudpickle
import pickle
import sweetviz

dat = st.file_uploader(' ',type=['csv','txt'])
if dat is not None:
    df = pd.read_csv(dat)
    va = ['predict']
    x = df.drop('predict',axis =1)
    y = df['predict']
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3)
    model = SGDRegressor()
    model.fit(xtrain,ytrain)
    
    if st.button('show report'):
        repo = sweetviz.analyze([df,'data'],target_feat='predict')
        repo.show_html('report.html')
    
    def get_table_download_link(model):
        """Generates a link allowing the data in a given panda dataframe to be downloaded
        in:  dataframe
        out: href string
        """
        csv = pickle.dump(model,open('model4','wb'))
        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/pkl;base64,{b64}">Download csv file</a>'
        return href
    def get_table_download_link1(model):
        """Generates a link allowing the data in a given panda dataframe to be downloaded
        in:  dataframe
        out: href string
        """
        csv = pickle.dump(model,open('model4','wb'))
        #b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/pkl;utf8,{csv}">Download csv file</a>'
        return href
    
    def get_table_download(df):
        """Generates a link allowing the data in a given panda dataframe to be downloaded
        in:  dataframe
        out: href string
        """
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        #href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
        return b64
    
    if st.button('save model'):
        st.write(pickle.dump(model,open('model301','wb')))
    
    if st.button('create'):
    
        st.markdown(get_table_download_link1(model), unsafe_allow_html=True)
    
    
