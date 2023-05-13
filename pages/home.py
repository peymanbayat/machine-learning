import streamlit as st
import pandas as pd
import numpy as np
from pages import functions

def app():
    st.markdown("""
        
    
    """)
    st.title("Machine Learning Project")
    # st.write(" _MS: Data Science | Module: Théorie des sondages | Supervized by: Pr. Ahmed Mousannif_")
    st.markdown("""
        ___Solution by Ayoub Nainia___ |  [LinkedIn](https://www.linkedin.com/feed/) | [GitHub](https://github.com/nainiayoub)
        
        The idea is to make a prediction of the target variable `Display (Y)`
        using as independent variables `X1...X7`, from the data provided below.
        The dataframe header has been replaced with the first row of data.
        
    """)

    data = './data/new_Base_CDM_balanced_V2.csv'

    global df_data
    df_data = functions.read_data(data)
    df = functions.read_data(data)
    global feats
    feats = list(df['Feature'].unique())
    global enseigne
    enseigne = list(df['ENSEIGNE'].unique())

    df_to_display = df_data
    options = st.multiselect('Dataframe columns', options= list(df_to_display.columns), default=list(df_to_display.columns))
    # st.write(options)
    st.dataframe(df_to_display[options])
    csv = functions.convert_df(df_data)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='ew_Base_CDM_balanced_V2.csv',
        mime='text/csv',
    )