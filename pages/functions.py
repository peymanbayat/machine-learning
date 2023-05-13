import streamlit as st
import pandas as pd
from sklearn import preprocessing
import numpy as np

# read data
@st.cache(allow_output_mutation=True)
def read_data(data):
    df_data = pd.read_csv(data, sep=";")
    # replace dataframe header with first row
    new_header = df_data.iloc[0]
    df_data = df_data[1:]
    df_data.columns = new_header

    return df_data

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

@st.cache
def unique_vals(df_data):
    col_unique_vals = list(df_data.nunique())
    columns = df_data.columns
    unique_vals = [columns[i]+' : '+str(col_unique_vals[i]) for i in range(len(columns))]

    return unique_vals

@st.cache
def per_unique_vals(df_data):
    per_unique = []
    for i in range(len(df_data.columns)):
        uniqueVals = len(df_data[df_data.columns[i]].unique())
        lenOriginal = len(df_data[df_data.columns[i]])
        per = uniqueVals/lenOriginal * 100
        per = float("{0:.2f}".format(per))
        res = df_data.columns[i]+": "+str(per)+"%"
        per_unique.append(res)

    return per_unique

@st.cache(allow_output_mutation=True)
def encode(df_data, catg_options):
    for i in catg_options:
        feat = list(df_data[i].unique())

        for j in feat:
            index = feat.index(j)
            df_data[i] = df_data[i].replace([j], index)
    return df_data

@st.cache
def rescale(features, option):
    if option == 'Normalization':
        normalized_features = preprocessing.normalize(features)
    else:
        normalized_features = preprocessing.scale(features)

    df_normalized = pd.DataFrame(np.row_stack(normalized_features))
    df_normalized.columns = ['cor_sales_in_vol',	'cor_sales_in_val',	'CA_mag',	'value',	'ENSEIGNE',	'VenteConv',	'Feature']
    
    return df_normalized

@st.cache
def predict_display(input_data, rescaled_opt, clf):
    if rescaled_opt == 'Normalization':
        normalized_input_data = preprocessing.normalize(input_data)
    else:
        normalized_input_data = preprocessing.scale(input_data)
    to_np_array = np.array(normalized_input_data)
    dl = pd.DataFrame(to_np_array.reshape(-1, len(to_np_array)))
    res = clf.predict(dl)

    return res[0]