import streamlit as st
from sklearn import preprocessing
from pages import home, functions

def app():
    st.title("Data Preparation")
    df_data = home.df_data
    

    st.markdown("""
        ### Encode Text Data

        Machine learning models require all input and output variables to be numeric.
        This means that if our data contains categorical data, which it does, we must encode 
        it to numbers before we can fit and evaluate our model.
    """)
    with st.expander("Original Data"):
        st.dataframe(df_data)
    catg_options = st.multiselect("Categorical variables", options = ['ENSEIGNE', 'Feature', 'Display'], default = ['ENSEIGNE', 'Feature'])
    
    df_data = functions.encode(df_data, catg_options)
    
    with st.expander('Encoded data'):
        st.success("Categorical variables encoded sucessfully!")
        st.dataframe(df_data)

    st.write("### Rescaling: Normalization / Standardization")
    global label
    label = df_data['Display']
    features = df_data.drop('Display', axis=1)
    global option_prepare
    option_prepare = st.selectbox("How would you like to rescale data?", options=['Standardization', 'Normalization'])
    global normalized_features
    normalized_features = functions.rescale(features, option_prepare)
    with st.expander("Rescaled data"):
        st.dataframe(normalized_features)

    
