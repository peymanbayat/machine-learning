import streamlit as st
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from pages import home, functions

def app():
    # st.title("## Data visualization")
    st.markdown("""
    ## Data visualization
    We want to plot data distributions to have better grasp of how our data features correlate.
    
    """)
    df_data = home.df_data
    with st.expander("Data"):
        st.dataframe(df_data)

    col1, col2= st.columns(2)
    with col1:
        fig = plt.figure()
        sns.countplot(x='Display', data=df_data).set_title("Target variable distribution")
        st.pyplot(fig)

    with col2:
        fig = plt.figure()
        sns.countplot(x='Display', data=df_data, hue="Feature").set_title("Target variable distribution with Feature column")
        st.pyplot(fig)

    entities = list(df_data['ENSEIGNE'].unique())
    with st.expander("ENSEIGNE data distribution"):
        description = 'ENSEIGNE entities | Total: '+str(len(entities))
        options = st.multiselect(description, options= entities, default=entities[:5])
        selected = 'Selected entities : '+str(len(options))
        st.write(selected)
        fig = plt.figure()
        df = df_data[df_data['ENSEIGNE'].isin(options)]
        sns.countplot(x='ENSEIGNE', data=df).set_title("Feature variable distribution")
        st.pyplot(fig)

    
    
    with st.expander("Column values"):
        col3, col4 = st.columns(2)
        with col3:
            unique_vals_col = functions.unique_vals(df_data) 
            st.write("_Number of unique values per data column_")
            st.write(unique_vals_col)

        with col4:
            per_uniq_vals = functions.per_unique_vals(df_data)
            st.write("_Percentage of unique values per data column_")
            st.write(per_uniq_vals)

    # st.markdown("""
    #     ### Encode Text Data

    #     Machine learning models require all input and output variables to be numeric.
    #     This means that if our data contains categorical data, which it does, we must encode 
    #     it to numbers before we can fit and evaluate our model.
    # """)

    # catg_options = st.multiselect("Categorical variables", options = ['ENSEIGNE', 'Feature'], default = ['ENSEIGNE', 'Feature'])
    # if st.button('Encode variables'):
    #     df_data = functions.encode(df_data, catg_options)
    #     st.success("Categorical variables encoded sucessfully!")
    #     st.dataframe(df_data)


        