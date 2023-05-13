import streamlit as st
from sklearn.model_selection import train_test_split
#Import svm model
from sklearn import svm
from sklearn import metrics

from pages import home, prepData, functions

def app():
    st.markdown("""
        ## Classification models
        ### Defining the model - SVM
        ___Support Vector Machines (SVM)___ classifier separates data points using a hyperplane with the largest amount 
        of margin. The SVM classifier is also known as a discriminative classifier. 
        SVM finds an optimal hyperplane which helps in classifying new data points. 
    
    """)
    feats = home.feats
    enseigne = home.enseigne
    df_data = home.df_data
    label = df_data.Display
    df_rescaled = prepData.normalized_features
    with st.expander("Rescaled data"):
        st.dataframe(df_rescaled)

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(df_rescaled, label, test_size=0.3,random_state=42) # 70% training and 30% test 
    kernel = st.selectbox("Select Kernel", options=['rbf', 'linear'])
    
    if kernel:
        #Create a svm Classifier
        if 'clf' not in st.session_state:
            clf = svm.SVC(kernel= kernel)
            st.session_state['clf'] = clf 
        #Train the model using the training sets
        if 'fitted' not in st.session_state:
            fitted = st.session_state['clf'].fit(X_train, y_train)
            st.session_state['fitted'] = fitted
        
        with st.expander('Evaluation'):
            if 'y_pred' not in st.session_state:
                y_pred = st.session_state['clf'].predict(X_test)
                st.session_state['y_pred'] = y_pred

            acc = metrics.accuracy_score(y_test, st.session_state['y_pred'])
            accuracy = "Accuracy: "+str(float("{0:.2f}".format(acc*100)))+"%"
            
            st.info(accuracy)
            col1, col2, col3, col4 = st.columns(4)
            input_data = []

            with col1:
                min_val1 = int(min(df_data['cor_sales_in_vol']))
                max_val1 = int(max(df_data['cor_sales_in_vol']))
                Cor_sales_in_vol = st.number_input('Cor_sales_in_vol', min_value=min_val1, max_value=max_val1, value=max_val1, key=0, step=1)
                input_data.append(Cor_sales_in_vol)
            with col2:
                min_val2 = float(min(df_data['cor_sales_in_val']))
                max_val2 = float(max(df_data['cor_sales_in_val']))
                Cor_sales_in_val = st.number_input('Cor_sales_in_val', min_value=min_val2, max_value=max_val2, value=max_val2, key=1)
                input_data.append(Cor_sales_in_val)
            with col3:
                min_val3 = int(min(df_data['CA_mag']))
                max_val3 = int(max(df_data['CA_mag']))
                CA_mag = st.number_input('CA_mag', min_value=min_val3, max_value=max_val3, value=max_val3, key=2)
                input_data.append(CA_mag)
            with col4:
                min_val4 = int(min(df_data['value']))
                max_val4 = int(max(df_data['value']))
                value_opt = st.number_input('value', min_value=min_val4, max_value=max_val4, value=max_val4, key=3)
                input_data.append(value_opt)

            col5, col6, col7 = st.columns(3)
            with col5:
                options_enseign = ['CORA', 'LECLERC', 'AUCHAN', 'CARREFOUR', 'CASINO', 'SUPER U', 'GEANT', 'CARREFOUR MARKET', 'FRANPRIX', 'INTERMARCHE', 'ECOMARCHE', 'MONOPRIX', 'SIMPLY MARKET', 'OTHERS', 'MATCH', 'PRISUNIC', 'HYPER U', 'SHOPI', 'MARCHE U']
                ens_opt = st.selectbox('ENSEIGNE', options=options_enseign, key=0)
                ens_opt = options_enseign.index(ens_opt)
                input_data.append(ens_opt)

            with col6:
                min_val6 = int(min(df_data['VenteConv']))
                max_val6 = int(max(df_data['VenteConv']))
                VenteConv = st.number_input('VenteConv', min_value=min_val6, max_value=max_val6, value=max_val6, key=4)
                input_data.append(VenteConv)

            with col7:
                options_feature = ['No Feat', 'Feat']
                Feat_opt = st.selectbox('Feature', options=options_feature, key=1)
                Feat_opt = options_feature.index(Feat_opt)
                input_data.append(Feat_opt)
            display = ['No_Displ', 'Displ']
            if input_data:
                st.write("___Input Data___")
                st.write(input_data)
                rescaled_opt = prepData.option_prepare
                if st.button("Predict / Classify"):
                    if 'clf' in st.session_state:
                        prediction = functions.predict_display(input_data, rescaled_opt, st.session_state['clf'])
                        prediction_phrase = "Predicted class: "+prediction
                        st.info(prediction_phrase)
                


    st.markdown("""
        ### Defining the model - LDA
        ___Linear Discriminant Analysis (LDA)___ is a predictive modeling algorithm for multi-class classification. 
        It can also be used as a dimensionality reduction technique, 
        providing a projection of a training dataset that best separates the examples by their assigned class.

        * __Without MDLP__: `accuracy = 0,736`
        * __With MDLP__: `accuracy = 0.745`
    
    """)