import streamlit as st
from multiapp import Multiapp
from pages import home, data, prepData, buildModels

app = Multiapp()

app.add_app("Home", home.app)
app.add_app("Data visualization", data.app)
app.add_app("Data preparation", prepData.app)
app.add_app("Defining models", buildModels.app)



app.run()