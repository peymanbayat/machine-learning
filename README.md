# ML Display Modeling
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/nainiayoub/machine-learning-display-modeling/main/app.py)

Target variable prediction web app, with different supervised learning algorithms.
The idea is to make a prediction of the target variable `Display` using as independent variables `X1...X7`, from the data provided in [`./data`](https://github.com/nainiayoub/machine-learning-display-modeling/tree/main/data)

## Project Demo
https://user-images.githubusercontent.com/50157142/153604034-a3507c3e-14ae-44c5-b848-467fefb431e4.mp4

## Features
### Data visualization
We want to plot data distributions to have better grasp of how our data features correlate.

### Data Encoding
Machine learning models require all input and output variables to be numeric. This means that if our data contains categorical data, which it does, we must encode it to numbers before we can fit and evaluate our model.

### Data Rescaling
The preprocessed data may contain attributes with a mixtures of scales for various quantities.
Many machine learning methodsare more effective if the data attributes have the same scale. 
Two popular data scaling methods are `normalization` and `standardization`.

## Defined models
