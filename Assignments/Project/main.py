import streamlit as st
import pickle as pkl
import pandas as pd
from pandas.io.json import json_normalize
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score
import warnings
warnings.filterwarnings("ignore")

st.title("Team 3: DSCI_633-Foundations Of Data Science and Analytics Project")
st.header("Group Members: ")
st.subheader("Omkar Khanvilkar")
st.subheader("Pranav Nair")
st.subheader("Sujan Dutta")
st.subheader("Varun Tandon")

# Loading the final model
model = pkl.load(open(r"./models/grid_search_xgb_model.pkl","rb"));


# Load the preprocessed test data
@st.cache
def load_data():
    data  = pd.read_json(r"./data/test_df_preprocessed.json")
    return data
test_df = load_data();

# Take from the user a user_id
user_id = int(st.text_input("Enter the user id below."))

# If the user_id exists in the preprocessed test data index, go inside the if loop
if(user_id in test_df.index.tolist()):
        
    # Get the column values for this user_id from the preprocessed_test data
    test = pd.DataFrame(test_df.iloc[user_id, :]).T

    # Predict on this data using the model
    result = model.predict_proba(test)[:,1][0]

    # Output the appropriate label according to the resultant probability obtained
    if result>=0.5:
        st.write(f"The predicted segment is: pos")
    else:
        st.write(f"The predicted segment is: neg")

# Else print that the user_id does not exist         
else:
    st.write("User id does not exist");
