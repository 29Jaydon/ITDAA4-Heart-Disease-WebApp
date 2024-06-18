import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm, preprocessing

# creating the header and associated paragraphs
st.header(':orange[Heart Disease Prediction]', divider='red')
st.write('### Hi there👋')
st.write('### Welcome to the Heart Disease Prediction Website')
st.write('Please fill in the following patient details. If any test result is missing please fill in the median result.')
st.write('For some input fileds you may need to enter number value. Please refer to the instructions below. ')

st.header(':orange[Instructions]', divider='red')
st.write('For Chest Pain Type please use the following codes: 1= Male, 0=Female')
st.write('For Chest Pain Type please use the following codes:')
st.write(' 0 = Typical Angina, 1 = Atypical Angina, 2 = Non-Anginal Pain, 3 = Asymtopmatic')
st.write('For Fasting Blood Sugar >120 mg/dl please use the following codes: 1 = True, 0 = False')
st.write('For Resting ECG please use the following codes: 1 = Normal, 2 = Abnormal, 3 = Ventricular Hypertrophy')
st.write('For Exercise Induced Angina please use the following codes: 1= Yes, 0= No')
st.write('For slope please use the following codes: 1 = Upsloping, 2 = Flat, 3 = Downsloping')
st.write('For Atherosclerosis please use the following codes: 0 = Mild, 1 = Moderate, 3 = Severe')
st.write('For Status of Heart please use the following codes: 1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect')
st.write(' ')

# creating the input fields 
col1, col2 = st.columns(2)

age =col1.number_input("Age", min_value=0, max_value=110)
sex =col1.number_input("Sex",min_value=0, max_value=1)
cp =col1.number_input("Chest Pain Type ",min_value=0, max_value=3)
res_bp =col1.number_input("Resting Blood Pressure (in mm Hg)",min_value=0, max_value=220)
chol =col1.number_input("Serum Cholestrol (in mg/dl )",min_value=0)
fbs =col1.number_input("Fasting Blood Sugar >120 mg",min_value=0, max_value=1)
restecg =col1.number_input("Resting ECG",min_value=0, max_value=3)

thalach =col2.number_input("Max Heart Rate Achieved",min_value=10)
exang =col2.number_input("Exercise Induced Angina",min_value=0, max_value=1)
oldpeak =col2.number_input("ST Depression Induced by Exercise",min_value=0.0)
slope =col2.number_input("Slope of ST Segment",min_value=0, max_value=2)
ca =col2.number_input("Atherosclerosis",min_value=0, max_value=3)
Thal = col2.number_input("Status of Heart",min_value=1, max_value=3)

st.header(':orange[Prediction]', divider='red')

# python code to read-in the csv file 
df = pd.read_csv("heart.csv")

# data cleaning 
merged_col = 'age;sex;cp;trestbps;chol;fbs;restecg;thalach;exang;oldpeak;slope;ca;thal;target'
unmerged_col = merged_col.split(';')

df.rename(columns = { merged_col:'merged_col'}, inplace = True)
elements = df['merged_col'].str.split(';', expand= True )
elements.columns = unmerged_col 
df_clean = elements

df_clean.rename(columns =
    {'cp': 'cp_type','trestbps':'resting_bp', 'restecg': 'rest_ecg', 'thalach':'max_hr', 'thal':'status'},inplace= True)

df_clean=df_clean.astype(float)

# Creating the machine learning model
x=df_clean.drop('target',axis=1)
y=df_clean['target']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=0)
svm_model = svm.SVC()
svm_model.fit(x_train, y_train)
score=svm_model.score(x_train, y_train )

patient_arr = np.array([age, sex, cp, res_bp, chol, fbs, restecg, thalach, exang, oldpeak, slope ,ca, Thal])

# Model prediction
pred= svm_model.predict([patient_arr])

if (patient_arr[0] == 0) and (patient_arr[7] == 10):
    st.write('#### The Prediction of The Model Will Appear Here.')
else:
    if pred== [0.]:
        st.write('This Patient is :green[Not at risk] for heart disease 👍 :green[No further tests are required]')
    if pred== [1.]:
        st.write('This Patient :red[is at risk] for heart disease 👎 :red[further tests are required]')


