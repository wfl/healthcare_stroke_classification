##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: EDA.py
Purpose: Perform Exploratody Data Analysis

Tools: Pandas. matplotlib, and seaborn
References:

Dataset:
    The data was provided by McKinsey & Company for a hackathon event hosted by 
    Analytics Vidhya.   

Total number of variables = 11
Target = stroke
Number of entries = 43400
Descrition of variables:
    id = Patient ID
    gender = Gender of the patient 
    age = Patient’s age
    hypertention = Suffering from hypertension?
    heart_disease = Presence of heart disease? 
    ever_married = Ever Married?
    work_type = Type of occupation
    residence_type = Area type of resident 
    avg_glucose_level = Average glucose level after a meal
    bmi = Body mass index
    smoking_status = Smoking status
    Stroke = Has suffered or is suffering from stroke?
"""


# Standard Libraries
import pandas as pd

#   Visualization
import matplotlib.pyplot as plt
import seaborn as sns
#   
# Data Pre-processing
# Load data
path = 'data/'
df = pd.read_csv(path+'train_2v.csv')

# Set patients id as index as well
df.set_index([df.index.values,'id'], drop=True, inplace=True)

# Data Exploration
# Size of data 
#   (43400 observations in train data; 18601 observations in test data)
#   THere are 12 variables ( The variable strock is the target)
df.shape

# Look at the data
df.head(5)

# # Find out the high level information in the data, 
#   i.e., the variables' data type, sizes and check for missing values
df.info()

#df_test.info()

# Check those features that both have missing observations
df_NaN = df[df['bmi'].isnull() & df['smoking_status'].isnull()]
print(df_NaN.head(5))
print(df_NaN.shape)
del df_NaN


# Find out more about the catagorical features
print('\nfeature: gender')
print(df['gender'].value_counts())
print('\nfeature: hypertension')
print(df['hypertension'].value_counts())
print('\nfeature: heart_disease')
print(df['heart_disease'].value_counts())
print('\nfeature: ever_married')
print(df['ever_married'].value_counts())
print('\nfeature: work_type')
print(df['work_type'].value_counts())
print('\nfeature: Residence_type')
print(df['Residence_type'].value_counts())
print('\nfeature: smoking_status')
print(df['smoking_status'].value_counts())

# Find out the summary in the numerical features.
print('\nfeature: age')
print(df['age'].describe())
print('\nfeature: avg_glucose_level')
print(df['avg_glucose_level'].describe())
print('\nfeature: bmi')
print(df['bmi'].describe())

# Find out more about the target variable
print('\ntarget: stroke')
df['stroke'].value_counts(normalize=True)


# Set all the plots' style
sns.set_style("whitegrid")
# Plot the class distrbution of the stroke data

#sns.countplot(y_train)
#sns.countplot(y_r, palette=["g", "m"])
#plt.xticks(ticks=[0,1],labels=['No','Yes'])
#plt.xticks(fontsize=12)
#plt.yticks(fontsize=12)
#plt.xlabel('Suffering from stroke', fontsize=16)
#plt.ylabel('count', fontsize=16)
#plt.savefig('Histogram_train_SMOTE_stroke', dpi=300)
#plt.clf()

sns.countplot(x='stroke', data=df)
plt.xticks(ticks=[0,1],labels=['No','Yes'])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('Histogram of stroke', fontsize=16)
plt.xlabel('Suffering from stroke', fontsize=16)
plt.ylabel('count', fontsize=16)
plt.savefig('Histogram_stroke', dpi=300)
plt.clf()
plt.show()


# Plot the distribution of age
sns.distplot(df['age'])
plt.vlines(round(df['age'].mean(),2), 0, 0.02, colors='k', linestyles='dashed')
plt.vlines(10, 0, 0.02, colors='r', linestyles='dashed')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Histogram of patients' ages", fontsize=16)
plt.xlabel('Age', fontsize=16)
##plt.ylabel('count', fontsize=16)
#plt.savefig('Histogram_age', dpi=300)
#plt.clf()
plt.show()

# Plot the distribution of avg_glucose_level
sns.distplot(df['avg_glucose_level'])
plt.vlines(round(df['avg_glucose_level'].mode(),2), 0, 0.02, colors='k', linestyles='dashed')
plt.vlines(120, 0, 0.02, colors='r', linestyles='dashed')
plt.vlines(140, 0, 0.02, colors='r', linestyles='dashed')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Histogram of patients' average glucose level", fontsize=16)
plt.xlabel('Glucose level after a meal (mg/dL)', fontsize=16)
##plt.ylabel('count', fontsize=16)
#plt.savefig('Histogram_glucose', dpi=300)
#plt.clf()
plt.show()

# Plot the distribution of bmi
sns.distplot(df['bmi'].dropna(axis=0))
plt.vlines(round(df['bmi'].mean(),2), 0, 0.06, colors='k', linestyles='dashed')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Histogram of patients' BMI", fontsize=16)
plt.xlabel('bmi',fontsize=16)
##plt.ylabel('count',fontsize=16)
#plt.savefig('Histogram_bmi', dpi=300)
#plt.clf()
plt.show()
'''

# WE NEED a feature function pairplot!!! 
'''
# Plot the distribution of gender
sns.countplot(x='gender', data=df)
plt.title('Histogram of gender')
plt.xlabel('Gender')
plt.ylabel('count')
plt.show()

# Plot the distribution of hypertension
sns.countplot(x='hypertension', data=df)
plt.title('Histogram of hypertension')
plt.xlabel('Suffering from hypertension')
plt.ylabel('count')
plt.xticks(ticks=[0,1],labels=['No','Yes'])
plt.show()

# Plot the distribution of heart disease
sns.countplot(x='heart_disease', data=df)
plt.title('Histogram of heart disease')
plt.xlabel('Suffering from heart disease')
plt.ylabel('count')
plt.xticks(ticks=[0,1],labels=['No','Yes'])
plt.show()

# Plot the distribution of ever_married
sns.countplot(x='ever_married', data=df)
plt.title('Histogram of ever_married')
plt.xlabel('Have ever married')
plt.ylabel('count')
plt.show()

# Plot the distribution of work_type
sns.countplot(x='work_type', data=df)
plt.title('Histogram of occupation')
plt.xlabel('Type of occupation')
plt.ylabel('count')
plt.xticks(rotation=45)
plt.show()

# Plot the distribution of Residence_type
sns.countplot(x='Residence_type', data=df)
plt.title('Histogram of residence type')
plt.xlabel('Area of residence')
plt.ylabel('count')
plt.show()

# Plot the distribution of smoking_status
sns.countplot(x='smoking_status', data=df)
plt.title('Histogram of smoke status')
plt.xlabel('Smoke status')
plt.ylabel('count')
plt.show()

# ----------------------------------------------
# Prepare data for classifiers
# ----------------------------------------------
# 1) For Logitic Regression 
# Apply one-hot encoding (keep K-1 dummy bariables)  to nominal categorical features
df_edrop1 = df.copy()
df_edrop1 = pd.get_dummies(df_edrop1, prefix='gender', columns=['gender'], drop_first=True)
df_edrop1 = pd.get_dummies(df_edrop1, prefix='hypertension', columns=['hypertension'], drop_first=True)
df_edrop1 = pd.get_dummies(df_edrop1, prefix='heart_disease', columns=['heart_disease'], drop_first=True)
df_edrop1 = pd.get_dummies(df_edrop1, columns=['ever_married'], drop_first=True)
df_edrop1 = pd.get_dummies(df_edrop1, prefix='work_type', columns=['work_type'], drop_first=True)
df_edrop1 = pd.get_dummies(df_edrop1, prefix='Residence_type', columns=['Residence_type'], drop_first=True)
df_edrop1 = pd.get_dummies(df_edrop1, prefix='smoking_status', columns=['smoking_status'], drop_first=True)

# Handle the missing values in the bmi feature
#     Replace the missing values with median instead of mean to handle future skewed data
#     If the distribution isn't skewed, the bmi's mean and median should be close
df_edrop1['bmi'].fillna(df_edrop1['bmi'].median(), inplace=True)
sns.distplot(df_edrop1['bmi'])

# Handle the missing values in the smoke_status feature
#     Remove all the null entries
df_edrop1.dropna(axis=0, how='any', inplace=True)
print(df_edrop1.columns)
# Pickle the DataFrama 
#with open('stroke_encoded_dropone.pickle', 'wb') as write_to:
#        pickle.dump(df_edrop1, write_to)


# 2) For SVM
# Apply one-hot encoding (keep K-1 dummy bariables)  to nominal categorical features
df_e = df.copy()
df_e = pd.get_dummies(df_e , prefix='gender', columns=['gender'])
df_e = pd.get_dummies(df_e, prefix='hypertension', columns=['hypertension'])
df_e = pd.get_dummies(df_e, prefix='heart_disease', columns=['heart_disease'])
df_e = pd.get_dummies(df_e, columns=['ever_married'])
df_e = pd.get_dummies(df_e, prefix='work_type', columns=['work_type'])
df_e = pd.get_dummies(df_e, prefix='Residence_type', columns=['Residence_type'])
df_e = pd.get_dummies(df_e, prefix='smoking_status', columns=['smoking_status'])

# Handle the missing values in the bmi feature
#     Replace the missing values with median instead of mean to handle future skewed data
#     If the distribution isn't skewed, the bmi's mean and median should be close
df_e['bmi'].fillna(df_e['bmi'].median(), inplace=True)

# Handle the missing values in the smoke_status feature
#     Remove all the null entries
df_e.dropna(axis=0, how='any', inplace=True)
# Pickle the DataFrama 
#with open('stroke_encoded.pickle', 'wb') as write_to:
#        pickle.dump(df_e, write_to)


# 3) For Naive Base and tree-base models
df_o = df.copy()

# Handle the missing values in the bmi feature
#     Replace the missing values with median instead of mean to handle future skewed data
#     If the distribution isn't skewed, the bmi's mean and median should be close
df_o['bmi'].fillna(df_o['bmi'].median(), inplace=True)

# Handle the missing values in the smoke_status feature
#     Remove all the null entries
df_o.dropna(axis=0, how='any', inplace=True)
# Pickle the DataFrama 
#with open('stroke_original.pickle', 'wb') as write_to:
#        pickle.dump(df_o, write_to)



'''
# NEED Description
df.groupby('stroke').mean()
df.groupby('hypertension').mean()
df.groupby('heart_disease').mean()
df.groupby('gender').mean()
df.groupby('ever_married').mean()
df.groupby('work_type').mean()
df.groupby('Residence_type').mean()
df.groupby('smoking_status').mean()
'''


