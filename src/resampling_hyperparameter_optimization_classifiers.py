#!/usr/bin/env python3
"""
Name: resampling_hyperparameter_optimization_classifiers.py
Purpose: To improve the models' sensitivity in predicting the positive class
        that is a minority class in an imbalanced dataset. 
        Sampling methods, such as SMOTE, ADASYN, Random Over Sampler and Random 
        Under Sampler are implemented to sample the training data set. 
        Models' paramaters (hyperparameters) are tuned with 
        Randomized Search.
Tools: Pandas, scikit-learn, imbalanced-learn, and pickle
References:  
    https://stackoverflow.com/questions/48370150/how-to-implement-smote-in-cross-validation-and-gridsearchcv
    Incorporating Oversampling in an ML Pipeline https://bsolomon1124.github.io/oversamp/
    http://dev-aux.com/python/how-to-predict_proba-with-linearsvc    
"""

# Standard libraries
import pandas as pd
import numpy as np
import pickle

# Preprocessing module
from sklearn.preprocessing import StandardScaler

# sampling methods in imbalanced-learn api
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# The classification models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

# Load customized functions from the functions.py
from functions import get_cross_val_score_imblearn, plot_metric_curves
from functions import get_hyperparameter_opt_imblearn



if __name__ == "__main__":

    # Prepare training, validation, and test datasets for Logistic Regression
    #   The data is specifically for Logistic Regression. After applying
    #   one-hot encoding for every categorical feature, one dummy variable
    #   is dropped. 
    df_edrop1 = pd.read_pickle('stroke_encoded_dropone.pickle')
    X = df_edrop1.drop( columns=['stroke'])
    y = df_edrop1['stroke']
    X_train_val, X_test, y_train_val, y_test1 = \
        train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
        
    X_train, X_val, y_train, y_val = \
        train_test_split(X_train_val, y_train_val, test_size=0.25, \
                         stratify=y_train_val, random_state=43)
        
    # Apply Standard Scaler to training and validation sets
    std = StandardScaler()
    std.fit(X_train.values)    
    X_train_scaled = std.transform(X_train.values)
    #Apply the scaler to the val and test set
    X_val_scaled1 = std.transform(X_val.values)
    X_test_scaled1 = std.transform(X_test.values)    
    
    # Model 1: Logistic Regression 
    dict_scores = {}
    models = []
    data_val = []
    model_names = []
    mscores = ['f1_macro', 'f1_weighted']
    #mscores = ['f1_macro', 'f1_weighted', 'precision_macro','recall_macro']
    print("Logistic Regression ...")
    # Create the oversampling and undersampling methods' instances 
    sm = SMOTE(random_state=42)
    adas = ADASYN(random_state=42)
    rus = RandomUnderSampler(random_state=42)
    ros = RandomOverSampler(random_state=42)    
    # Create paramater search space
    param_grid = {'model__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
                  'model__penalty': ['l1','l2'],
                  'model__class_weight':['balanced', None]}
    # Create a model instance      
    model = LogisticRegression(class_weight='balanced',random_state=41)
    # Optimize the hyperparamaters in param_grid        
    rsCV_SMOTE = get_hyperparameter_opt_imblearn(model, sm, param_grid, mscores, X_train_val, y_train_val)
    rsCV_ADASYN = get_hyperparameter_opt_imblearn(model, adas, param_grid, mscores, X_train_val, y_train_val)
    rsCV_RUS = get_hyperparameter_opt_imblearn(model, rus, param_grid, mscores, X_train_val, y_train_val)
    rsCV_ROS = get_hyperparameter_opt_imblearn(model, ros, param_grid, mscores, X_train_val, y_train_val)
    print("LR: RS best f1 (SMOTE) = ",rsCV_SMOTE.best_score_) 
    print("LR: RS best f1 (ADASYN) = ",rsCV_ADASYN.best_score_) 
    print("LR: RS best f1 (RUS) = ",rsCV_RUS.best_score_) 
    print("LR: RS best f1 (ROS) = ",rsCV_ROS.best_score_)    
    #After optimization, there are the model with the best parameters.    
    lr_model_SMOTE = LogisticRegression(C=rsCV_SMOTE.best_params_['model__C'],
                    penalty=rsCV_SMOTE.best_params_['model__penalty'],
                    class_weight=rsCV_SMOTE.best_params_['model__class_weight'],
                    random_state=41)
    lr_model_ADASYN = LogisticRegression(C=rsCV_ADASYN.best_params_['model__C'],
                    penalty=rsCV_ADASYN.best_params_['model__penalty'],
                    class_weight=rsCV_ADASYN.best_params_['model__class_weight'],
                    random_state=41) 
    lr_model_RUS = LogisticRegression(C=rsCV_RUS.best_params_['model__C'],
                    penalty=rsCV_RUS.best_params_['model__penalty'],
                    class_weight=rsCV_RUS.best_params_['model__class_weight'],
                    random_state=41)  
    lr_model_ROS = LogisticRegression(C=rsCV_ROS.best_params_['model__C'],
                    penalty=rsCV_ROS.best_params_['model__penalty'],
                    class_weight=rsCV_ROS.best_params_['model__class_weight'],
                    random_state=41)              
    # Get metrics or score through cross validation  
    scores1 = get_cross_val_score_imblearn(lr_model_SMOTE, sm, X_train_val, y_train_val)
    scores2 = get_cross_val_score_imblearn(lr_model_ADASYN, adas, X_train_val, y_train_val)
    scores3 = get_cross_val_score_imblearn(lr_model_RUS, rus, X_train_val, y_train_val)
    scores4 = get_cross_val_score_imblearn(lr_model_ROS, ros, X_train_val, y_train_val)
    # Save the score in a dictionary   
    dict_scores['Logistic Regression SMOTE']= scores1
    dict_scores['Logistic Regression ADASYN']= scores2
    dict_scores['Logistic Regression RUS']= scores3
    dict_scores['Logistic Regression ROS']= scores4
    # First, apply sampling methods to randomly sample the scaled  train data
    #   Them, fit the model with the sampled training data      
    X_r, y_r = sm.fit_sample(X_train_scaled,y_train)
    lr_model_SMOTE.fit(X_r, y_r)
    models.append(lr_model_SMOTE)
    model_names.append('Logistic-SMOTE')
    
    X_r, y_r = adas.fit_sample(X_train_scaled,y_train)
    lr_model_ADASYN.fit(X_r, y_r)
    models.append(lr_model_ADASYN)
    model_names.append('Logistic-ADASYN')    
    
    X_r, y_r = rus.fit_sample(X_train_scaled,y_train)
    lr_model_RUS.fit(X_r, y_r)
    models.append(lr_model_RUS)
    model_names.append('Logistic-RUS')        
    
    X_r, y_r = ros.fit_sample(X_train_scaled,y_train)
    lr_model_ROS.fit(X_r, y_r)
    models.append(lr_model_ROS) 
    model_names.append('Logistic-ROS')         

    # If it is in the testing stage, use validation data         
#    data_val.append([X_val_scaled,y_val])
    # Otherwiese, use the test data set     
    data_val.append([X_test_scaled1,y_test1])    

    # Print the scores in dataframe
    df_score = pd.DataFrame.from_dict(dict_scores,orient='index', 
        columns=['F1_macro','Precision_macro','Recall_macro','F1_weight', 'Confusion Matrix']) 
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_score)
        
    #---------------------------------------------------
    # Pickle the scores and list of model objects
    #---------------------------------------------------     
    df_logistic_sampling_score = df_score
    with open('logistic_sampling_dfscore.pickle', 'wb') as write_to:
        pickle.dump(df_logistic_sampling_score, write_to)
        
    logistic_sampling_model = models
    with open('logistic_sampling_correct.pickle', 'wb') as write_to:
        pickle.dump(logistic_sampling_model, write_to)        
        
    # Plot the ROC and Precision-Recall curves
    plot_ind_metric_curves(models, data_val, model_names, plot_name='Logistic_sampling')
    
    #---------------------------------------------------
    # If you want to load the pickle files
    #---------------------------------------------------     
#    with open('logistic_sampling.pickle', 'rb') as read_to:
#        pickle.load(read_to)
#    models =  logistic_sampling_model
    

    # -------------------------------------------------------------------------
    # Prepare training, validation, and test datasets 
    #   The data is specifically for  other classifiers. The catogorical
    #   data are encoded with one-hot encoding.
    df_e = pd.read_pickle('stroke_encoded.pickle')
    X = df_e.drop( columns=['stroke'])
    y = df_e['stroke']
    X_train_val, X_test, y_train_val, y_test = \
        train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
        
    X_train, X_val, y_train, y_val = \
        train_test_split(X_train_val, y_train_val, test_size=0.25, \
                         stratify=y_train_val, random_state=43)
        
    # Apply Standard Scaler to training and validation sets
    std = StandardScaler()
    std.fit(X_train.values)    
    X_train_scaled = std.transform(X_train.values)
    #Apply the scaler to the val and test set
    X_val_scaled = std.transform(X_val.values)
    X_test_scaled = std.transform(X_test.values)    
    
    dict_scores = {}
    models = []
    data_val = []
    model_names = []
    mscores = ['f1_macro', 'f1_weighted']
    
    print("linearSVC ...")
    # Create the oversampling and undersampling methods' instances 
    sm = SMOTE(random_state=42)
    adas = ADASYN(random_state=42)
    rus = RandomUnderSampler(random_state=42)
    ros = RandomOverSampler(random_state=42)
    # Create paramater search space      
    param_grid = {'model__penalty': ['l1','l2'],
                  'model__loss':['squared_hinge'],
                  'model__C': np.logspace(-1, 3, 10),
                  'model__class_weight':['balanced', None]}
    # Create a model instance      
    model = LinearSVC(random_state=41)    
    # Optimize the hyperparamaters in param_grid        
    rsCV_SMOTE = get_hyperparameter_opt_imblearn(model, sm, param_grid, mscores, X_train_val, y_train_val)
    rsCV_ADASYN = get_hyperparameter_opt_imblearn(model, adas, param_grid, mscores, X_train_val, y_train_val)
    rsCV_RUS = get_hyperparameter_opt_imblearn(model, rus, param_grid, mscores, X_train_val, y_train_val)
    rsCV_ROS = get_hyperparameter_opt_imblearn(model, ros, param_grid, mscores, X_train_val, y_train_val)
    print("LinearSVC: RS best f1 (SMOTE) = ",rsCV_SMOTE.best_score_) 
    print("LinearSVC: RS best f1 (ADASYN) = ",rsCV_ADASYN.best_score_) 
    print("LinearSVC: RS best f1 (RUS) = ",rsCV_RUS.best_score_) 
    print("LinearSVC: RS best f1 (ROS) = ",rsCV_ROS.best_score_)    
    #After optimization, there are the model with the best parameters.        
    lsvc_model_SMOTE = LinearSVC(penalty=rsCV_SMOTE.best_params_['model__penalty'],
                           loss=rsCV_SMOTE.best_params_['model__loss'],
                           C=rsCV_SMOTE.best_params_['model__C'],
                           class_weight=rsCV_SMOTE.best_params_['model__class_weight'],                       
                           random_state=41)    
    
    lsvc_model_ADASYN = LinearSVC(penalty=rsCV_ADASYN.best_params_['model__penalty'],
                           loss=rsCV_ADASYN.best_params_['model__loss'],
                           C=rsCV_ADASYN.best_params_['model__C'],
                           class_weight=rsCV_ADASYN.best_params_['model__class_weight'],                       
                           random_state=41)    
    
    lsvc_model_RUS = LinearSVC(penalty=rsCV_RUS.best_params_['model__penalty'],
                           loss=rsCV_RUS.best_params_['model__loss'],
                           C=rsCV_RUS.best_params_['model__C'],
                           class_weight=rsCV_RUS.best_params_['model__class_weight'],                       
                           random_state=41)    

    lsvc_model_ROS = LinearSVC(penalty=rsCV_ROS.best_params_['model__penalty'],
                           loss=rsCV_ROS.best_params_['model__loss'],
                           C=rsCV_ROS.best_params_['model__C'],
                           class_weight=rsCV_ROS.best_params_['model__class_weight'],                       
                           random_state=41)
 
    # Get metrics or score through cross validation  
    scores1 = get_cross_val_score_imblearn(lsvc_model_SMOTE, sm, X_train_val, y_train_val)
    scores2 = get_cross_val_score_imblearn(lsvc_model_ADASYN, adas, X_train_val, y_train_val)
    scores3 = get_cross_val_score_imblearn(lsvc_model_RUS, rus, X_train_val, y_train_val)
    scores4 = get_cross_val_score_imblearn(lsvc_model_ROS, ros, X_train_val, y_train_val)
    # Save the score in a dictionary   
    dict_scores['LinearSVC SMOTE']= scores1
    dict_scores['LinearSVC ADASYN']= scores2
    dict_scores['LinearSVC RUS']= scores3
    dict_scores['LinearSVC ROS']= scores4 
    # First, apply sampling methods to randomly sample the scaled  train data
    #   Them, fit the model with the sampled training data      
    X_r, y_r = sm.fit_sample(X_train_scaled,y_train)
    lsvc_model_SMOTE.fit(X_r, y_r)
    models.append(lsvc_model_SMOTE)
    model_names.append('LinearSVC-SMOTE')
    
    X_r, y_r = adas.fit_sample(X_train_scaled,y_train)
    lsvc_model_ADASYN.fit(X_r, y_r)
    models.append(lsvc_model_ADASYN)
    model_names.append('LinearSVC-ADASYN')    
    
    X_r, y_r = rus.fit_sample(X_train_scaled,y_train)
    lsvc_model_RUS.fit(X_r, y_r)
    models.append(lsvc_model_RUS)
    model_names.append('LinearSVC-RUS')        
    
    X_r, y_r = ros.fit_sample(X_train_scaled,y_train)
    lsvc_model_ROS.fit(X_r, y_r)
    models.append(lsvc_model_ROS) 
    model_names.append('LinearSVC-ROS')         
    
    # If it is in the testing stage, use validation data         
#    data_val.append([X_val_scaled,y_val])
    # Otherwiese, use the test data set     
    data_val.append([X_test_scaled,y_test])
    # Print the scores in dataframe    
    df_score = pd.DataFrame.from_dict(dict_scores,orient='index', 
        columns=['F1_macro','Precision_macro','Recall_macro','F1_weight', 'Confusion Matrix'])
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_score)

    # Plot the ROC and Precision-Recall curves. plot_metric_curves is a customized function
    plot_ind_metric_curves(models, data_val, model_names, plot_name='LinearSVC_sampling')
        
    #---------------------------------------------------
    # Pickle the scores and list of model objects
    #---------------------------------------------------  
    df_linearSVC_sampling_score = df_score
    with open('linearSVC_sampling_dfscore.pickle', 'wb') as write_to:
        pickle.dump(df_linearSVC_sampling_score, write_to)
        
    linearSVC_sampling_model = models
    with open('linearSVC_sampling.pickle', 'wb') as write_to:
        pickle.dump(linearSVC_sampling_model, write_to)        
        
    #---------------------------------------------------
    # If you want to load the pickle files
    #---------------------------------------------------     
#    with open('linearSVC_sampling.pickle', 'rb') as read_to:
#        pickle.load(read_to)
#    models = linearSVC_sampling_model
#    

    
    
    # Model 3: Naive Bayes
    dict_scores = {}
    models = []
    data_val = []
    model_names = []
    mscores = ['f1_macro', 'f1_weighted']
    
    print("Naive Bayes ...")
    # Create the oversampling and undersampling methods' instances 
    sm = SMOTE(random_state=42)
    adas = ADASYN(random_state=42)
    rus = RandomUnderSampler(random_state=42)
    ros = RandomOverSampler(random_state=42)
    # Create a model instance      
    Gnb_model = GaussianNB()
    # Get metrics or score through cross validation     
    scores1 = get_cross_val_score_imblearn(Gnb_model, sm, X_train_val, y_train_val)
    scores2 = get_cross_val_score_imblearn(Gnb_model, adas, X_train_val, y_train_val)
    scores3 = get_cross_val_score_imblearn(Gnb_model, rus, X_train_val, y_train_val)
    scores4 = get_cross_val_score_imblearn(Gnb_model, ros, X_train_val, y_train_val)
    # Save the score in a dictionary       
    dict_scores['Naive Bayes SMOTE']= scores1
    dict_scores['Naive Bayes ADASYN']= scores2
    dict_scores['Naive Bayes RUS']= scores3
    dict_scores['Naive Bayes ROS']= scores4
    
    # First, apply sampling methods to randomly sample the scaled  train data
    #   Them, fit the model with the sampled training data  
    Gnb_model_SMOTE = GaussianNB(priors=None, var_smoothing=1e-09)
    X_r, y_r = sm.fit_sample(X_train_scaled,y_train)
    Gnb_model_SMOTE.fit(X_r, y_r)
    models.append(Gnb_model_SMOTE)
    model_names.append('Naive Bayes-SMOTE')
    
    Gnb_model_ADASYN = GaussianNB(priors=None, var_smoothing=1e-09)
    X_r, y_r = adas.fit_sample(X_train_scaled,y_train)
    Gnb_model_ADASYN.fit(X_r, y_r)
    models.append(Gnb_model_ADASYN)
    model_names.append('Naive Bayes-ADASYN')    
    
    Gnb_model_RUS = GaussianNB(priors=None, var_smoothing=1e-09)
    X_r, y_r = rus.fit_sample(X_train_scaled,y_train)
    Gnb_model_RUS.fit(X_r, y_r)
    models.append(Gnb_model_RUS)
    model_names.append('Naive Bayes-RUS')   
     
    Gnb_model_ROS = GaussianNB(priors=None, var_smoothing=1e-09)
    X_r, y_r = ros.fit_sample(X_train_scaled,y_train)
    Gnb_model_ROS.fit(X_r, y_r)
    models.append(Gnb_model_ROS) 
    model_names.append('Naive Bayes-ROS')         
    
    # If it is in the testing stage, use validation data         
#    data_val.append([X_val_scaled,y_val])
    # Otherwiese, use the test data set     
    data_val.append([X_test_scaled,y_test])
    # Print the scores in dataframe    
    df_score = pd.DataFrame.from_dict(dict_scores,orient='index', 
        columns=['F1_macro','Precision_macro','Recall_macro','F1_weight', 'Confusion Matrix'])
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_score)
   
    # Plot the ROC and Precision-Recall curves. plot_metric_curves is a customized function
    plot_ind_metric_curves(models, data_val, model_names, plot_name='NaiveBayes_sampling')
         
    #---------------------------------------------------
    # Pickle the scores and list of model objects
    #---------------------------------------------------  
    df_NaiveB_sampling_score = df_score
    with open('NaiveB_sampling_dfscore.pickle', 'wb') as write_to:
        pickle.dump(df_NaiveB_sampling_score, write_to)
     
    NaiveB_sampling_model = models
    with open('NaiveB_sampling.pickle', 'wb') as write_to:
        pickle.dump(NaiveB_sampling_model, write_to)        
        
    #---------------------------------------------------
    # If you want to load the pickle files
    #---------------------------------------------------     
#    with open('NaiveB_sampling.pickle', 'rb') as read_to:
#        pickle.load(read_to)
#    models = NaiveB_sampling_model
#    


    # Model 4: Decision Tree Classifier
    dict_scores = {}
    models = []
    data_val = []
    model_names = []
    mscores = ['f1_macro', 'f1_weighted']
    
    print("Decision Tree ...")
    # Create the oversampling and undersampling methods' instances 
    sm = SMOTE(random_state=42)
    adas = ADASYN(random_state=42)
    rus = RandomUnderSampler(random_state=42)
    ros = RandomOverSampler(random_state=42)    
    # Create paramater search space  
    param_grid = {'model__splitter': ['best', 'random'],
                  'model__min_samples_split' : [2, 4, 6, 10, 15, 25],
                  'model__max_depth': np.linspace(1,21,10),
                  'model__max_features': ['auto','log2',None],
                  'model__criterion': ['gini', 'entropy'],
                  'model__class_weight':['balanced', None]}
    # Create a model instance      
    model = DecisionTreeClassifier(random_state=41)      
    # Optimize the hyperparamaters in param_grid    
    rsCV_SMOTE = get_hyperparameter_opt_imblearn(model, sm, param_grid, mscores, X_train_val, y_train_val)
    rsCV_ADASYN = get_hyperparameter_opt_imblearn(model, adas, param_grid, mscores, X_train_val, y_train_val)
    rsCV_RUS = get_hyperparameter_opt_imblearn(model, rus, param_grid, mscores, X_train_val, y_train_val)
    rsCV_ROS = get_hyperparameter_opt_imblearn(model, ros, param_grid, mscores, X_train_val, y_train_val)
    print("DT: RS best f1 (SMOTE) = ",rsCV_SMOTE.best_score_) 
    print("DT: RS best f1 (ADASYN) = ",rsCV_ADASYN.best_score_) 
    print("DT: RS best f1 (RUS) = ",rsCV_RUS.best_score_) 
    print("DT: RS best f1 (ROS) = ",rsCV_ROS.best_score_) 
    #After optimization, there are the model with the best parameters.    
    dtree_model_SMOTE = DecisionTreeClassifier(
                   max_features=rsCV_SMOTE.best_params_['model__max_features'],
                   criterion=rsCV_SMOTE.best_params_['model__criterion'],
                   splitter=rsCV_SMOTE.best_params_['model__splitter'],
                   max_depth=rsCV_SMOTE.best_params_['model__max_depth'],
                   min_samples_split=rsCV_SMOTE.best_params_['model__min_samples_split'],
                   class_weight=rsCV_SMOTE.best_params_['model__class_weight'],                  
                   random_state=41)    
    

    dtree_model_ADASYN = DecisionTreeClassifier(
                   max_features=rsCV_ADASYN.best_params_['model__max_features'],
                   criterion=rsCV_ADASYN.best_params_['model__criterion'],
                   splitter=rsCV_ADASYN.best_params_['model__splitter'],
                   max_depth=rsCV_ADASYN.best_params_['model__max_depth'],
                   min_samples_split=rsCV_ADASYN.best_params_['model__min_samples_split'],
                   class_weight=rsCV_ADASYN.best_params_['model__class_weight'],                  
                   random_state=41) 
    
    dtree_model_RUS = DecisionTreeClassifier(
                   max_features=rsCV_RUS.best_params_['model__max_features'],
                   criterion=rsCV_RUS.best_params_['model__criterion'],
                   splitter=rsCV_RUS.best_params_['model__splitter'],
                   max_depth=rsCV_RUS.best_params_['model__max_depth'],
                   min_samples_split=rsCV_RUS.best_params_['model__min_samples_split'],
                   class_weight=rsCV_RUS.best_params_['model__class_weight'],                  
                   random_state=41)  
    
    dtree_model_ROS = DecisionTreeClassifier(
                   max_features=rsCV_ROS.best_params_['model__max_features'],
                   criterion=rsCV_ROS.best_params_['model__criterion'],
                   splitter=rsCV_ROS.best_params_['model__splitter'],
                   max_depth=rsCV_ROS.best_params_['model__max_depth'],
                   min_samples_split=rsCV_ROS.best_params_['model__min_samples_split'],
                   class_weight=rsCV_ROS.best_params_['model__class_weight'],                  
                   random_state=41)             
    # Get metrics or score through cross validation  
    scores1 = get_cross_val_score_imblearn(dtree_model_SMOTE, sm, X_train_val, y_train_val)
    scores2 = get_cross_val_score_imblearn(dtree_model_ADASYN, adas, X_train_val, y_train_val)
    scores3 = get_cross_val_score_imblearn(dtree_model_RUS, rus, X_train_val, y_train_val)
    scores4 = get_cross_val_score_imblearn(dtree_model_ROS, ros, X_train_val, y_train_val)
    # Save the score in a dictionary   
    dict_scores['Decision Tree SMOTE']= scores1
    dict_scores['Decision Tree ADASYN']= scores2
    dict_scores['Decision Tree RUS']= scores3
    dict_scores['Decision Tree ROS']= scores4
    # First, apply sampling methods to randomly sample the scaled  train data
    #   Them, fit the model with the sampled training data      
    X_r, y_r = sm.fit_sample(X_train_scaled,y_train)
    dtree_model_SMOTE.fit(X_r, y_r)
    models.append(dtree_model_SMOTE)
    model_names.append('Decision Tree-SMOTE')
    
    X_r, y_r = adas.fit_sample(X_train_scaled,y_train)
    dtree_model_ADASYN.fit(X_r, y_r)
    models.append(dtree_model_ADASYN)
    model_names.append('Decision Tree-ADASYN')    
    
    X_r, y_r = rus.fit_sample(X_train_scaled,y_train)
    dtree_model_RUS.fit(X_r, y_r)
    models.append(dtree_model_RUS)
    model_names.append('Decision Tree-RUS')        
    
    X_r, y_r = ros.fit_sample(X_train_scaled,y_train)
    dtree_model_ROS.fit(X_r, y_r)
    models.append(dtree_model_ROS) 
    model_names.append('Decision Tree-ROS')         
    
    # If it is in the testing stage, use validation data         
#    data_val.append([X_val_scaled,y_val])
    # Otherwiese, use the test data set     
    data_val.append([X_test_scaled,y_test])
    # Print the scores in dataframe    
    df_score = pd.DataFrame.from_dict(dict_scores,orient='index', 
        columns=['F1_macro','Precision_macro','Recall_macro','F1_weight', 'Confusion Matrix'])
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_score)

    # Plot the ROC and Precision-Recall curves. plot_metric_curves is a customized function
    plot_ind_metric_curves(models, data_val, model_names, plot_name='DecisionTree_sampling')   
     
    #---------------------------------------------------
    # Pickle the scores and list of model objects
    #---------------------------------------------------  
    df_decisiontree_sampling_score = df_score
    with open('decisiontree_sampling_dfscore.pickle', 'wb') as write_to:
        pickle.dump(df_decisiontree_sampling_score, write_to)
        
    decisiontree_sampling_model = models
    with open('decisiontree_sampling.pickle', 'wb') as write_to:
        pickle.dump(decisiontree_sampling_model, write_to)        
        
    #---------------------------------------------------
    # If you want to load the pickle files
    #---------------------------------------------------     
#    with open('decisiontree_sampling.pickle', 'rb') as read_to:
#        pickle.load(read_to)
#    models = decisiontree_sampling_model
#    


    
    # Model 5: Bagging Classifier with decision tree
    dict_scores = {}
    models = []
    data_val = []
    model_names = []
    mscores = ['f1_macro', 'f1_weighted']
        
    print("Bagging Classifier with decision tree ...")
    # Create the oversampling and undersampling methods' instances 
    sm = SMOTE(random_state=42)
    adas = ADASYN(random_state=42)
    rus = RandomUnderSampler(random_state=42)
    ros = RandomOverSampler(random_state=42)    
    # Create paramater search space  
    param_grid = {'model__n_estimators': np.linspace(10,50,5,dtype=np.int),
                  'model__max_features' : [2, 4, 6, 8, 10, 12]}   
    # Create a model instance  
    model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), random_state=41)
    # Optimize the hyperparamaters in param_grid    
    rsCV_SMOTE = get_hyperparameter_opt_imblearn(model, sm, param_grid, mscores, X_train_val, y_train_val)
    rsCV_ADASYN = get_hyperparameter_opt_imblearn(model, adas, param_grid, mscores, X_train_val, y_train_val)
    rsCV_RUS = get_hyperparameter_opt_imblearn(model, rus, param_grid, mscores, X_train_val, y_train_val)
    rsCV_ROS = get_hyperparameter_opt_imblearn(model, ros, param_grid, mscores, X_train_val, y_train_val)
    print("Bagging DT: RS best f1 (SMOTE) = ",rsCV_SMOTE.best_score_) 
    print("Bagging DT: RS best f1 (ADASYN) = ",rsCV_ADASYN.best_score_) 
    print("Bagging DT: RS best f1 (RUS) = ",rsCV_RUS.best_score_) 
    print("Bagging DT: RS best f1 (ROS) = ",rsCV_ROS.best_score_) 
    #After optimization, there are the model with the best parameters. 
    bag_dtree_model_SMOTE = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                        n_estimators=rsCV_SMOTE.best_params_['model__n_estimators'],
                        max_features=rsCV_SMOTE.best_params_['model__max_features'],
                        random_state=41)
    
    bag_dtree_model_ADASYN = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                        n_estimators=rsCV_ADASYN.best_params_['model__n_estimators'],
                        max_features=rsCV_ADASYN.best_params_['model__max_features'],
                        random_state=41)
    
    bag_dtree_model_RUS = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                        n_estimators=rsCV_RUS.best_params_['model__n_estimators'],
                        max_features=rsCV_RUS.best_params_['model__max_features'],
                        random_state=41)
    
    bag_dtree_model_ROS = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                        n_estimators=rsCV_ROS.best_params_['model__n_estimators'],
                        max_features=rsCV_ROS.best_params_['model__max_features'],
                        random_state=41)        

    # Get metrics or score through cross validation  
    scores1 = get_cross_val_score_imblearn(bag_dtree_model_SMOTE, sm, X_train_val, y_train_val)
    scores2 = get_cross_val_score_imblearn(bag_dtree_model_ADASYN, adas, X_train_val, y_train_val)
    scores3 = get_cross_val_score_imblearn(bag_dtree_model_RUS, rus, X_train_val, y_train_val)
    scores4 = get_cross_val_score_imblearn(bag_dtree_model_ROS, ros, X_train_val, y_train_val)
    # Save the score in a dictionary   
    dict_scores['Bagging DT SMOTE']= scores1
    dict_scores['Bagging DT ADASYN']= scores2
    dict_scores['Bagging DT RUS']= scores3
    dict_scores['Bagging DT ROS']= scores4  
    # First, apply sampling methods to randomly sample the scaled  train data
    #   Them, fit the model with the sampled training data  
    X_r, y_r = sm.fit_sample(X_train_scaled,y_train)
    bag_dtree_model_SMOTE.fit(X_r, y_r)
    models.append(bag_dtree_model_SMOTE)
    model_names.append('Bagging DT-SMOTE')
    
    X_r, y_r = adas.fit_sample(X_train_scaled,y_train)
    bag_dtree_model_ADASYN.fit(X_r, y_r)
    models.append(bag_dtree_model_ADASYN)
    model_names.append('Bagging DT-ADASYN')    
    
    X_r, y_r = rus.fit_sample(X_train_scaled,y_train)
    bag_dtree_model_RUS.fit(X_r, y_r)
    models.append(bag_dtree_model_RUS)
    model_names.append('Bagging DT-RUS')        
    
    X_r, y_r = ros.fit_sample(X_train_scaled,y_train)
    bag_dtree_model_ROS.fit(X_r, y_r)
    models.append(bag_dtree_model_ROS) 
    model_names.append('Bagging DT-ROS')         
    
    # If it is in the testing stage, use validation data         
#    data_val.append([X_val_scaled,y_val])
    # Otherwiese, use the test data set     
    data_val.append([X_test_scaled,y_test])
    # Print the scores in dataframe    
    df_score = pd.DataFrame.from_dict(dict_scores,orient='index', 
        columns=['F1_macro','Precision_macro','Recall_macro','F1_weight', 'Confusion Matrix'])
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_score)

    # Plot the ROC and Precision-Recall curves. plot_metric_curves is a customized function
    plot_ind_metric_curves(models, data_val, model_names, plot_name='BaggingDT_sampling')
        
    #---------------------------------------------------
    # Pickle the scores and list of model objects
    #---------------------------------------------------  
    df_baggingDT_sampling_score = df_score
    with open('baggingDT_sampling_dfscore.pickle', 'wb') as write_to:
        pickle.dump(df_baggingDT_sampling_score, write_to)
        
    baggingDT_sampling_model = models
    with open('baggingDT_sampling.pickle', 'wb') as write_to:
        pickle.dump(baggingDT_sampling_model, write_to)        

    #---------------------------------------------------
    # If you want to load the pickle files
    #---------------------------------------------------         
#    with open('baggingDT_sampling.pickle', 'rb') as read_to:
#        pickle.load(read_to)
#    models = baggingDT_sampling_model
#    



       
    # Model 6: Random Forest Classifier
    dict_scores = {}
    models = []
    data_val = []
    model_names = []
    mscores = ['f1_macro', 'f1_weighted']
            
    print("Random Forest ...")
    # Create the oversampling and undersampling methods' instances 
    sm = SMOTE(random_state=42)
    adas = ADASYN(random_state=42)
    rus = RandomUnderSampler(random_state=42)
    ros = RandomOverSampler(random_state=42) 
    # Create paramater search space      
    param_grid = {'model__n_estimators': np.linspace(10,50,5,dtype=np.int),
                  'model__max_depth': np.linspace(1,21,10),
                  'model__max_features': [2, 4, 6, 8, 10, 12], #['auto','log2', None],
                  'model__min_samples_split' : [2, 4, 6, 10, 15, 25],
                  'model__criterion': ['gini', 'entropy'],
                  'model__class_weight':['balanced', None]}
    # Create a model instance      
    model = RandomForestClassifier(random_state=41)
    # Optimize the hyperparamaters in param_grid        
    rsCV_SMOTE = get_hyperparameter_opt_imblearn(model, sm, param_grid, mscores, X_train_val, y_train_val)
    rsCV_ADASYN = get_hyperparameter_opt_imblearn(model, adas, param_grid, mscores, X_train_val, y_train_val)
    rsCV_RUS = get_hyperparameter_opt_imblearn(model, rus, param_grid, mscores, X_train_val, y_train_val)
    rsCV_ROS = get_hyperparameter_opt_imblearn(model, ros, param_grid, mscores, X_train_val, y_train_val)
    print("Random Forest: RS best f1 (SMOTE) = ",rsCV_SMOTE.best_score_) 
    print("Random Forest: RS best f1 (ADASYN) = ",rsCV_ADASYN.best_score_) 
    print("Random Forest: RS best f1 (RUS) = ",rsCV_RUS.best_score_) 
    print("Random Forest: RS best f1 (ROS) = ",rsCV_ROS.best_score_)    
    #After optimization, there are the model with the best parameters.    
    rtree_model_SMOTE = RandomForestClassifier(n_estimators=rsCV_SMOTE.best_params_['model__n_estimators'],
                   max_features=rsCV_SMOTE.best_params_['model__max_features'],
                   criterion=rsCV_SMOTE.best_params_['model__criterion'],
                   max_depth=rsCV_SMOTE.best_params_['model__max_depth'],
                   min_samples_split=rsCV_SMOTE.best_params_['model__min_samples_split'],
                   class_weight=rsCV_SMOTE.best_params_['model__class_weight'],                  
                   random_state=41) 
    
    rtree_model_ADASYN = RandomForestClassifier(n_estimators=rsCV_ADASYN.best_params_['model__n_estimators'],
                   max_features=rsCV_ADASYN.best_params_['model__max_features'],
                   criterion=rsCV_ADASYN.best_params_['model__criterion'],
                   max_depth=rsCV_ADASYN.best_params_['model__max_depth'],
                   min_samples_split=rsCV_ADASYN.best_params_['model__min_samples_split'],
                   class_weight=rsCV_ADASYN.best_params_['model__class_weight'],                  
                   random_state=41)  
    
    rtree_model_RUS = RandomForestClassifier(n_estimators=rsCV_RUS.best_params_['model__n_estimators'],
                   max_features=rsCV_RUS.best_params_['model__max_features'],
                   criterion=rsCV_RUS.best_params_['model__criterion'],
                   max_depth=rsCV_RUS.best_params_['model__max_depth'],
                   min_samples_split=rsCV_RUS.best_params_['model__min_samples_split'],
                   class_weight=rsCV_RUS.best_params_['model__class_weight'],                  
                   random_state=41)  
    
    rtree_model_ROS = RandomForestClassifier(n_estimators=rsCV_ROS.best_params_['model__n_estimators'],
                   max_features=rsCV_ROS.best_params_['model__max_features'],
                   criterion=rsCV_ROS.best_params_['model__criterion'],
                   max_depth=rsCV_ROS.best_params_['model__max_depth'],
                   min_samples_split=rsCV_ROS.best_params_['model__min_samples_split'],
                   class_weight=rsCV_ROS.best_params_['model__class_weight'],                  
                   random_state=41)
    # Get metrics or score through cross validation  
    scores1 = get_cross_val_score_imblearn(rtree_model_SMOTE, sm, X_train_val, y_train_val)
    scores2 = get_cross_val_score_imblearn(rtree_model_ADASYN, adas, X_train_val, y_train_val)
    scores3 = get_cross_val_score_imblearn(rtree_model_RUS, rus, X_train_val, y_train_val)
    scores4 = get_cross_val_score_imblearn(rtree_model_ROS, ros, X_train_val, y_train_val)
    # Save the score in a dictionary  
    dict_scores['Random Forest SMOTE']= scores1
    dict_scores['Random Forest ADASYN']= scores2
    dict_scores['Random Forest RUS']= scores3
    dict_scores['Random Forest ROS']= scores4      
    # First, apply sampling methods to randomly sample the scaled  train data
    #   Them, fit the model with the sampled training data      
    X_r, y_r = sm.fit_sample(X_train_scaled,y_train)
    rtree_model_SMOTE.fit(X_r, y_r)
    models.append(rtree_model_SMOTE)
    model_names.append('Random Forest-SMOTE')
    
    X_r, y_r = adas.fit_sample(X_train_scaled,y_train)
    rtree_model_ADASYN.fit(X_r, y_r)
    models.append(rtree_model_ADASYN)
    model_names.append('Random Forest-ADASYN')    
    
    X_r, y_r = rus.fit_sample(X_train_scaled,y_train)
    rtree_model_RUS.fit(X_r, y_r)
    models.append(rtree_model_RUS)
    model_names.append('Random Forest-RUS')        
    
    X_r, y_r = ros.fit_sample(X_train_scaled,y_train)
    rtree_model_ROS.fit(X_r, y_r)
    models.append(rtree_model_ROS) 
    model_names.append('Random Forest-ROS')         
    # If it is in the testing stage, use validation data         
#    data_val.append([X_val_scaled,y_val])
    # Otherwiese, use the test data set     
    data_val.append([X_test_scaled,y_test])
   
    df_score = pd.DataFrame.from_dict(dict_scores,orient='index', 
        columns=['F1_macro','Precision_macro','Recall_macro','F1_weight', 'Confusion Matrix'])
    #df_score = pd.DataFrame.from_dict(dict_scores,orient='index', 
    #    columns=['F1 score','Precision','Recall','Hinge Loss','Confusion Matrix'])
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_score)

    # Plot the ROC and Precision-Recall curves. plot_metric_curves is a customized function
    plot_ind_metric_curves(models, data_val, model_names, plot_name='RandomForest_sampling')  
      
    #---------------------------------------------------
    # Pickle the scores and list of model objects
    #---------------------------------------------------  
    df_randomforest_sampling_score = df_score
    with open('randomforest_sampling_dfscore.pickle', 'wb') as write_to:
        pickle.dump(df_randomforest_sampling_score, write_to)
        
    randomforest_sampling_model = models
    with open('randomforest_sampling.pickle', 'wb') as write_to:
        pickle.dump(randomforest_sampling_model, write_to)        
        
    #---------------------------------------------------
    # If you want to load the pickle files
    #---------------------------------------------------     
#    with open('randomforest_sampling.pickle', 'rb') as read_to:
#        pickle.load(read_to)
#    models = randomforest_sampling_model
#    

    