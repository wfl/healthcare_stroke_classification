#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 13:49:26 2019

@author: leong
"""


# Standard libraries
import pandas as pd
import pickle
import numpy as np

# Preprocessing module
from sklearn.preprocessing import StandardScaler

# Pipeline modules
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as Pipeline_imb

# sampling methods in imbalanced-learn api
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler

# The classification models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import Nystroem
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# Load customized functions from the functions.py
from functions import prepare_data_for_ML_models, get_best_hyperparameters
from functions import get_cross_val_score



def base_model(model, X_train, y_train, is_svm=False):
    """
    This function calculates the performance scores on the validation sets for 
    the model during cross-validation and generate a pipeline object for future use. 
        
    This function is generate  classifiers with their default hyperparameters 
    
    Input arguments
    model: classifer/model object
    X_train, y_train: training datasets (feature and target respectively) 
    is_svm: If it is True, the classifier is support vector classifier
    """
    
    if is_svm: # Is it is Support Vector Classifier
        feature_map_nystroem = Nystroem(gamma=0.2, n_components=300, random_state=25)
        pipeline_steps = [('scale', StandardScaler()),('featuremap', feature_map_nystroem),
                      ('model',model)] 
    else:
        pipeline_steps = [('scale', StandardScaler()),('model',model)] 
       
    scores = get_cross_val_score(pipeline_steps, X_train, y_train)
    
    pipe_model = Pipeline(steps=pipeline_steps)
    #pipe_model.fit(X_train, y_train)    
    
    return pipe_model, scores


def class_weight_model(model_name, X_train, y_train):
    """
    This function tunes the hyperparemeters of a classifier using RandomizedSearch, 
    calculates the performance scores on the validation sets for the model with 
    the best hyperparamaters during cross-validation, and generate a pipeline
    object for future use. 
    
    Input arguments
    model_name: Specify classifier's name
    X_train, y_train: training datasets (feature and target respectively)     
    """
    
    mscores = 'recall_macro' #'f1_macro'    
    if model_name == 'Logistic':
        lr_model = LogisticRegression(random_state=41)
        # Create paramater search space
        param_grid = {'model__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
                  'model__penalty': ['l1','l2'],
                  'model__class_weight':['balanced']} 
        pipeline_steps = [('scale', StandardScaler()),('model',lr_model)] 
        rsCV_weighted = get_best_hyperparameters(pipeline_steps, param_grid, mscores, X_train, y_train)   
        
        model = LogisticRegression(C=rsCV_weighted.best_params_['model__C'],
                penalty=rsCV_weighted.best_params_['model__penalty'],
                class_weight='balanced',random_state=41) 
        pipeline_steps = [('scale', StandardScaler()),('model',model)]
        # Calulate the performance scores on the validation set with 
        #   crossvalidation 
        scores = get_cross_val_score(pipeline_steps, X_train, y_train)
        
        pipe_model = Pipeline(steps=pipeline_steps)
        #pipe_model.fit(X_train, y_train)
         
        
    elif model_name == 'SVM':
        lsvc_model = LinearSVC(random_state=41)
        # Create paramater search space    
        param_grid = {'featuremap__gamma': [0.001, 0.01, 0.1, 0.2, 1],
                  'featuremap__n_components': [100, 200, 300, 400],
                  'model__penalty': ['l2'],
                  'model__loss':['hinge','squared_hinge'],
                  'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
                  'model__class_weight':['balanced']} 
        feature_map_nystroem = Nystroem(random_state=25)
        pipeline_steps = [('scale', StandardScaler()),('featuremap', feature_map_nystroem),
                      ('model',lsvc_model)]    
        rsCV_weighted = get_best_hyperparameters(pipeline_steps, param_grid, mscores, X_train, y_train)   
        
        model = LinearSVC(penalty=rsCV_weighted.best_params_['model__penalty'],
                           loss=rsCV_weighted.best_params_['model__loss'],
                           C=rsCV_weighted.best_params_['model__C'],
                           class_weight='balanced',                       
                           random_state=41)  
        feature_map_nystroem = Nystroem(gamma=rsCV_weighted.best_params_['featuremap__gamma'], 
                                        n_components=rsCV_weighted.best_params_['featuremap__n_components'], 
                                        random_state=25)
        pipeline_steps = [('scale', StandardScaler()),('featuremap', feature_map_nystroem),
                      ('model',model)]         
        scores = get_cross_val_score(pipeline_steps, X_train, y_train)

        pipe_model = Pipeline(steps=pipeline_steps)
        #pipe_model.fit(X_train, y_train)
        
        
    elif model_name == 'RF':    #Random Forest
        rtree_model = RandomForestClassifier(random_state=41)
        # Create paramater search space    
        param_grid = {'model__n_estimators': np.linspace(10,50,5,dtype=np.int),
                  'model__max_depth': [2, 3, 4, 5],
                  'model__max_features': ['auto','log2', None],
                  'model__min_samples_split' : [2, 4, 6, 10],
                  'model__criterion': ['gini', 'entropy'],
                  'model__class_weight':['balanced']}
        pipeline_steps = [('scale', StandardScaler()),('model',rtree_model)] 
        rsCV_weighted = get_best_hyperparameters(pipeline_steps, param_grid, mscores, X_train, y_train)  
        
        model = RandomForestClassifier(n_estimators=rsCV_weighted.best_params_['model__n_estimators'],
                   max_features=rsCV_weighted.best_params_['model__max_features'],
                   criterion=rsCV_weighted.best_params_['model__criterion'],
                   max_depth=rsCV_weighted.best_params_['model__max_depth'],
                   min_samples_split=rsCV_weighted.best_params_['model__min_samples_split'],
                   class_weight='balanced',                  
                   random_state=41)         
        pipeline_steps = [('scale', StandardScaler()),('model',model)]
        scores = get_cross_val_score(pipeline_steps, X_train, y_train)

        pipe_model = Pipeline(steps=pipeline_steps)
        #pipe_model.fit(X_train, y_train)
        
    elif model_name == 'GBC':   #Gradient Boosting
        gbc_model = GradientBoostingClassifier(random_state=41)
        # Create paramater search space    
        param_grid = {'model__n_estimators': np.linspace(10,50,5,dtype=np.int),
                  'model__learning_rate': [0.001, 0.01, 0.1, 1, 10],
                  'model__subsample': [0.5, 1], 
                  'model__max_depth': [3, 4, 5, 6],
                  'model__max_features': ['auto','log2', None],
                  'model__min_samples_split' : [2, 4, 6, 10],
                  'model__min_samples_leaf': [1, 2]}
        pipeline_steps = [('scale', StandardScaler()),('model',gbc_model)] 
        rsCV_weighted = get_best_hyperparameters(pipeline_steps, param_grid, mscores, X_train, y_train)  
        
        model = GradientBoostingClassifier(n_estimators=rsCV_weighted.best_params_['model__n_estimators'],
                   learning_rate=rsCV_weighted.best_params_['model__learning_rate'],
                   subsample=rsCV_weighted.best_params_['model__subsample'],
                   max_features=rsCV_weighted.best_params_['model__max_features'],
                   min_samples_leaf=rsCV_weighted.best_params_['model__min_samples_leaf'],
                   max_depth=rsCV_weighted.best_params_['model__max_depth'],
                   min_samples_split=rsCV_weighted.best_params_['model__min_samples_split'],                  
                   random_state=41)         
        pipeline_steps = [('scale', StandardScaler()),('model',model)]
        scores = get_cross_val_score(pipeline_steps, X_train, y_train)

        pipe_model = Pipeline(steps=pipeline_steps)
        #pipe_model.fit(X_train, y_train)
        
    return pipe_model, rsCV_weighted, scores


def resampling_model(model_name, sampling, X_train, y_train):
    """
    This function apply resampling strategy to restore class balance in the 
    training data, tunes the hyperparemeters of a classifier using RandomizedSearch, 
    calculates the performance scores on the validation sets for the model with 
    the best hyperparamaters during cross-validation, and generate a pipeline
    object for future use. 
        
    Input arguments
    model_name: Specify classifier's name
    sampling: resampling object
    X_train, y_train: training datasets (feature and target respectively)     
    """
    # Create the oversampling and undersampling methods' instances
    mscores = 'recall_macro' #'f1_macro'     

    if model_name == 'Logistic':
        lr_model = LogisticRegression(random_state=41)
        # Create paramater search space
        param_grid = {'model__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
                  'model__penalty': ['l1','l2']}
        pipeline_steps = [('sampling_method', sampling),
                          ('scale', StandardScaler()),('model',lr_model)]
        rsCV_resample = get_best_hyperparameters(pipeline_steps, param_grid, mscores, X_train, y_train, imb_class=True)
        
        model = LogisticRegression(C=rsCV_resample.best_params_['model__C'],
                    penalty=rsCV_resample.best_params_['model__penalty'],
                    random_state=41) 
        pipeline_steps = [('sampling_method', sampling),
                          ('scale', StandardScaler()),('model',model)]
        scores = get_cross_val_score(pipeline_steps, X_train, y_train, imb_class=True)  

        pipe_model = Pipeline_imb(steps=pipeline_steps)
        #pipe_model.fit(X_train, y_train)
                
        
    elif model_name == 'SVM':
        lsvc_model = LinearSVC(random_state=41)
        # Create paramater search space    
        param_grid = {'featuremap__gamma': [0.001, 0.01, 0.1, 0.2, 1],
                  'featuremap__n_components': [100, 200, 300, 400],
                  'model__penalty': ['l2'],
                  'model__loss':['hinge','squared_hinge'],
                  'model__C': [0.001, 0.01, 0.1, 1, 10, 100]
                  } 
        feature_map_nystroem = Nystroem(random_state=25)
        pipeline_steps = [('sampling_method', sampling), ('scale', StandardScaler()),
                          ('featuremap', feature_map_nystroem),('model',lsvc_model)]    
        rsCV_resample = get_best_hyperparameters(pipeline_steps, param_grid, mscores, X_train, y_train, imb_class=True)

        model = LinearSVC(penalty=rsCV_resample.best_params_['model__penalty'],
                           loss=rsCV_resample.best_params_['model__loss'],
                           C=rsCV_resample.best_params_['model__C'],                      
                           random_state=41)  
        feature_map_nystroem = Nystroem(gamma=rsCV_resample.best_params_['featuremap__gamma'], 
                                        n_components=rsCV_resample.best_params_['featuremap__n_components'], 
                                        random_state=25)
        pipeline_steps = [('sampling_method', sampling), ('scale', StandardScaler()),
                          ('featuremap', feature_map_nystroem),('model',model)]        
        scores = get_cross_val_score(pipeline_steps, X_train, y_train, imb_class=True)

        pipe_model = Pipeline_imb(steps=pipeline_steps)
        #pipe_model.fit(X_train, y_train)
        
             
    elif model_name == 'RF':    #Random Forest
        rtree_model = RandomForestClassifier(random_state=41)
        # Create paramater search space    
        param_grid = {'model__n_estimators': np.linspace(10,50,5,dtype=np.int),
                  'model__max_depth': [2, 3, 4, 5],
                  'model__max_features': ['auto','log2', None],
                  'model__min_samples_split' : [2, 4, 6, 10],
                  'model__criterion': ['gini', 'entropy']}
        pipeline_steps = [('sampling_method', sampling),
                          ('scale', StandardScaler()),('model',rtree_model)]
        rsCV_resample = get_best_hyperparameters(pipeline_steps, param_grid, mscores, X_train, y_train, imb_class=True)
         
        model = RandomForestClassifier(n_estimators=rsCV_resample.best_params_['model__n_estimators'],
                   max_features=rsCV_resample.best_params_['model__max_features'],
                   criterion=rsCV_resample.best_params_['model__criterion'],
                   max_depth=rsCV_resample.best_params_['model__max_depth'],
                   min_samples_split=rsCV_resample.best_params_['model__min_samples_split'],                  
                   random_state=41)         
        pipeline_steps = [('sampling_method', sampling),
                          ('scale', StandardScaler()),('model',model)]
        scores = get_cross_val_score(pipeline_steps, X_train, y_train, imb_class=True)
        
        pipe_model = Pipeline_imb(steps=pipeline_steps)
        #pipe_model.fit(X_train, y_train)        
 
    elif model_name == 'GBC':   #Gradient Boosting
        gbc_model = GradientBoostingClassifier(random_state=41)
        # Create paramater search space    
        param_grid = {'model__n_estimators': np.linspace(10,50,5,dtype=np.int),
                  'model__learning_rate': [0.001, 0.01, 0.1, 1, 10],
                  'model__subsample': [0.5, 1], 
                  'model__max_depth': [3, 4, 5, 6],
                  'model__max_features': ['auto','log2', None],
                  'model__min_samples_split' : [2, 4, 6, 10],
                  'model__min_samples_leaf': [1, 2]}
        pipeline_steps = [('sampling_method', sampling),
                          ('scale', StandardScaler()),('model',gbc_model)]
        rsCV_resample = get_best_hyperparameters(pipeline_steps, param_grid, mscores, X_train, y_train, imb_class=True)
        model = GradientBoostingClassifier(n_estimators=rsCV_resample.best_params_['model__n_estimators'],
                   learning_rate=rsCV_resample.best_params_['model__learning_rate'],
                   subsample=rsCV_resample.best_params_['model__subsample'],
                   max_features=rsCV_resample.best_params_['model__max_features'],
                   min_samples_leaf=rsCV_resample.best_params_['model__min_samples_leaf'],
                   max_depth=rsCV_resample.best_params_['model__max_depth'],
                   min_samples_split=rsCV_resample.best_params_['model__min_samples_split'],                  
                   random_state=41)         
        pipeline_steps = [('sampling_method', sampling),
                          ('scale', StandardScaler()),('model',model)]
        scores = get_cross_val_score(pipeline_steps, X_train, y_train, imb_class=True)

        pipe_model = Pipeline_imb(steps=pipeline_steps)
        #pipe_model.fit(X_train, y_train)       
             
    return pipe_model, rsCV_resample, scores



if __name__ == "__main__":
    # ******************************
    #       Load the data set
    #   They are located in data/processed folder
    # ******************************
    # Load dataset1 (With smoking status)
    with open('df_strat_train_smoke_clean.pickle', 'rb') as read_to:
        df_strat_train_smoke = pickle.load(read_to)  
    # Data Set 1
    with open('df_strat_test_smoke_clean.pickle', 'rb') as read_to:
        df_strat_test_smoke = pickle.load(read_to)
    
    # Load dataset2 (Just patients who haven't smoked)
    with open('df_strat_train_neversmoke_clean.pickle', 'rb') as read_to:
        df_strat_train_neversmoke = pickle.load(read_to)  
    # Data Set 1
    with open('df_strat_test_neversmoke_clean.pickle', 'rb') as read_to:
        df_strat_test_neversmoke = pickle.load(read_to)        
  
    # A list of categorical attributes or features in the 2 data sets
    categorical_list_data1 = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'residence_type', 'smoking_status']
    categorical_list_data2 = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'residence_type']
    
    # A list of target variable in the 2 data sets
    target_list = ['stroke']
    
    # ***************************************** 
    #       Prepare dataframes for results
    # ***************************************** 
    col_names = ['strategy', 'model', 'macro_precision', 'macro_recall', 'macro_F1', 'pipemodel_object', 'rscvmodel_object']
    df_result_data1 = pd.DataFrame(columns=col_names)
    df_result_data2 = pd.DataFrame(columns=col_names)
        
    # ===========================================================================
    # Data 1 - Set drop_one dummy TRUE (For Regression with an intercept)
    df_X_train_smoke1, df_y_train_smoke1, df_X_test_smoke1, df_y_test_smoke1 = prepare_data_for_ML_models(df_strat_train_smoke, 
                    df_strat_test_smoke, categorical_list_data1, target_list, drop_first_dummy=True)
    # Data 2 - Set drop_one dummy TRUE (For Regression with an intercept)
    df_X_train_n1, df_y_train_n1, df_X_test_n1, df_y_test_n1 = prepare_data_for_ML_models(df_strat_train_neversmoke, 
        df_strat_test_neversmoke, categorical_list_data2, target_list, drop_first_dummy=True)
        
    # ===========================================================================
    # Data 1 - Set drop_one dummy FALSE
    df_X_train_smoke2, df_y_train_smoke2, df_X_test_smoke2, df_y_test_smoke2 = prepare_data_for_ML_models(df_strat_train_smoke, 
                    df_strat_test_smoke, categorical_list_data1, target_list)
    # Data 2 - Set drop_one dummy FALSE
    df_X_train_n2, df_y_train_n2, df_X_test_n2, df_y_test_n2 = prepare_data_for_ML_models(df_strat_train_neversmoke, 
        df_strat_test_neversmoke, categorical_list_data2, target_list)

    
    # ****************************** 
    #       Base Models
    # ******************************
    print("BASE MODEL ... ")
    
    print("Logistic Regression ...")
    lr_model = LogisticRegression(random_state=41)
    model1, scores1 = base_model(lr_model, df_X_train_smoke1, df_y_train_smoke1)
    add_list = ['base','Logistic Regression']+ scores1 + [model1, 'None']
    df_result_data1.loc[len(df_result_data1)] = add_list

    model2, scores2 = base_model(lr_model, df_X_train_n1, df_y_train_n1)    
    add_list = ['base','Logistic Regression']+ scores2 + [model2, 'None']
    df_result_data2.loc[len(df_result_data2)] = add_list  
    
    for modelname in ['SVM', 'RF', 'GBC']:
        if modelname =='SVM': 
            print("Nystroem and linearSVC ...")
            m_text = 'LinearSVC_nystroem'
            cmodel = LinearSVC(random_state=41)
            model1, scores1 = base_model(cmodel, df_X_train_smoke2, df_y_train_smoke2, is_svm=True)    
            model2, scores2 = base_model(cmodel, df_X_train_n2, df_y_train_n2, is_svm=True)    
             
        elif modelname == 'RF':
            print("Random Forest ...")  
            m_text = 'Random Forest'
            cmodel = RandomForestClassifier(random_state=41)
            model1, scores1 = base_model(cmodel, df_X_train_smoke2, df_y_train_smoke2)
            model2, scores2 = base_model(cmodel, df_X_train_n2, df_y_train_n2)   
 
        elif modelname == 'GBC':
            print("Gradient Boosting ...")  
            m_text = 'Gradient Boosting'
            cmodel = GradientBoostingClassifier(random_state=41)
            model1, scores1 = base_model(cmodel, df_X_train_smoke2, df_y_train_smoke2)
            model2, scores2 = base_model(cmodel, df_X_train_n2, df_y_train_n2)   
 
           
        add_list = ['base',m_text]+ scores1 + [model1, 'None']
        df_result_data1.loc[len(df_result_data1)] = add_list            
        add_list = ['base', m_text] + scores2 + [model2, 'None']
        df_result_data2.loc[len(df_result_data2)] = add_list   
    
    # ****************************** 
    #       Merhod: Class_weight
    # ******************************
    print("WEIGHT THE CLASSES - handling imbalanced classes ")
    
    print("Logistic Regression ...")
    model1, rsCVmodel1, scores1 = class_weight_model('Logistic', df_X_train_smoke1, df_y_train_smoke1)
    add_list = ['weight','Logistic Regression'] + scores1 + [model1, rsCVmodel1]
    df_result_data1.loc[len(df_result_data1)] = add_list    

    model2, rsCVmodel2, scores2 = class_weight_model('Logistic', df_X_train_n1, df_y_train_n1)    
    add_list = ['weight','Logistic Regression'] + scores2 + [model2, rsCVmodel2]
    df_result_data2.loc[len(df_result_data2)] = add_list      
        
    for modelname in ['SVM', 'RF', 'GBC']:
        if modelname == 'SVM': 
            print("Nystroem and linearSVC ...")
            m_text = 'LinearSVC_nystroem'
        elif modelname == 'RF':
            print("Random Forest ...")  
            m_text = 'Random Forest'
        elif modelname == 'GBC':
            print("Gradient Boosting ...")  
            m_text = 'Gradient Boosting'
            
        model1, rsCVmodel1, scores1 = class_weight_model(modelname, df_X_train_smoke2, df_y_train_smoke2)    
        model2, rsCVmodel2, scores2 = class_weight_model(modelname, df_X_train_n2, df_y_train_n2)   
        
        add_list = ['weight',m_text] + scores1 + [model1, rsCVmodel1]
        df_result_data1.loc[len(df_result_data1)] = add_list            
        add_list = ['weight',m_text] + scores2 + [model2, rsCVmodel2]
        df_result_data2.loc[len(df_result_data2)] = add_list   
        


    # ****************************** 
    #       Merhod: Resampling
    # ******************************
    print("Resampling - handling imbalanced classes ")
    resampling_methods = ['SMOTE', 'ADASYN', 'ROS']
    sm = SMOTE(random_state=42)
    adas = ADASYN(random_state=42)
    ros = RandomOverSampler(random_state=42)
    
    i = 0
    for samp in [sm, adas, ros]:
           
        print("Logistic Regression ...")
        model1, rsCVmodel1, scores1 = resampling_model('Logistic', samp, df_X_train_smoke1, df_y_train_smoke1)
        add_list = [resampling_methods[i],'Logistic Regression']+ scores1 + [model1, rsCVmodel1]
        df_result_data1.loc[len(df_result_data1)] = add_list
        
        model2, rsCVmodel2, scores2 = resampling_model('Logistic', samp, df_X_train_n1, df_y_train_n1)
        add_list = [resampling_methods[i],'Logistic Regression']+ scores2 + [model2, rsCVmodel2]
        df_result_data2.loc[len(df_result_data2)] = add_list
              
        for modelname in ['SVM', 'RF', 'GBC']:
            if modelname == 'SVM': 
                print("Nystroem and linearSVC ...")
                m_text = 'LinearSVC_nystroem'
            elif modelname == 'RF':
                print("Random Forest ...")  
                m_text = 'Random Forest'
            elif modelname == 'GBC':
                print("Gradient Boosting ...")  
                m_text = 'Gradient Boosting'
       
            model1, rsCVmodel1, scores1 = resampling_model(modelname, samp, df_X_train_smoke2, df_y_train_smoke2)
            model2, rsCVmodel2, scores2 = resampling_model(modelname, samp, df_X_train_n2, df_y_train_n2)
            
            add_list = [resampling_methods[i], m_text] + scores1 + [model1, rsCVmodel1]
            df_result_data1.loc[len(df_result_data1)] = add_list            
            add_list = [resampling_methods[i], m_text] + scores2 + [model2, rsCVmodel2]
            df_result_data2.loc[len(df_result_data2)] = add_list   

        i += 1     
         

    # Pickle the (Need to generate roc, precision-recall curves and feature important graphs) 
    #   They are located in results_in_dataframes folder
    #with open('df_resultmodels_smoke_data.pickle', 'wb') as write_to:
    #   pickle.dump(df_result_data1, write_to)
       
    #with open('df_resultmodels_neversmoke_data.pickle', 'wb') as write_to:
    #   pickle.dump(df_result_data2, write_to)        
        
    

    
    
    