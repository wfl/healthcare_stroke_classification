#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 10:50:18 2019

@author: leong
"""

# Standard libraries
import pandas as pd
import pickle

from functions import prepare_data_for_ML_models
from functions import calculate_metrics, plot_metric_curves


if __name__ == "__main__":
    
    # ******************************
    #       Load the data set
    #  They are located in data/processed folder
    # ******************************
    # Load training dataset1 (With smoking status)
    with open('df_strat_train_smoke_clean.pickle', 'rb') as read_to:
        df_strat_train_smoke = pickle.load(read_to)  
    # Load test Dataset 1
    with open('df_strat_test_smoke_clean.pickle', 'rb') as read_to:
        df_strat_test_smoke = pickle.load(read_to)
    
    # Load training dataset2 (Just patients who haven't smoked)
    with open('df_strat_train_neversmoke_clean.pickle', 'rb') as read_to:
        df_strat_train_neversmoke = pickle.load(read_to)  
    # Load test Dataset 1
    with open('df_strat_test_neversmoke_clean.pickle', 'rb') as read_to:
        df_strat_test_neversmoke = pickle.load(read_to)        
  
    # A list of categorical attributes or features in the 2 data sets
    categorical_list_data1 = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'residence_type', 'smoking_status']
    categorical_list_data2 = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'residence_type']
    
    # A list of target variable in the 2 data sets
    target_list = ['stroke']
    
    # ===========================================================================
    #    Apply one-hot encoding to categorical variables (For Regression)
    #
    # Dataset 1 - Set drop_one dummy TRUE (For Regression with an intercept)
    df_X_train_smoke1, df_y_train_smoke1, df_X_test_smoke1, df_y_test_smoke1 = prepare_data_for_ML_models(df_strat_train_smoke, 
                    df_strat_test_smoke, categorical_list_data1, target_list, drop_first_dummy=True)
    # Dataset 2 - Set drop_one dummy TRUE (For Regression with an intercept)
    df_X_train_n1, df_y_train_n1, df_X_test_n1, df_y_test_n1 = prepare_data_for_ML_models(df_strat_train_neversmoke, 
        df_strat_test_neversmoke, categorical_list_data2, target_list, drop_first_dummy=True)
        
    # ===========================================================================
    #    Apply one-hot encoding to categorical variables (For SVC, RF, and GBC)
    #    
    # Dataset 1 - Set drop_one dummy FALSE
    df_X_train_smoke2, df_y_train_smoke2, df_X_test_smoke2, df_y_test_smoke2 = prepare_data_for_ML_models(df_strat_train_smoke, 
                    df_strat_test_smoke, categorical_list_data1, target_list)
    # Dataser 2 - Set drop_one dummy FALSE
    df_X_train_n2, df_y_train_n2, df_X_test_n2, df_y_test_n2 = prepare_data_for_ML_models(df_strat_train_neversmoke, 
        df_strat_test_neversmoke, categorical_list_data2, target_list)

    # **********************************************************
    #       Load model dataframes
    #         Contain model objects and validation's scores
    #   They are located in results_in_dataframes folder    
    # **********************************************************
    with open('df_resultmodels_smoke_data.pickle', 'rb') as read_to:
        df_result_data1 = pickle.load(read_to)  
        
    with open('df_resultmodels_neversmoke_data.pickle', 'rb') as read_to:
        df_result_data2 = pickle.load(read_to)  
    
        
    # ******************************************************
    #      Save the test results in the daraframes 
    # ******************************************************       
    col_names = ['strategy', 'model', 'macro_precision', 'macro_recall', 'macro_F1', 'conf_matrix']   
    df_test_data1 = pd.DataFrame(columns=col_names)
    df_test_data2 = pd.DataFrame(columns=col_names)

    
    # ******************************************************
    #      Calculate test results 
    # ******************************************************        
    # 1) Calculate precision, recall, F1, and confusion matrix from test dataset
    #       Store them in dataframes
    #for index, row in df_test_data1.iterrows():
    for index in range(len(df_result_data1)):
         row1 = df_result_data1.loc[[index]]
         row2 = df_result_data2.loc[[index]]
         
         if row1['model'].values[0] == 'Logistic Regression':
             model = row1['pipemodel_object'].values[0]
             model.fit(df_X_train_smoke1, df_y_train_smoke1)
             test_scores = calculate_metrics(model, df_X_test_smoke1, df_y_test_smoke1)
             add_list = [row1['strategy'].values[0], row1['model'].values[0]] + test_scores
             df_test_data1.loc[len(df_test_data1)] = add_list 
             
             model = row2['pipemodel_object'].values[0]
             model.fit(df_X_train_n1, df_y_train_n1)
             test_scores = calculate_metrics(model, df_X_test_n1, df_y_test_n1)
             add_list = [row2['strategy'].values[0], row2['model'].values[0]] + test_scores
             df_test_data2.loc[len(df_test_data2)] = add_list               
         else:        
             model = row1['pipemodel_object'].values[0]
             model.fit(df_X_train_smoke2, df_y_train_smoke2)
             test_scores = calculate_metrics(model, df_X_test_smoke2, df_y_test_smoke2)
             add_list = [row1['strategy'].values[0], row1['model'].values[0]] + test_scores
             df_test_data1.loc[len(df_test_data1)] = add_list 
             
             model = row2['pipemodel_object'].values[0]
             model.fit(df_X_train_n2, df_y_train_n2)
             test_scores = calculate_metrics(model, df_X_test_n2, df_y_test_n2)
             add_list = [row2['strategy'].values[0], row2['model'].values[0]] + test_scores
             df_test_data2.loc[len(df_test_data2)] = add_list   
     
    '''
    # They are located in results_in_dataframes folder 
    with open('df_testresult_smoke_data.pickle', 'wb') as write_to:
       pickle.dump(df_test_data1, write_to)
       
    with open('df_testresult_neversmoke_data.pickle', 'wb') as write_to:
       pickle.dump(df_test_data2, write_to)

    with open('df_testresult_smoke_data.pickle', 'rb') as read_to:
        df_test_data1 = pickle.load(read_to)  
        
    with open('df_testresult_neversmoke_data.pickle', 'rb') as read_to:
        df_test_data2 = pickle.load(read_to)  
    '''
        
    #----------------------------------------------
    # 2) Create ROC and precision-recall curves
    #----------------------------------------------    
    y_test_smoke1 = df_y_test_smoke1['stroke'].values
    y_test_n1 = df_y_test_n1['stroke'].values
    y_test_smoke2 = df_y_test_smoke2['stroke'].values
    y_test_n2 = df_y_test_n2['stroke'].values    
    

    # Logistics Regression
    df = df_result_data1[df_result_data1['model']=='Logistic Regression']
    plot_metric_curves(df, df_X_test_smoke1, y_test_smoke1, df_X_train_smoke1, df_y_train_smoke1, metric_curve='pc', title='Precision-Recall curves', plot_name='LR_smokes1') 
    plot_metric_curves(df, df_X_test_smoke1, y_test_smoke1, df_X_train_smoke1, df_y_train_smoke1, metric_curve='roc', title='ROC curves', plot_name='LR_smokes1')    
    df = df_result_data2[df_result_data2['model']=='Logistic Regression']
    plot_metric_curves(df, df_X_test_n1, y_test_n1, df_X_train_n1, df_y_train_n1, metric_curve='pc', title='Precision-Recall curves', plot_name='LR_nosmoke1') 
    plot_metric_curves(df, df_X_test_n1, y_test_n1, df_X_train_n1, df_y_train_n1, metric_curve='roc', title='ROC curves', plot_name='LR_nosmoke1')    
 
    # nystroem+LinearSVC
    df = df_result_data1[df_result_data1['model']=='LinearSVC_nystroem']
    plot_metric_curves(df, df_X_test_smoke2, y_test_smoke2, df_X_train_smoke2, df_y_train_smoke2, metric_curve='pc', title='Precision-Recall curves', plot_name='LSVC_smokes2')    
    plot_metric_curves(df, df_X_test_smoke2, y_test_smoke2, df_X_train_smoke2, df_y_train_smoke2, metric_curve='roc', title='ROC curves', plot_name='LSVC_smokes2') 
    df = df_result_data2[df_result_data2['model']=='LinearSVC_nystroem']
    plot_metric_curves(df, df_X_test_n2, y_test_n2, df_X_train_n2, df_y_train_n2, metric_curve='pc', title='Precision-Recall curves', plot_name='LSVC_nosmoke2')
    plot_metric_curves(df, df_X_test_n2, y_test_n2, df_X_train_n2, df_y_train_n2, metric_curve='roc', title='ROC curves', plot_name='LSVC_nosmoke2') 
    
    # Random Forest        
    df = df_result_data1[df_result_data1['model']=='Random Forest']
    plot_metric_curves(df, df_X_test_smoke2, y_test_smoke2, df_X_train_smoke2, df_y_train_smoke2, metric_curve='pc', title='Precision-Recall curves',plot_name='RFC_smokes2')    
    plot_metric_curves(df, df_X_test_smoke2, y_test_smoke2, df_X_train_smoke2, df_y_train_smoke2, metric_curve='roc', title='ROC curves',plot_name='RFC_smokes2')    
    df = df_result_data2[df_result_data2['model']=='Random Forest']
    plot_metric_curves(df, df_X_test_n2, y_test_n2, df_X_train_n2, df_y_train_n2, metric_curve='pc', title='Precision-Recall curves',plot_name='RFC_nosmoke2')
    plot_metric_curves(df, df_X_test_n2, y_test_n2, df_X_train_n2, df_y_train_n2, metric_curve='roc', title='ROC curves',plot_name='RFC_nosmoke2')

     # Gradient Boosting                 
    df = df_result_data1[df_result_data1['model']=='Gradient Boosting']
    plot_metric_curves(df, df_X_test_smoke2, y_test_smoke2, df_X_train_smoke2, df_y_train_smoke2, metric_curve='pc', title='Precision-Recall curves',plot_name='GBC_smokes2')    
    plot_metric_curves(df, df_X_test_smoke2, y_test_smoke2, df_X_train_smoke2, df_y_train_smoke2, metric_curve='roc', title='ROC curves',plot_name='GBC_smokes2')    
    df = df_result_data2[df_result_data2['model']=='Gradient Boosting']
    plot_metric_curves(df, df_X_test_n2, y_test_n2, df_X_train_n2, df_y_train_n2, metric_curve='pc', title='Precision-Recall curves',plot_name='GBC_nosmoke2')
    plot_metric_curves(df, df_X_test_n2, y_test_n2, df_X_train_n2, df_y_train_n2, metric_curve='roc', title='ROC curves',plot_name='GBC_nosmoke2')
    
    
       
    #-------------------------------------------------------    
    # 4) Find the feature importances from Random Forest
    #reference:  https://towardsdatascience.com/running-random-forests-inspect-the-feature-importances-with-this-code-2b00dd72b92e 
    #-------------------------------------------------------     
    df_LR_feature_imp1 = pd.DataFrame(index=df_X_train_smoke1.columns)
    df_LR_feature_imp2 = pd.DataFrame(index=df_X_train_n1.columns)    
    df_RF_feature_imp1 = pd.DataFrame(index=df_X_train_smoke2.columns)
    df_RF_feature_imp2 = pd.DataFrame(index=df_X_train_n2.columns)  
    df_GB_feature_imp1 = pd.DataFrame(index=df_X_train_smoke2.columns)
    df_GB_feature_imp2 = pd.DataFrame(index=df_X_train_n2.columns)
    
    for index in range(len(df_result_data1)):
        row1 = df_result_data1.loc[[index]]
        row2 = df_result_data2.loc[[index]]
         
        if row1['model'].values[0] == 'Logistic Regression':    
            model = row1['pipemodel_object'].values[0]            
            model.fit(df_X_train_smoke1, df_y_train_smoke1)
            if row1['strategy'].values[0] in ['SMOTE', 'ADASYN', 'ROS']:
                df_LR_feature_imp1[row1['strategy'].values[0]] = model.steps[2][1].coef_[0]
            else:
                df_LR_feature_imp1[row1['strategy'].values[0]] = model.steps[1][1].coef_[0]

            model = row2['pipemodel_object'].values[0]
            model.fit(df_X_train_n1, df_y_train_n1)          
            if row2['strategy'].values[0] in ['SMOTE', 'ADASYN', 'ROS']:
                df_LR_feature_imp2[row2['strategy'].values[0]] = model.steps[2][1].coef_[0]
            else:
                df_LR_feature_imp2[row2['strategy'].values[0]] = model.steps[1][1].coef_[0] 
                
        elif row1['model'].values[0] == 'Random Forest':
            model = row1['pipemodel_object'].values[0]
            model.fit(df_X_train_smoke2, df_y_train_smoke2)             
            if row1['strategy'].values[0] in ['SMOTE', 'ADASYN', 'ROS']:
                df_RF_feature_imp1[row1['strategy'].values[0]] = model.steps[2][1].feature_importances_
            else:
                df_RF_feature_imp1[row1['strategy'].values[0]] = model.steps[1][1].feature_importances_
           
            model = row2['pipemodel_object'].values[0]
            model.fit(df_X_train_n2, df_y_train_n2)
            if row2['strategy'].values[0] in ['SMOTE', 'ADASYN', 'ROS']:
                df_RF_feature_imp2[row2['strategy'].values[0]] = model.steps[2][1].feature_importances_
            else:
                df_RF_feature_imp2[row2['strategy'].values[0]] = model.steps[1][1].feature_importances_
          
        elif row1['model'].values[0] == 'Gradient Boosting':
            model = row1['pipemodel_object'].values[0]
            model.fit(df_X_train_smoke2, df_y_train_smoke2)             
            if row1['strategy'].values[0] in ['SMOTE', 'ADASYN', 'ROS']:
                df_GB_feature_imp1[row1['strategy'].values[0]] = model.steps[2][1].feature_importances_
            else:
                df_GB_feature_imp1[row1['strategy'].values[0]] = model.steps[1][1].feature_importances_
           
            model = row2['pipemodel_object'].values[0]
            model.fit(df_X_train_n2, df_y_train_n2)
            if row2['strategy'].values[0] in ['SMOTE', 'ADASYN', 'ROS']:
                df_GB_feature_imp2[row2['strategy'].values[0]] = model.steps[2][1].feature_importances_
            else:
                df_GB_feature_imp2[row2['strategy'].values[0]] = model.steps[1][1].feature_importances_
             
    
    # Make plots and save them
    df_LR_feature_imp1.plot.barh()
    plt.title('Logistic Regression (with smoking status)')
    #plt.savefig('LR_smokes1_featureImportance', bbox_inches='tight', dpi=300)
    #lt.clf()    
    plt.show()

    df_LR_feature_imp2.plot.barh()
    plt.title('Logistic Regression (have never smoked)')
    #plt.savefig('LR_nosmokes1_featureImportance', bbox_inches='tight', dpi=300)
    #plt.clf()    
    plt.show()    

    df_RF_feature_imp1.plot.barh()
    plt.title('Random Forest Classifier (with smoking status)')
    #plt.savefig('RFC_smokes2_featureImportance', bbox_inches='tight', dpi=300)
    #plt.clf()    
    plt.show()  
    
    df_RF_feature_imp2.plot.barh()
    plt.title('Random Forest Classifier (have never smoked)')    
    #plt.savefig('RFC_nosmokes2_featureImportance', bbox_inches='tight', dpi=300)
    #plt.clf()    
    plt.show()  

    df_GB_feature_imp1.plot.barh()
    plt.title('Gradient Boosting Classifier (with smoking status)')    
    #plt.savefig('GBC_smokes2_featureImportance', bbox_inches='tight', dpi=300)
    #plt.clf()    
    plt.show()  

    df_GB_feature_imp2.plot.barh()
    plt.title('Gradient Boosting Classifier (have never smoked)')    
    #plt.savefig('GBC_nosmokes2_featureImportance', bbox_inches='tight', dpi=300)
    #plt.clf()    
    plt.show()  

    '''  
    list_df_feature_importances = [df_LR_feature_imp1, df_LR_feature_imp2, df_RF_feature_imp1, \
                                   df_RF_feature_imp2, df_GB_feature_imp1, df_GB_feature_imp2]        
    
    ## Pickle he datadrames with feature importance values and coefficients in a LIST
    #    They are located in results_in_dataframes folder 
    with open('list_df_feature_importances.pickle', 'wb') as write_to:
       pickle.dump(list_df_feature_importances, write_to)
    '''
    


    