#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: functions.py
Purpose: A collection of customized functions for 
        1) plotting ROC and Precision-Recall curves
        2) optimizing the classifiers' parameters (hyperparameters)
        3) implementing the oversampling and undersampling methods on the data
        4) calculate the performance metrics (scores) such as confusion matrix
            f1-score, recall, and precision.
Tools: Pandas, scikit-learn, imbalanced-learn, pickle, seaborn, and matplotlib
References:  
    https://stackoverflow.com/questions/48370150/how-to-implement-smote-in-cross-validation-and-gridsearchcv
    Incorporating Oversampling in an ML Pipeline https://bsolomon1124.github.io/oversamp/
    http://dev-aux.com/python/how-to-predict_proba-with-linearsvc  
    https://scikit-learn.org/stable/modules/model_evaluation.html#multimetric-scoring
"""


# Standard libraries
import pandas as pd
import numpy as np
import warnings

# Pipeline modules
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as Pipeline_imb


# Model selection
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import RandomizedSearchCV

# Metric
from sklearn import metrics
from sklearn.metrics import classification_report

# Warning - exceptons
from sklearn.exceptions import DataConversionWarning, UndefinedMetricWarning
from sklearn.exceptions import ConvergenceWarning

# visualization
import matplotlib.pyplot as plt

# suppress warning messages
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def prepare_data_for_ML_models(df_train, df_test, categorical_list, target_list, drop_first_dummy=False):
    '''
    This function is to convert the nominal categorical features to dummy variables
    
    Input arguments
    df_train, df_test: Training and test datasets
    categorical_list: A list of nominal categorical features or variables
    target_list: A list of target's names 
    drop_first_dummy: If is True, the first summy vaeiable is droped for each categorical  feature 
    '''
    # Handling categorical variables (or features)
    #   Apply one-hot encoding to nominal categorical features
    
    if categorical_list:    # If there are categorical features
        if drop_first_dummy:    # If drop first dummy variable is set to TRUE
            for c in categorical_list:
                df_train = pd.get_dummies(df_train, prefix=c, columns=[c], drop_first=True)
                df_test = pd.get_dummies(df_test, prefix=c, columns=[c], drop_first=True)
            
            if len(df_train.columns) != len(df_test.columns):
                # df_trains is missing of 'gender_Other' because there is no data
                gender_Other=list(np.zeros(len(df_train), dtype=int))
                df_train.insert(4, 'gender_Other', gender_Other)
        else:
            for c in categorical_list:
                df_train = pd.get_dummies(df_train, prefix=c, columns=[c])
                df_test = pd.get_dummies(df_test, prefix=c, columns=[c])
 
            if len(df_train.columns) != len(df_test.columns):
                # df_trains is missing of 'gender_Other' because there is no data
                gender_Other=list(np.zeros(len(df_train), dtype=int))
                df_train.insert(5, 'gender_Other', gender_Other)           

                
    # Prepare X and y data
    if target_list:
        df_X_train = df_train.drop(columns=target_list)
        df_y_train = df_train[target_list]
        df_X_test = df_test.drop(columns=target_list)
        df_y_test = df_test[target_list]   
        return df_X_train, df_y_train, df_X_test, df_y_test
    else:
        return print("ERROR: Missing of target variable name, Please provide target'names in target_list.")


def get_best_hyperparameters(pipeline_steps, param_grid, score, features, target, imb_class=False, cvfolds=5):
    '''
    Optimize/tuning hyperparameters with cross-validation only
    
    Input arguments
    pipeline_steps: The list of steps to be impelemented in a pipeline object
    param_grid: Hyperparameters' search space determined by user
    score: Specify a metric used to tune the hyperparameters
    features: The data that contain features only (i.e., X)
    target: The target (i.e., y)
    imb_class: Set True for imblearn's pipeline, else use sklearn's pipeline object
    cvfolds: Cross-validation splitting strategy
    '''
    
    # Data    
    X = features
    y = target

    # Construct the pipeline (from scikit-learn api)
    #pipe = Pipeline(steps=[('scale', StandardScaler()),
    #                       ('model',model)]
    #                        )
    if imb_class: 
        pipe = Pipeline_imb(steps=pipeline_steps)
    else:
        pipe = Pipeline(steps=pipeline_steps)
    # Stratified K-Folds cross-validator that returns stratified folds
    kf = StratifiedKFold(n_splits=cvfolds, shuffle=True, random_state=4)
    
    # Instamtiate the randomized search class
    rsCV = RandomizedSearchCV(pipe,
                                 param_grid,
                                 scoring=score, n_iter = 150, 
                                 cv = kf,
                                 error_score=0,
                                 random_state=1, 
                                 return_train_score=False)
    # Conduct random search
    rsCV.fit(X,y)

    return rsCV



def get_cross_val_score(pipeline_steps, features, target, imb_class=False, cvfolds=10):
    '''
    Calculate the classifier's performance score by cross validation 
    
    Input arguments    
    pipeline_steps: The list of steps to be impelemented in a pipeline object
    features: The data that contain features only (i.e., X)
    target: The target (i.e., y)
    imb_class: Set True for imblearn's pipeline, else use sklearn's pipeline object
    cvfolds: Cross-validation splitting strategy    
    '''
    # Data        
    X = features
    y = target
    # Specify the performance metrics    
    scores = ['precision_macro','recall_macro', 'f1_macro']
    # Construct the pipeline (from scikit-learn api)    
    if imb_class: 
        pipe = Pipeline_imb(steps=pipeline_steps)
    else:
        pipe = Pipeline(steps=pipeline_steps)
    # Stratified K-Folds cross-validator that returns stratified folds      
    kf = StratifiedKFold(n_splits=cvfolds, shuffle=True)
    
    scoresv = []
    for score in scores:    
        score_cv = cross_val_score(pipe, X, y, cv = kf, scoring=score)
        # Average of f1, precision, recall, and weighed f1        
        scoresv.append(np.mean(score_cv))

    '''
    # Calculate confusing matrix
    # This block codes is obtain from scikit-learm webpage
    # https://scikit-learn.org/stable/modules/model_evaluation.html#multimetric-scoring
    def tn(y_true, y_pred): return metrics.confusion_matrix(y_true, y_pred)[0, 0]
    def fp(y_true, y_pred): return metrics.confusion_matrix(y_true, y_pred)[0, 1]
    def fn(y_true, y_pred): return metrics.confusion_matrix(y_true, y_pred)[1, 0]
    def tp(y_true, y_pred): return metrics.confusion_matrix(y_true, y_pred)[1, 1]
    score_cm = {'tp': metrics.make_scorer(tp), 'tn': metrics.make_scorer(tn),
               'fp': metrics.make_scorer(fp), 'fn': metrics.make_scorer(fn)}
    cv_results = cross_validate(pipe.fit(X, y), X, y, cv=kf, scoring=score_cm)
    avg_tp =  np.mean(cv_results['test_tp'])
    avg_tn =  np.mean(cv_results['test_tn'])    
    avg_fp =  np.mean(cv_results['test_fp'])
    avg_fn =  np.mean(cv_results['test_fn'])
    
    
                  prediction           
                   0       1                
                 -----   -----      
              0 | TN   |  FP        
        True     -----   -----     
              1 | FN   |  TP       
    
    cm = [[avg_tn, avg_fp],[avg_fn, avg_tp]]
    
    # Average of confusing matrix    
    scoresv.append(cm)
    '''
    return scoresv


def calculate_metrics(model,features,target, model_name=''):
    '''
    A simple function to calculate the classifier's performance score

    Input arguments    
    model: Classifier object
    features: The data that contain features only (i.e., X)
    target: The target (i.e., y)
    model_name: The name of the classifier. Examples:
        Logistic Regression
        Naive Bayes
        SVM
        LinearSVC
        Decision Tree
        Random Forest
        etc    
    '''    
 
    # Data      
    X = features
    y = target
    
    # --------
    # Calculate the matrics
    # --------  
    # Predicted targets, y
    ypred = model.predict(X)
    
    # Confusion matrix
    cm = metrics.confusion_matrix(y,ypred)        
    # F1 score
    f1 = metrics.f1_score(y,ypred, average='macro')
    # Precision
    pc = metrics.precision_score(y,ypred, average='macro')
    # Recall
    rc = metrics.recall_score(y,ypred, average='macro')    

    return [pc,rc,f1,cm]

    """
    # Make a confusion matrix heatmap
    df_cm = pd.DataFrame(cm, index=['No', 'Yes'], columns=['No', 'Yes'])
    sns.heatmap(df_cm, annot=True, fmt="d")
    plt.ylabel('True condition')
    plt.xlabel('Predicted condition')
    plt.show()

    # If the classifier is SVM, calculate an extra metric, hinge luss 
    if model_name == 'SVM' or model_name == 'LinearSVC':
        # Predicted targets, y
        ypred_df = model.decision_function(X)
        h_L = metrics.hinge_loss(y,ypred_df)
        return [f1,pc,rc,h_L,cm]
    else:
        return [f1,pc,rc,0,cm]
    """



def plot_metric_curves(df_model, df_X_test, y_test,  df_X_train, df_y_train, metric_curve='roc', title='', plot_name=''):
    '''
    Plot the ROC and Precision-Recall curves for a mixture of classifiers

    Input arguments    
    df_model: Classifier objects store in a dataframe
    df_X_train, df_y_train: Training datasets, features and target respectively
    df_X_test, y_test: Test datasets, features and target respectively
    metric_curve: Specify which curves to plot, 'roc' or 'pc'
    title: Specify plot title name
    plot_name: Specify a unique name for image version (.png) of the plots
    '''
    # define the colormap
    cmap = plt.cm.tab10
    
    i = 0
    # Plot the precision-recall curves 
    for index in df_model.index.values:
        row = df_model.loc[[index]]
        model = row['pipemodel_object'].values[0]
        model.fit(df_X_train, df_y_train)
        
        if row['model'].values[0] == 'Logistic Regression':
            ypred_prob = model.predict_proba(df_X_test)[:,1]  
            label_text = 'LR+'+ row['strategy'].values[0]

        elif row['model'].values[0] == 'LinearSVC_nystroem':
            ypred_prob = model.decision_function(df_X_test) 
            label_text = 'LSVC+' + row['strategy'].values[0]
 
        elif row['model'].values[0] == 'Random Forest':     
            ypred_prob = model.predict_proba(df_X_test)[:,1]  
            label_text = 'RFC+' + row['strategy'].values[0]

        elif row['model'].values[0] == 'Gradient Boosting': 
            ypred_prob = model.decision_function(df_X_test)
            label_text = 'GBC+' + row['strategy'].values[0]

        if metric_curve == 'pc':        
            # Plot the precision-recall curves 
            pc, rc, threshold = metrics.precision_recall_curve(y_test, ypred_prob)    
            aps = metrics.average_precision_score(y_test, ypred_prob)
            label_text += ' (aps = ' + str(round(aps,2)) + ')'
            plt.plot(rc, pc, color=cmap(i),linewidth=2,label=label_text)            
        else:   #roc
            # Plot the ROC curves and calculate AUC
            fpr, tpr, threshold = metrics.roc_curve(y_test, ypred_prob)
            auc = metrics.roc_auc_score(y_test, ypred_prob)
            label_text += ' (auc = ' + str(round(auc,2)) + ')'
            plt.plot(fpr, tpr, color=cmap(i),linewidth=2,label=label_text)         
            
        i += 1
        
    # Make plot
    plt.xlim([0, 1])
    plt.ylim([0, 1]) 
    if metric_curve == 'pc':             
        plt.xlabel('Recall', fontsize=16)
        plt.ylabel('Precision', fontsize=16)
        plt.title(title)
        plt.legend(loc="upper right")
        #plt.savefig(plot_name+'_PRcurve', dpi=300)
        #plt.clf()
        plt.show()
    else:   #roc
        plt.plot([0, 1], [0, 1],color='orange',linestyle='dashed')
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.title(title)
        plt.legend(loc="lower right")
        #plt.savefig(plot_name+'_ROCcurve', dpi=300)
        #plt.clf()    
        plt.show()                   
        
    
if __name__ == "__main__":
    pass
