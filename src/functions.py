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

# Preprocessing module
from sklearn.preprocessing import StandardScaler

# Pipeline modules
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as Pipeline_imb

from sklearn.calibration import CalibratedClassifierCV

# Model selection
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.model_selection import RandomizedSearchCV

# Metric
from sklearn import metrics
from sklearn.metrics import classification_report

# Warning - exceptons
from sklearn.exceptions import DataConversionWarning, UndefinedMetricWarning
from sklearn.exceptions import ConvergenceWarning

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# suppress warning messages
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def get_hyperparameter_opt(model, param_grid, scores, features, target, cvfolds=5):
    '''
    Optimize/tuning hyperparameters with cross-validation only
    
    model: Classifier object
    param_grid: Hyperparameters' search space determined by user
    scores: Specify the performance metrics used to tune the hyperparameters
    features: The data that contain features only (i.e., X)
    target: The target (i.e., y)
    cvfolds: Cross-validation splitting strategy
    '''
    
    # Data    
    X = features
    y = target

    # Construct the pipeline (from scikit-learn api)
    pipe = Pipeline(steps=[('scale', StandardScaler()),
                           ('model',model)]
                            )

    # Stratified K-Folds cross-validator that returns stratified folds
    kf = StratifiedKFold(n_splits=cvfolds, shuffle=True, random_state=4)
    
    for score in scores:
        # Instamtiate the randomized search class
        rsCV = RandomizedSearchCV(pipe,
                                  param_grid,
                                  scoring=score, n_iter = 100, 
                                  cv = kf,
                                  error_score=0,
                                  random_state=1, 
                                  return_train_score=False)
        # Conduct random search
        rsCV.fit(X,y)

    return rsCV


def get_hyperparameter_opt_imblearn(model, sampling_method, param_grid, scores, features, target, cvfolds=3):
    '''
    For imbalanced class data 
    Optimize/tuning hyperparameters with cross-validation after implementing
    a sampling method to sample the data
    
    model: Classifier object
    sampling_method: Specify a sampling method
    param_grid: Hyperparameters' search space determined by user
    scores: Specify the performance metrics used to tune the hyperparameters
    features: The data that contain features only (i.e., X)
    target: The target (i.e., y)
    cvfolds: Cross-validation splitting strategy
    '''

    # Data
    X = features
    y = target

    # Construct the pipeline (from imbalanced-learn api)    
    pipe = Pipeline(steps=[('scale', StandardScaler()),
                           ('sampling_method', sampling_method),
                           ('model',model)]
                            )
    # Stratified K-Folds cross-validator that returns stratified folds    
    kf = StratifiedKFold(n_splits=cvfolds, shuffle=True, random_state=4)
    
    for score in scores:
        # Instamtiate the randomized search class
        rsCV = RandomizedSearchCV(pipe,
                                  param_grid,
                                  scoring= score, n_iter = 100, 
                                  cv = kf,
                                  random_state=1, 
                                  return_train_score=False)
        # Conduct random search
        rsCV.fit(X,y)

    return rsCV


def get_cross_val_score_imblearn(model, sampling_method, features, target, cvfolds=10):
    '''
    For imbalanced class data     
    Calculate the classifier's performance score by cross validation 
    
    model: Classifier object
    sampling_method: Specify a sampling method
    features: The data that contain features only (i.e., X)
    target: The target (i.e., y)
    cvfolds: Cross-validation splitting strategy
    '''
    # Data    
    X = features
    y = target
    
    # Specify the performance metrics
    scores = ['f1_macro','precision_macro','recall_macro','f1_weighted']
    # Construct the pipeline (from imbalanced-learn api)    
    pipe = Pipeline_imb(steps=[('scale', StandardScaler()),
                           ('sampling_method', sampling_method),
                           ('model',model)])    
    # Stratified K-Folds cross-validator that returns stratified folds         
    kf = StratifiedKFold(n_splits=cvfolds, shuffle=True)
    
    scoresv = []
    for score in scores:    
        score_cv = cross_val_score(pipe, X, y, cv = kf, scoring=score)
        # Average of f1, precision, recall, and weighed f1
        scoresv.append(np.mean(score_cv))

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
    
    '''
                  prediction           
                   0       1                
                 -----   -----      
              0 | TN   |  FP        
        True     -----   -----     
              1 | FN   |  TP       
    '''
    # Average of confusing matrix
    cm = [[avg_tn, avg_fp],[avg_fn, avg_tp]]
    scoresv.append(cm)
    
    return scoresv


def get_cross_val_score(model, features, target, cvfolds=10):
    '''
    Calculate the classifier's performance score by cross validation 
    
    model: Classifier object
    features: The data that contain features only (i.e., X)
    target: The target (i.e., y)
    cvfolds: Cross-validation splitting strategy    
    '''
    # Data        
    X = features
    y = target
    # Specify the performance metrics    
    scores = ['f1_macro','precision_macro','recall_macro','f1_weighted']
    # Construct the pipeline (from scikit-learn api)    
    pipe = Pipeline(steps=[('scale', StandardScaler()),('model',model)])
    # Stratified K-Folds cross-validator that returns stratified folds      
    kf = StratifiedKFold(n_splits=cvfolds, shuffle=True)
    
    scoresv = []
    for score in scores:    
        score_cv = cross_val_score(pipe, X, y, cv = kf, scoring=score)
        # Average of f1, precision, recall, and weighed f1        
        scoresv.append(np.mean(score_cv))

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
    
    '''
                  prediction           
                   0       1                
                 -----   -----      
              0 | TN   |  FP        
        True     -----   -----     
              1 | FN   |  TP       
    '''
    cm = [[avg_tn, avg_fp],[avg_fn, avg_tp]]
    # Average of confusing matrix    
    scoresv.append(cm)
    return scoresv



def calculate_metrics(model,features,target, model_name=''):
    '''
    A simple function to calculate the classifier's performance score
    
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


def plot_metric_curves(models, data,  model_names, plot_name='', Xtrain=None, ytrain=None):
    '''
    Plot the ROC and Precision-Recall curves for a mixture of classifiers
    
    model: Classifier object
    data: A list contains a list of features and target
    model_name: The name of the classifier. Examples:
        Logistic Regression
        Naive Bayes
        SVM
        LinearSVC
        Decision Tree
        Random Forest
        etc    
    plot_name: Specify a unique name for image version (.png) of the plots
    Xtrain: Training data that contain features only
    ytrain: Training data for target (y)
    '''    
    # define the colormap
    cmap = plt.cm.tab10
    
    for i in range(len(models)):
        # Load the data set        
        if model_names[i] in 'Logistic Regression' or model_names[i] is 'Logistic-SMOTE':
        #if i==0:
            X, y = data[0]
        else:
            X, y = data[1]
       
        # If the classifier is SVM, then calculate the class probabilities 
        if model_names[i] == 'LinearSVC' or 'SVM'in model_names[i]:
            if Xtrain is not None:
                # Platt’s method calibration
                cal_linearsvc = CalibratedClassifierCV(base_estimator=models[i], 
                                                        method='sigmoid', cv=5) 
                cal_linearsvc.fit(Xtrain, ytrain)
                ypred_prob = cal_linearsvc.predict_proba(X)[:,1] 
            else:
               ypred_prob = models[i].decision_function(X)
        else:
            ypred_prob = models[i].predict_proba(X)[:,1]     
            
        # Plot the ROC curves and calculate AUC
        fpr, tpr, threshold = metrics.roc_curve(y,ypred_prob)
        auc = metrics.roc_auc_score(y,ypred_prob)
        plt.plot(fpr, tpr, color=cmap(i),linewidth=4,
                 label= model_names[i]+ '(auc = %0.2f)' % auc)
        
    plt.plot([0, 1], [0, 1],color='orange',linestyle='dashed')
    plt.xlim([0, 1])
    plt.ylim([0, 1])      
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC curves')
    plt.legend(loc="lower right")
    #plt.savefig(plot_name+'_ROCcurve', dpi=300)
    #plt.clf()    
    plt.show()

    for i in range(len(models)):
        # Load the data set      
        if model_names[i] in 'Logistic Regression' or model_names[i] is 'Logistic-SMOTE':
        #if i==0:
            X, y = data[0]       
        else:
            X, y = data[1]
        
        # If the classifier is SVM, then calculate the class probabilities 
        if model_names[i] == 'LinearSVC' or 'SVM'in model_names[i]:
            if Xtrain is not None:
                # Platt’s method calibration
                cal_linearsvc = CalibratedClassifierCV(base_estimator=models[i], 
                                                        method='sigmoid', cv=5) 
                cal_linearsvc.fit(Xtrain, ytrain)
                ypred_prob = cal_linearsvc.predict_proba(X)[:,1] 
            else:
               ypred_prob = models[i].decision_function(X)
        else:
            ypred_prob = models[i].predict_proba(X)[:,1]  
            
        # Plot the precision-recall curves and calculate the aps
        pc, rc, threshold = metrics.precision_recall_curve(y,ypred_prob)
        aps = metrics.average_precision_score(y,ypred_prob)
        plt.plot(rc, pc, color=cmap(i),linewidth=4,
                 label= model_names[i]+ '(aps = %0.2f)' % aps)

    plt.plot([1,0], [0,1],color='orange',linestyle='dashed')
    plt.xlim([0, 1])
    plt.ylim([0, 1])        
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.title('Precision-Recall curves')
    plt.legend(loc="upper right")
    #plt.savefig(plot_name+'_PRcurve', dpi=300)
    #plt.clf()
    plt.show()


def plot_ind_metric_curves(models, data,  model_names, plot_name, Xtrain=None, ytrain=None):
    '''
    Plot the ROC and Precision-Recall curves for a specific classifier
    
    model: Classifier object
    data: A list contains a list of features and target
    model_name: The name of the classifier. Examples:
        Logistic Regression
        Naive Bayes
        SVM
        LinearSVC
        Decision Tree
        Random Forest
        etc    
    plot_name: Specify a unique name for image version (.png) of the plots
    Xtrain: Training data that contain features only
    ytrain: Training data for target (y)
    '''    

    # define the colormap
    cmap = plt.cm.tab10
    
    for i in range(len(models)):
        # Load the data set           
        X, y = data[0]

        # If the classifier is SVM, then calculate the class probabilities 
        if 'LinearSVC' in model_names[i] or 'SVM'in model_names[i]:
            if Xtrain is not None:
                # Implement Platt’s method calibration
                cal_linearsvc = CalibratedClassifierCV(base_estimator=models[i], 
                                                        method='sigmoid', cv=5) 
                cal_linearsvc.fit(Xtrain, ytrain)
                ypred_prob = cal_linearsvc.predict_proba(X)[:,1] 
            else:
               ypred_prob = models[i].decision_function(X)
        else:
            ypred_prob = models[i].predict_proba(X)[:,1]     
            
        # Plot the ROC curves and calculate AUC
        fpr, tpr, threshold = metrics.roc_curve(y,ypred_prob)
        auc = metrics.roc_auc_score(y,ypred_prob)
        plt.plot(fpr, tpr, color=cmap(i),linewidth=2,
                 label= model_names[i]+ '(auc = %0.2f)' % auc)
        
    plt.plot([0, 1], [0, 1],color='orange',linestyle='dashed')
    plt.xlim([0, 1])
    plt.ylim([0, 1])      
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC curves')
    plt.legend(loc="lower right")
    #plt.savefig(plot_name+'_ROCcurve', dpi=300)
    #plt.clf()    
    plt.show()

    for i in range(len(models)):
        # Load the data set           
        X, y = data[0]

        # If the classifier is SVM, then calculate the class probabilities 
        if 'LinearSVC' in model_names[i] or 'SVM'in model_names[i]:
            if Xtrain is not None:
                # Implement Platt’s method calibration
                cal_linearsvc = CalibratedClassifierCV(base_estimator=models[i], 
                                                        method='sigmoid', cv=5) 
                cal_linearsvc.fit(Xtrain, ytrain)
                ypred_prob = cal_linearsvc.predict_proba(X)[:,1] 
            else:
               ypred_prob = models[i].decision_function(X)
        else:
            ypred_prob = models[i].predict_proba(X)[:,1]  
            
        # Plot the precision-recall curves and calculate the aps
        pc, rc, threshold = metrics.precision_recall_curve(y,ypred_prob)
        aps = metrics.average_precision_score(y,ypred_prob)
        plt.plot(rc, pc, color=cmap(i),linewidth=2,
                 label= model_names[i]+ '(aps = %0.2f)' % aps)

    plt.plot([1,0], [0,1],color='orange',linestyle='dashed')
    plt.xlim([0, 1])
    plt.ylim([0, 1])        
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.title('Precision-Recall curves')
    plt.legend(loc="upper right")
    #plt.savefig(plot_name+'_PRcurve', dpi=300)
    #plt.clf()
    plt.show()



if __name__ == "__main__":
    pass
