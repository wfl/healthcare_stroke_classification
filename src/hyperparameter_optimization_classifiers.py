"""
Name: hyperparameter_optimization_classifiers.py
Purpose: Models' paramaters (hyperparameters) are tuned with 
        Randomized Search to improve the models' ability in predicting the 
        positive class that is a minority class in an imbalanced dataset. 
Tools: Pandas, scikit-learn, imbalanced-learn, and pickle
References:  
    https://stackoverflow.com/questions/48370150/how-to-implement-smote-in-cross-validation-and-gridsearchcv
    Incorporating Oversampling in an ML Pipeline https://bsolomon1124.github.io/oversamp/
"""


"""
Name: spot_checking_classifiers.py
Purpose: As a starting point, we want a few base models that yield good
        performance. Here, one or a few base models are determined from 
        a list of models.
Tools: Pandas, scikit-learn, and pickle
References:  
    https://stats.stackexchange.com/questions/288095/what-algorithms-require-one-hot-encoding
    https://www.kaggle.com/c/home-credit-default-risk/discussion/63499
    http://dev-aux.com/python/how-to-predict_proba-with-linearsvc
"""


# Standard libraries
import pandas as pd
import numpy as np
import pickle

# Preprocessing module
from sklearn.preprocessing import StandardScaler

# The classification models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

# Model Selection
from sklearn.model_selection import train_test_split

# Load customized functions from the functions.py
from functions import get_hyperparameter_opt, get_cross_val_score, plot_metric_curves


if __name__ == "__main__":

    dict_scores = {}
    models = []
    data_val = []
    model_names = []
    mscores = ['f1_macro', 'f1_weighted', 'precision_macro','recall_macro']
    
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
    X_val_scaled = std.transform(X_val.values)
    X_test_scaled1 = std.transform(X_test.values)    
   
    # Model 1: Logistic Regression   
    print("Logistic Regression ...")
    # Create paramater search space
    param_grid = {'model__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
                  'model__penalty': ['l1','l2'],
                  'model__class_weight':['balanced', None]}
    # Create a model instance     
    model = LogisticRegression(random_state=41)
    # Optimize the hyperparamaters in param_grid
    rsCV = get_hyperparameter_opt(model,param_grid, mscores,X_train_val, y_train_val)
    print("LR: RS best f1 = ",rsCV.best_score_) 
    
    '''
    After optimization, this is the model with the best parameters.
    lr_model =LogisticRegression(C=0.01, class_weight='balanced', dual=False,
           fit_intercept=True, intercept_scaling=1, max_iter=100,
           multi_class='warn', n_jobs=None, penalty='l1', random_state=41,
           solver='warn', tol=0.0001, verbose=0, warm_start=False)    
    '''
    lr_model = LogisticRegression(C=rsCV.best_params_['model__C'],
                    penalty=rsCV.best_params_['model__penalty'],
                    class_weight=rsCV.best_params_['model__class_weight'],
                    random_state=41) 
    # Get metrics or score through cross validation      
    scores = get_cross_val_score(lr_model, X_train_val, y_train_val)
    # Save the score in a dictionary     
    dict_scores['Logistic Regression']= scores
    # Fit the model with the scaled training data        
    lr_model.fit(X_train_scaled, y_train)
    # Save the model object in a list
    models.append(lr_model)
    # Save the model's name in a list       
    model_names.append('Logistic')
    # If it is in the testing stage, use validation data
    data_val.append([X_val_scaled,y_val])
    # Otherwiese, use the test data set
    data_val.append([X_test_scaled1,y_test1])
    
    
    
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
    

    
    print("LinearSVC ...")
    # Create paramater search space    
    param_grid = {'model__penalty': ['l1','l2'],
                  'model__loss':['squared_hinge'],
                  'model__C': np.logspace(-1, 3, 10),
                  'model__class_weight':['balanced', None]}
    # Create a model instance     
    model = LinearSVC(random_state=41)
    # Optimize the hyperparamaters in param_grid    
    rsCV = get_hyperparameter_opt(model,param_grid, mscores,X_train_val, y_train_val)
    print("LinearSVC: RS best f1 = ",rsCV.best_score_)
    '''
    After optimization, this is the model with the best parameters.
    lsvc_model = LinearSVC(C=0.1, class_weight='balanced', dual=True, fit_intercept=True,
      intercept_scaling=1, loss='squared_hinge', max_iter=1000,
      multi_class='ovr', penalty='l2', random_state=41, tol=0.0001,
      verbose=0)    
    '''
    lsvc_model = LinearSVC(penalty=rsCV.best_params_['model__penalty'],
                           loss=rsCV.best_params_['model__loss'],
                           C=rsCV.best_params_['model__C'],
                           class_weight=rsCV.best_params_['model__class_weight'],                       
                           random_state=41)

    # Get metrics or score through cross validation  
    scores = get_cross_val_score(lsvc_model, X_train_val, y_train_val)
    # Save the score in a dictionary      
    dict_scores['LinearSVC']= scores
    # Fit the model with the scaled training data      
    lsvc_model.fit(X_train_scaled, y_train)
    # Save the model object in a list   
    models.append(lsvc_model)
    # Save the model's name in a list      
    model_names.append('LinearSVC')
    # Reserve for SVM and LinearSVC. They are for Platt’s method calibration,
    #   to calculate probability class if sklearn.metrics.decision_function()
    #   isn't available. 
    #Xtrain = X_train_scaled
    #ytrain = y_train

#    ## Ignore this classifier for now    
#    print("SVM-rbf...")
#    # Create paramater search space    
#    param_grid = {'model__C': np.logspace(-1, 3, 10),
#                  'model__gamma': np.linspace(0.0001, 10, 10),
#                  'model__class_weight':['balanced', None]}
#    # Create a model instance      
#    model = SVC(kernel='rbf', random_state=41)
#    # Optimize the hyperparamaters in param_grid    
#    rsCV = get_hyperparameter_opt(model,param_grid, mscores,X_train_val, y_train_val) 
#    print("SVM-rbf: RS best f1 = ",rsCV.best_score_)
#    
#    svm_rbf_model = SVC(kernel='rbf', 
#                        C=rsCV.best_params_['model__C'], 
#                        gamma=rsCV.best_params_['model__gamma'], 
#                        class_weight=rsCV.best_params_['model__class_weight'],
#                        random_state=41)

#    # Get metrics or score through cross validation  
#    scores = get_cross_val_score(svm_rbf_model, X_train_val, y_train_val) 
#    # Save the score in a dictionary     
#    dict_scores['SVM-rbf']= scores
#    # Fit the model with the scaled training data       
#    svm_rbf_model.fit(X_train_scaled, y_train)
#    # Save the model object in a list    
#    models.append(svm_rbf_model)    
#    # Save the model's name in a list      
#    model_names.append('SVM-rbf')
    

    # Model 3: Naive Bayes
    print("Naive Bayes ...")
    # Create a model instance     
    Gnb_model = GaussianNB()
    scores = get_cross_val_score(model, X, y)  
    # Save the score in a dictionary      
    dict_scores['Naive Bayes']= scores
    # Fit the model with the scaled training data      
    Gnb_model.fit(X_train_scaled, y_train)
    # Save the model object in a list 
    models.append(Gnb_model)
    # Save the model's name in a list      
    model_names.append('Naive Bayes') 
    
    # Model 4: Decision Tree Classifier
    print("Decision Tree ...")
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
    rsCV = get_hyperparameter_opt(model,param_grid, mscores,X_train_val, y_train_val) 
    print("DT: RS best f1 = ",rsCV.best_score_)
    '''
    After optimization, this is the model with the best parameters.
    Ddtree_model = DecisionTreeClassifier(class_weight='balanced', criterion='gini',
             max_depth=3.2222222222222223, max_features='log2',
             max_leaf_nodes=None, min_impurity_decrease=0.0,
             min_impurity_split=None, min_samples_leaf=1,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             presort=False, random_state=41, splitter='best')
    '''    
    dtree_model = DecisionTreeClassifier(
                   max_features=rsCV.best_params_['model__max_features'],
                   criterion=rsCV.best_params_['model__criterion'],
                   splitter=rsCV.best_params_['model__splitter'],
                   max_depth=rsCV.best_params_['model__max_depth'],
                   min_samples_split=rsCV.best_params_['model__min_samples_split'],
                   class_weight=rsCV.best_params_['model__class_weight'],                  
                   random_state=41)    
    # Get metrics or score through cross validation  
    scores = get_cross_val_score(dtree_model, X_train_val, y_train_val)
    # Save the score in a dictionary     
    dict_scores['Decision Tree']= scores
    # Fit the model with the scaled training data       
    dtree_model.fit(X_train_scaled, y_train)
    # Save the model object in a list 
    models.append(dtree_model)
    # Save the model's name in a list      
    model_names.append('Decision Tree')
        
    # Model 5: Bagging Classifier with decision tree
    print("Bagging Classifier with decision tree ...")
    # Create paramater search space    
    param_grid = {'model__n_estimators': np.linspace(10,50,5,dtype=np.int),
                  'model__max_features' : [2, 4, 6, 8, 10, 12]}
    # Create a model instance      
    model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), random_state=41)
    # Optimize the hyperparamaters in param_grid    
    rsCV = get_hyperparameter_opt(model,param_grid, mscores,X_train_val, y_train_val) 
    print("Bagging-DT: RS best f1 = ",rsCV.best_score_)
    '''
    After optimization, this is the model with the best parameters.
    bag_dtree_model = BaggingClassifier(base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
             max_features=None, max_leaf_nodes=None,
             min_impurity_decrease=0.0, min_impurity_split=None,
             min_samples_leaf=1, min_samples_split=2,
             min_weight_fraction_leaf=0.0, presort=False, random_state=None,
             splitter='best'),
          bootstrap=True, bootstrap_features=False, max_features=12,
          max_samples=1.0, n_estimators=20, n_jobs=None, oob_score=False,
          random_state=41, verbose=0, warm_start=False)    
    '''    
    bag_dtree_model = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                        n_estimators=rsCV.best_params_['model__n_estimators'],
                        max_features=rsCV.best_params_['model__max_features'],
                        random_state=41)
    bag_dtree_model = BaggingClassifier(base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
             max_features=None, max_leaf_nodes=None,
             min_impurity_decrease=0.0, min_impurity_split=None,
             min_samples_leaf=1, min_samples_split=2,
             min_weight_fraction_leaf=0.0, presort=False, random_state=None,
             splitter='best'),
          bootstrap=True, bootstrap_features=False, max_features=12,
          max_samples=1.0, n_estimators=20, n_jobs=None, oob_score=False,
          random_state=41, verbose=0, warm_start=False)    
    # Get metrics or score through cross validation  
    scores = get_cross_val_score(bag_dtree_model, X_train_val, y_train_val)
    # Save the score in a dictionary      
    dict_scores['Bagging-Decision Tree']= scores
    # Fit the model with the scaled training data      
    bag_dtree_model.fit(X_train_scaled, y_train)
    # Save the model object in a list 
    models.append(bag_dtree_model)
    # Save the model's name in a list      
    model_names.append('Bagging-DT')
    
    # Model 6: Random Forest Classifier
    print("Random Forest ...")
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
    rsCV = get_hyperparameter_opt(model,param_grid, mscores,X_train_val, y_train_val)
    print("RF: RS best f1 = ",rsCV.best_score_)
    '''
    After optimization, this is the model with the best parameters.
    rtree_model = RandomForestClassifier(bootstrap=True, class_weight='balanced',
             criterion='entropy', max_depth=5.444444444444445,
             max_features=6, max_leaf_nodes=None, min_impurity_decrease=0.0,
             min_impurity_split=None, min_samples_leaf=1,
             min_samples_split=6, min_weight_fraction_leaf=0.0,
             n_estimators=40, n_jobs=None, oob_score=False, random_state=41,
             verbose=0, warm_start=False)    
    '''    
    rtree_model = RandomForestClassifier(n_estimators=rsCV.best_params_['model__n_estimators'],
                   max_features=rsCV.best_params_['model__max_features'],
                   criterion=rsCV.best_params_['model__criterion'],
                   max_depth=rsCV.best_params_['model__max_depth'],
                   min_samples_split=rsCV.best_params_['model__min_samples_split'],
                   class_weight=rsCV.best_params_['model__class_weight'],                  
                   random_state=41) 
    
    # Get metrics or score through cross validation  
    scores = get_cross_val_score(rtree_model, X_train_val, y_train_val) 
    # Save the score in a dictionary      
    dict_scores['Random Forest']= scores
    # Fit the model with the scaled training data      
    rtree_model.fit(X_train_scaled, y_train)
    # Save the model object in a list 
    models.append(rtree_model)
    # Save the model's name in a list      
    model_names.append('Random Forest')
    # If it is in the testing stage, use validation data     
#    data_val.append([X_val_scaled,y_val])
    # Otherwiese, use the test data set       
    data_val.append([X_test_scaled,y_test])
    
    # Print the scores in dataframe
    df_score_tune = pd.DataFrame.from_dict(dict_scores,orient='index', 
        columns=['F1_macro','Precision_macro','Recall_macro','F1_weight', 'Confusion Matrix'])
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_score_tune)
        
        
    # Plot the ROC and Precision-Recall curves. plot_metric_curves is a customized function
    plot_metric_curves(models, data_val, model_names, plot_name='basemodels_tune', Xtrain=None, ytrain=None)

    #---------------------------------------------------
    # Pickle the scores and list of model objects
    #---------------------------------------------------         
    with open('basemodelstune_dfscore.pickle', 'wb') as write_to:
        pickle.dump(df_score_tune, write_to)

    models_tune = models
    with open('basemodels_tune.pickle', 'wb') as write_to:
        pickle.dump(models_tune, write_to)

    #---------------------------------------------------
    # If you want to load the pickle files
    #---------------------------------------------------     
#    with open('basemodelstune_dfscore.pickle', 'rb') as read_to:
#        pickle.load(read_to)
#    with open('basemodels_tune.pickle', 'rb') as read_to:
#        pickle.load(read_to)
#    models = models_tune    
