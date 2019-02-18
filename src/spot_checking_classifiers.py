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
from functions import get_cross_val_score, plot_metric_curves


if __name__ == "__main__":

    dict_scores = {}
    models = []
    data_val = []
    model_names = []
    
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
        
    # Apply Standard Scaler to the training and validation sets
    std = StandardScaler()
    std.fit(X_train.values)    
    X_train_scaled = std.transform(X_train.values)
    #Apply the scaler to the val and test set
    X_val_scaled = std.transform(X_val.values)
    X_test_scaled1 = std.transform(X_test.values)    

    
    # Model 1: Logistic Regression    
    print("Logistic Regression ...")
    # Create a model instance 
    lr_model = LogisticRegression(random_state=41)
    # Get metrics or score through cross validation
    scores = get_cross_val_score(lr_model, X_train_val, y_train_val)
    # Save the score in a dictionaty
    dict_scores['Logistic Regression']= scores
    # Fit the model with the scaled training data
    lr_model.fit(X_train_scaled,y_train)
    # Save the model object in a list
    models.append(lr_model)
    # Save the model's name in a list
    model_names.append('Logistic')
    # Save the validation data in a list
    data_val.append([X_val_scaled,y_val])
    
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


    print("linearSVC ...")
    # Create a model instance     
    lsvc_model = LinearSVC(random_state=41)
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

    ## Ignore this classifier for now
#    print("SVM-rbf...")
#    # Create a model instance       
#    svm_rbf_model = SVC(kernel='rbf', gamma='scale', random_state=41)
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
    Gnb_model = GaussianNB()
    # Get metrics or score through cross validation
    scores = get_cross_val_score(Gnb_model, X, y)  
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
    # Create a model instance     
    dtree_model = DecisionTreeClassifier(random_state=41)
    # Get metrics or score through cross validation   
    scores = get_cross_val_score(dtree_model,X_train_val, y_train_val)
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
    # Create a model instance     
    bag_dtree_model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), 
                                        random_state=41)
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
    # Create a model instance     
    rtree_model = RandomForestClassifier(n_estimators=100, max_depth=2,
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
    
    # Save the validation data in a list    
    data_val.append([X_val_scaled,y_val])

    # Print the scores in dataframe
    df_score = pd.DataFrame.from_dict(dict_scores,orient='index', 
        columns=['F1_macro','Precision_macro','Recall_macro','F1_weight', 'Confusion Matrix'])    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_score)
        
    # Plot the ROC and Precision-Recall curves. plot_metric_curves is a customized function
    plot_metric_curves(models, data_val, model_names, plot_name='basemodels', Xtrain=None, ytrain=None)

    #---------------------------------------------------
    # Pickle the scores and list of model objects
    #---------------------------------------------------      
    with open('basemodels_dfscore.pickle', 'wb') as write_to:
        pickle.dump(df_score, write_to)
        
    base_models = models
    with open('basemodels.pickle', 'wb') as write_to:
        pickle.dump(base_models, write_to)

    #---------------------------------------------------
    # If you want to load the pickle files
    #---------------------------------------------------         
#    with open('basemodels_dfscore.pickle', 'rb') as read_to:
#        pickle.load(read_to)
    
#    with open('basemodels.pickle', 'rb') as read_to:
#        pickle.load(read_to)
#    models = base_models
   