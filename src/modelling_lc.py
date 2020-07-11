# imports 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report, roc_curve, auc, make_scorer
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE, ADASYN

from src import cm_functions_preprocessing as cmpre

# Classifier Libraries
from sklearn.linear_model import LogisticRegression, RidgeCV, LassoCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier

def run_model(classifier, X, y):
    model = classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 15)
    
    # model params
    print(model.fit(X_train, y_train))
    
    # recall scores:
    print(f"Training recall score: {recall_score(y_train, model.predict(X_train))}")
    print(f"Test recall score: {recall_score(y_test, model.predict(X_test))}")
    
    #Cross val scores for recall:
    print(f"Cross val Score train:  {cross_val_score(model, X_train, y_train, cv=5, scoring='recall')}")
    print(f"Cross val Score test:  {cross_val_score(model, X_test, y_test, cv=5, scoring='recall')}")
    
    # Confusion matrix:
    print(f"Train: {confusion_matrix(y_train, model.predict(X_train))}")
    print(f"Test: {confusion_matrix(y_test, model.predict(X_test))}")
    

def plot_feature_importances(model):
    n_features = model.n_features_
    plt.figure(figsize=(10, 10))
    plt.barh(range(n_features), model.feature_importances_) 
    plt.yticks(np.arange(n_features), X_train.columns.values, fontsize = 12) 
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    plt.title('Model Feature Importance', fontsize = 20)
    
def scale_balance_model(X_train, y_train, model, scaler = StandardScaler(), balance = SMOTE(random_state = 42)):
    # create kfolds object
    kf = KFold(n_splits = 5, random_state = 15)
    
    # create list to add recall scores
    validation_recall = []
    
    for train_ind, val_ind in kf.split(X_train, y_train):
        X_t, y_t = X_train.iloc[train_ind], y_train.iloc[train_ind]
        X_val, y_val = X_train.iloc[val_ind], y_train.iloc[val_ind]
        
        # instantiate and fit/transform scaler
        scaler = scaler
        X_t_sc = scaler.fit_transform(X_t)
        X_val_sc = scaler.transform(X_val)
        
        # instantiate and fit balancing object:
        balance = balance
        X_t_resampled, y_t_resampled = balance.fit_resample(X_t_sc, y_t)
        
        # fit model to X_t_resampled:
        model.fit(X_t_resampled, y_t_resampled)
        
        # append recall score to validation recall list:
        validation_recall.append(recall_score(y_val, model.predict(X_val_sc)))
        
    print(f"Validation recall scores: {validation_recall}")
    print(f"Mean validation recall score:  {np.mean(validation_recall)}")
    print(confusion_matrix(y_val, model.predict(X_val_sc)))
    
    # plot feature importance
#     feature_importance = model.feature_importances_
#     feat_importances = pd.Series(model.feature_importances_, index = X_t_resampled.columns)
#     feat_importances = feat_importances.nlargest(19)
#     feat_importances.plot(kind='barh' , figsize=(10,10))

def compare_models(X_train, y_train):
    # create kfolds object
    kf = KFold(n_splits = 5, shuffle=True, random_state = 42)
    
    for train_ind, val_ind in kf.split(X_train, y_train):

        performance = pd.DataFrame(columns=['Train_Recall','Test_Recall','Test_Specificity'])
        mu_performance = pd.DataFrame(columns=['Train_Recall','Test_Recall','Test_Specificity'])

        recall = make_scorer(recall_score)

        X_t, y_t = X_train.iloc[train_ind], y_train.iloc[train_ind]
        X_val, y_val = X_train.iloc[val_ind], y_train.iloc[val_ind]

        # instantiate and fit/transform scaler
        X_t_sc, X_val_sc = cmpre.ss_scale(X_t, X_val)

        # instantiate and fit SMOTE:
        smote = SMOTE(random_state = 42)
        X_t_sc_bal, y_t_bal = smote.fit_resample(X_t_sc, y_t)

        classifiers = [DecisionTreeClassifier(), KNeighborsClassifier(),
                   RandomForestClassifier(n_estimators = 10), AdaBoostClassifier(),
                   GradientBoostingClassifier()]

        for clf in classifiers:
            train_cv = cross_val_score(X=X_t_sc_bal, y=y_t_bal, 
                                       estimator=clf, scoring=recall,cv=10)

            # Predict
            y_pred = clf.fit(X_t_sc_bal, y_t_bal).predict(X_val_sc)

            conf_matrix = confusion_matrix(y_val,y_pred)

            # Store results
            performance.loc[clf.__class__.__name__+'_default',
                            ['Train_Recall','Test_Recall','Test_Specificity']] = [
                train_cv.mean(),
                recall_score(y_val,y_pred),
                conf_matrix[0,0]/conf_matrix[0,:].sum()
            ]
        mu_performance = pd.concat([mu_performance,performance])
        print(mu_performance, '\n')