# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import collections

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

# Other Libraries
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV, cross_val_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import make_scorer, precision_score
from collections import Counter
from sklearn.preprocessing import RobustScaler, LabelEncoder

def ADASYN_balance (X_train, y_train):
    train = X_train.join(y_train)

    print('Before balancing:',train.shape)
    print('\nCounts of non-churn(False) and churn(True) in training data:')
    print(train.churn.value_counts())

    # Oversample minority
    X_train_bal, y_train_bal = ADASYN(sampling_strategy='minority',random_state=0).fit_resample(X_train, y_train)

    X_train_bal = pd.DataFrame(X_train_bal,columns=X_train.columns)
    y_train_bal = pd.DataFrame(y_train_bal,columns=['churn'])
    balanced = X_train_bal.join(y_train_bal)
    
    print('\nAfter balancing:',balanced.shape)
    print('\nCounts of non-churn(False) and churn(True) in training data:')
    print(balanced.churn.value_counts())
        
    return (X_train_bal, y_train_bal, train, balanced)

def df_corr (df, th):
    """
    This function takes in a dataframe and correlation threshold where the 
    first column is the target (dependent variable) and the rest are features to be analyzed
    in a linear regression or multiple linear regression model. 
    We are looking for variables that are highly correlated with the target 
    variable but not collinear.
    Returns a list of positively correlated variables and a list of 
    negatively correlated variables. Also plots the heatmap, pair plots, 
    and prints the results summary.
    """
    import seaborn as sns
    sns.set(rc={'figure.figsize':(15, 15)})
    mask = np.triu(np.ones_like(df.corr(), dtype=np.bool))
    sns.heatmap(df.corr(), mask=mask);
    corrMatrix = df.corr()
    rows, cols = corrMatrix.shape
    flds = list(corrMatrix.columns)
    corr = df.corr().values
    neg_corr = []
    pos_corr = []
    print ("POSITIVE CORRELATIONS:")
    for j in range(1, cols):
        if corr[0,j] > th:
            print ('     ', flds[0], ' ', flds[j], ' ', corr[0,j])
            pos_corr.append(flds[j])
    print ("NEGATIVE CORRELATIONS:")
    for j in range(1, cols):
        if corr[0,j] < -th:
            print ('     ', flds[0], ' ', flds[j], ' ', corr[0,j])
            neg_corr.append(flds[j])
    # Pair Plots
    df_pos = df[pos_corr]
    df_neg = df[neg_corr]
    if pos_corr:
        sns.pairplot(df_pos)
    if neg_corr:
        sns.pairplot(df_neg)
    return pos_corr, neg_corr, df_pos, df_neg




