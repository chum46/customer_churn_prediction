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
from sklearn.linear_model import LogisticRegression, RidgeCV, LassoCV
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


def LRM_regularization_tune(X_train_sc, y_train):
    """
    {In progress}
    ...

    Parameters
    ----------

    Returns
    -------

    Example
    --------

    """
    C_param_range = [0.001, 0.01, 0.1, 1, 10, 100]
    names = [0.001, 0.01, 0.1, 1, 10, 100]
    colors = sns.color_palette('Set2')

    plt.figure(figsize=(10, 8))

    for n, c in enumerate(C_param_range):
        # Fit a model
        logreg = LogisticRegression(C=c, solver='liblinear')
        model_log = logreg.fit(X_train_sc, y_train)
        print(model_log) # Preview model params

        # Predict
        y_hat_train = logreg.predict(X_train_sc)

        y_train_score = model_log.decision_function(X_train_sc)

        fpr, tpr, thresholds = roc_curve(y_train, y_train_score)

        print('AUC for {}: {}'.format(names[n], auc(fpr, tpr)))
        print('-------------------------------------------------------')
        lw = 2
        plt.plot(fpr, tpr, color=colors[n],
                 lw=lw, label='ROC curve Normalization Weight: {}'.format(names[n]))

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show();
    return 
    

def crossValidate(): 
    """
     
    
    ...

    Attributes
    ----------


    Methods
    -------
    info(additional=""):
        Prints the person's name and age.
        
    """
    return

def recall_optim(y_true, y_pred):
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Recall will be worth a greater value than specificity
    rec = recall_score(y_true, y_pred) * 0.8 
    spe = conf_matrix[0,0]/conf_matrix[0,:].sum() * 0.2 
    
    # Imperfect recalls will lose a penalty. This means the best results 
    # will have perfect recalls and compete for specificity
    if rec < 0.8:
        rec -= 0.2
    return rec + spe 

def hp_tuning(clf, params, X_train, X_test, y_train, y_test):
    performance = pd.DataFrame(columns=['Train_Recall','Test_Recall','Test_Specificity'])
    
    # Load GridSearchCV
    search = GridSearchCV(
        estimator=clf,
        param_grid=params,
        n_jobs=-1,
        scoring='recall'
    )

    # Train search object
    search.fit(X_train, y_train)

    # Heading
    print('\n','-'*40,'\n',clf.__class__.__name__,'\n','-'*40)

    # Extract best estimator
    best = search.best_estimator_
    print('Best parameters: \n\n',search.best_params_,'\n')

    # Cross-validate on the train data
    print("TRAIN GROUP")
    train_cv = cross_val_score(X=X_train, y=y_train, 
                               estimator=best, scoring='recall',cv=10)
    print("\nCross-validation recall scores:",train_cv)
    print("Mean recall score:",train_cv.mean())

    # Now predict on the test group
    print("\nTEST GROUP")
    y_pred = best.fit(X_train, y_train).predict(X_test)
    print("\nRecall:",recall_score(y_test,y_pred))

    # Get classification report
    print(classification_report(y_test, y_pred))

    # Print confusion matrix
    fig = plt.figure(figsize = (10,7))
    conf_matrix = confusion_matrix(y_test,y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d')
    plt.show()

    # Store results
    performance.loc[clf.__class__.__name__+'_optimize',:] = [
        train_cv.mean(),
        recall_score(y_test,y_pred),
        conf_matrix[0,0]/conf_matrix[0,:].sum()
    ]
    
    # Look at the parameters for the top best scores
    display(pd.DataFrame(search.cv_results_).iloc[:,4:].sort_values(by='rank_test_score').head())
    display(performance)
    return performance, search.cv_results_
        