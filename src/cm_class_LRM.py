from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, recall_score
from sklearn.metrics import mean_squared_error, auc, average_precision_score
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

class LogRegModel(): 
    """
    A class to represent a Logistic Regression Model
    class cm_class_LRM.LogRegModel(X, y, **test_size=None, **random_state=None)
    Uses scikit-learn: https://scikit-learn.org/stable/index.html
    ...

    Attributes
    ----------
    X : pandas.core.frame.DataFrame
        Features
    y : pandas.core.series.Series
        Target variable
    **random_state : RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by `np.random`.
    **test_size : float, int or None, optional (default=None)
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.25.

    Methods
    -------
    scale_data():
        Scales features
        
    LRM_tts(self):
        Performs train_test_split
        
    def LRM_base_model(self):
        Creates initial model
    
    regularization_tune():
    """
    
    def __init__(self, X, y, **kwargs): 
        self.X = X 
        self.y = y 
        self.random_state = kwargs.get('random_state', None)
        self.test_size = kwargs.get('test_size', None)
#         self.y_train = None
#         self.y_test = None
#         self.X_train_sc, self.X_test_sc, self.y_train, self.y_test = None
#         self.coef_ = None
#         self.intercept_ = None
#         self.get_params = None
#         self.y_hat_train = None
#         self.cm_train = None
#         self.y_train_score = None
#         self.cv_train_accuracy = None
#         self.AP_train_score = None
#         self.train_fpr, self.train_tpr, self.train_thresholds = None
        return

    def LRM_tts(self): 
        """
        Splits the data into train and test, scales features using LRM_ss_scale method.
        Uses scikit-learn: https://scikit-learn.org/stable/index.html
        ...


        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size = self.test_size, 
            random_state=self.random_state)
        
        X_train_sc, X_test_sc = self.LRM_ss_scale(X_train, X_test)
        
        return X_train_sc, X_test_sc, y_train, y_test
      
    def LRM_ss_scale(self, X_train, X_test): 
        """
        Feature scaling using sklearn StandardScaler
        note: StandardScaler cannot guarantee balanced feature scales in the presence of outliers.
        Uses scikit-learn: https://scikit-learn.org/stable/index.html
        ...

        Parameters
        ----------
        pandas DataFrame or Series
            Train and Test feature datasets

        Returns
        -------
        pandas DataFrame or Series
            Scaled train and test features

        """
        ss = StandardScaler()

        X_train_scaled = ss.fit_transform(X_train)
        X_test_scaled = ss.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    def LRM_model(self):
        """
        Builds the logistic regression model and checks the performance of the model on the training data. 
        Assigns model attributes (model parameters, training data scores) to the class instance of LogRegModel.
        Uses scikit-learn: https://scikit-learn.org/stable/index.html
        ...
        

        """
        # Train/test split and scale
        X_train_sc, X_test_sc, y_train, y_test = self.LRM_tts()
        
        # Run LogisticRegression and fit model to training data
        logreg = LogisticRegression(random_state=self.random_state) 
        logreg.fit(X_train_sc, y_train)
        
        # Get model coefficients and parameters
        coef_ = logreg.coef_
        intercept_ = logreg.intercept_
        get_params = logreg.get_params
        
        # Predict outcomes for training data
        y_hat_train = logreg.predict(X_train_sc) 
        
        # Evaluate the model
        cm_train = confusion_matrix(np.array(y_train), np.array(y_hat_train))
        
        # Probability scores of the training set data
        y_train_score = logreg.decision_function(X_train_sc)
        
        # Calculate the cross validated training accuracy of the model
        cv_train_accuracy = cross_val_score(logreg, X_train_sc, y_train, cv=10, scoring='accuracy').mean()
        
        # Calculate the precision-recall score
        AP_train_score = average_precision_score(y_train, y_train_score)
        
        # Calculate the fpr, tpr, and thresholds for the training set
        train_fpr, train_tpr, train_thresholds = roc_curve(y_train, y_train_score)
        
        # Print summary of results
        names = ['Confusion Matrix', 'Cross Validation Score', 'Precision Recall Score']
        results = [cm_train, cv_train_accuracy, AP_train_score]
        
        print("TRAINING DATA RESULTS \n")
        [print('{}\n{} \n ----------------------'.format(n, r)) for n,r in zip(names,results)]
        
        self.LRM_regularization_tune(X_train_sc, y_train) 
        return results, coef_, intercept_, get_params
        
    
    
    
    
    
###_________________________WORK IN PROGRESS______________________________###    
    
    

    def LRM_regularization_tune(self, X_train_sc, y_train):
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
            logreg = LogisticRegression(C=c, random_state=self.random_state)
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
    
    
        
    #def LRM_print(self, ):
    
    
    
# OUTLIERS
#remove outliers:
# z = np.abs(stats.zscore(sales_and_res.SalePrice))
# sales_and_res = sales_and_res[z < 3]
        

# OPTIMIZATION    
# solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}, default='lbfgs'

#     Algorithm to use in the optimization problem.

#     - For small datasets, 'liblinear' is a good choice, whereas 'sag' and
#       'saga' are faster for large ones.
#     - For multiclass problems, only 'newton-cg', 'sag', 'saga' and 'lbfgs'
#       handle multinomial loss; 'liblinear' is limited to one-versus-rest
#       schemes.
#     - 'newton-cg', 'lbfgs', 'sag' and 'saga' handle L2 or no penalty
#     - 'liblinear' and 'saga' also handle L1 penalty
#     - 'saga' also supports 'elasticnet' penalty
#     - 'liblinear' does not support setting ``penalty='none'``

#     Note that 'sag' and 'saga' fast convergence is only guaranteed on
#     features with approximately the same scale. You can
#     preprocess the data with a scaler from sklearn.preprocessing.