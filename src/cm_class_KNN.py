from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, recall_score, f1_score, precision_score, roc_auc_score
from sklearn.metrics import mean_squared_error, auc, average_precision_score, accuracy_score
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

class knnModel(): 
    """
    A class to represent a knn Model
    class cm_class_KVV.kvvModel(X, y, **test_size=None, **random_state=None)
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
    KNN_tts()
    KNN_ss_scale(X_train, X_test, X_t, X_val)
    KNN_train()

    """
    
    def __init__(self, X, y, **kwargs): 
        self.X = X 
        self.y = y 
        self.k = None
        self.random_state = kwargs.get('random_state', None)
        self.test_size = kwargs.get('test_size', None)
        return

    def KNN_tts (self): 
        """
        Splits the data into train and test, scales features using LRM_ss_scale method.
        Uses scikit-learn: https://scikit-learn.org/stable/index.html
        ...


        """
        # Initial train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size = self.test_size, 
            random_state=self.random_state)
        
        # Secondary train/test split for KNN_train
        # X_trn2, X_tst2, y_trn2, y_tst2 = train_test_split(
        #    self.X_train, self.y_train, random_state=42, test_size = .25)
        
        # Scale data
        self.X_train_sc, self.X_test_sc = self.KNN_ss_scale()

        return self.X_train, self.y_train
    
    def KNN_ss_scale(self): 
        """
        Feature scaling using sklearn StandardScaler
        note: StandardScaler cannot guarantee balanced feature scales in the presence of outliers.
        Uses scikit-learn: https://scikit-learn.org/stable/index.html
        ...

        Parameters
        ----------
        pandas DataFrame or Series, Training subset of train/test split

        Returns
        -------
        pandas DataFrame or Series, Scaled train and test features

        """
        ss = StandardScaler()
        
        #X_ind = X_t.index
        #X_col = X_t.columns

        #X_t_s = pd.DataFrame(ss.fit_transform(X_t))
        #X_t_s.index = X_ind
        #X_t_s.columns = X_col

        #X_v_ind = X_val.index
        #X_val_s = pd.DataFrame(ss.transform(X_val))
        #X_val_s.index = X_v_ind
        #X_val_s.columns = X_col

        X_train_scaled = ss.fit_transform(self.X_train)
        X_test_scaled = ss.transform(self.X_test)
        
        return X_train_scaled, X_test_scaled
    
    def KNN_train (self):
        
        # Train-Test Split
        X_train, y_train = self.KNN_tts()
        
        # KFold
        kf = KFold(n_splits=5, random_state=self.random_state, shuffle=True)

        K = [] 
        training = [] 
        test = [] 
        k_scores_train = {}
        k_scores_val = {}

        for k in range(2,21):
            
            knn = KNeighborsClassifier(n_neighbors=k)
            accuracy_score_t = []
            accuracy_score_v = []
            
            for train_ind, val_ind in kf.split(X_train, y_train):

                X_t, y_t = X_train.iloc[train_ind], y_train.iloc[train_ind] 
                X_v, y_v = X_train.iloc[val_ind], y_train.iloc[val_ind]
                mm = MinMaxScaler()

                X_t_ind = X_t.index
                X_v_ind = X_v.index

                X_t_s = pd.DataFrame(mm.fit_transform(X_t))
                X_t_s.index = X_t_ind
                X_v_s = pd.DataFrame(mm.transform(X_v))
                X_v_s.index = X_v_ind

                knn.fit(X_t_s, y_t)

                y_pred_t = knn.predict(X_t_s)
                y_pred_v = knn.predict(X_v_s)

                accuracy_score_t.append(accuracy_score(y_t, y_pred_t))
                accuracy_score_v.append(accuracy_score(y_v, y_pred_v))

            K.append(k)
            training.append(np.mean(accuracy_score_t)) 
            test.append(np.mean(accuracy_score_v)) 
            k_scores_train[k] = np.mean(accuracy_score_t)
            k_scores_val[k] = np.mean(accuracy_score_v)
        
        # Plot the results
        ax = sns.stripplot(K, training, color=".3")
        ax = sns.stripplot(K, test, color="red")
        ax.set(xlabel ='k values', ylabel ='Accuracy') 
        plt.show()
        
        return k_scores_train, k_scores_val
    
    def KNN_predict (self, **kwargs):
        self.k = kwargs.get('k', None)
        
        clf = KNeighborsClassifier(n_neighbors = self.k) 
        clf.fit(self.X_train, self.y_train) 
        
        print(f"training accuracy: {clf.score(self.X_t_s, self.y_t)}")
        print(f"Val accuracy: {clf.score(self.X_val_s, self.y_val)}")

        y_hat = knn.predict(self.X_val_s)
        
        # plot_confusion_matrix(confusion_matrix(self.y_val, self.y_hat), classes=['No Churn', 'Churn'])       
        return clf.__dict__
    
    def KNN_tune_gridsearch (self):
        knn = KNeighborsClassifier()
        
        #Split data
        X_t, X_val, y_t, y_val = train_test_split(self.X_train, self.y_train, random_state=self.random_state)
        
        #Scale data
        mm = MinMaxScaler()
        X_t_s = mm.fit_transform(X_t)
        X_val_s = mm.transform(X_val)
        
        #Training the model
        knn.fit(X_t_s, y_t)
        
        #List Hyperparameters to tune
        hyperparameters = None
        leaf_size = list(range(1,50))
        n_neighbors = list(range(2,21))
        p=[1,2]
        hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

        #Making model
        clf = GridSearchCV(knn, hyperparameters, cv=5)
        best_model = clf.fit(X_t_s, y_t)
        
        #Best Hyperparameters Value
        print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
        print('Best p:', best_model.best_estimator_.get_params()['p'])
        print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
        
        #Predict testing set
        y_pred = best_model.predict(X_val_s)
        
        #Check performance using accuracy
        print('Accuracy: ', accuracy_score(y_val, y_pred))
        
        #Check performance using ROC
        print('ROC AUC Score: ', roc_auc_score(y_val, y_pred))
        
        self.get_params = best_model.best_estimator_.get_params()
        
        return self.get_params
        
        
        
        
        
        
        
        
        
