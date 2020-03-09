import time
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score

from sklearn.ensemble import RandomForestClassifier

    

def score_classifier(dataset,classifier,labels):

    """
    performs 3 random trainings/tests to build a confusion matrix and prints results with precision and recall scores
    :param dataset: the dataset to work on
    :param classifier: the classifier to use
    :param labels: the labels used for training and validation
    :return:
    """

    kf = KFold(n_splits=3,random_state=50,shuffle=True)
    confusion_mat = np.zeros((2,2))
    recall = 0
    for training_ids,test_ids in kf.split(dataset):
        training_set = dataset[training_ids]
        training_labels = labels[training_ids]
        test_set = dataset[test_ids]
        test_labels = labels[test_ids]
        classifier.fit(training_set,training_labels)
        predicted_labels = classifier.predict(test_set)
        confusion_mat+=confusion_matrix(test_labels,predicted_labels)
        recall += recall_score(test_labels, predicted_labels)
    recall/=3
    print(confusion_mat)
    print(recall)
    return recall


def correlation_heatmap(df1):
    """
    Print the Pearson correlation table between each features.

    Parameters
    ----------
    df1 : Pandas DataFrame

    Returns
    -------
    None.

    """
    
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    _ = sns.heatmap(df1.corr(), 
                    cmap = colormap,
                    square=True, 
                    cbar_kws={'shrink':.9 }, 
                    ax=ax,
                    annot=True, 
                    linewidths=0.1,vmax=1.0, linecolor='white',
                    annot_kws={'fontsize':12 })
    plt.title('Pearson Correlation of Features', y=1.05, size=15)


def seachForestClassifier(dataset, labels, n_jobs=8):
    """
    Search the best hyper parameters for the RandomForestClassifier.

    Parameters
    ----------
    dataset : The dataset to work on
    labels : The labels used for training and validation

    Returns
    -------
    The best RandomForestClassifier found.

    """
    MOD = RandomForestClassifier(random_state=0) 
    param_distributions = {"n_estimators" : np.linspace(2, 500, 500, dtype = "int"),  
                           "max_depth": [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, None], 
                           "min_samples_split": np.linspace(2, 50, 50, dtype = "int"),  
                           "max_features": ["sqrt", "log2",10, 20, None],
                           "random_state": [50]}
    random_search = RandomizedSearchCV(MOD,
                                       param_distributions = param_distributions, 
                                       n_iter = 20,
                                       scoring = "recall",
                                       return_train_score = True,
                                       random_state = 50,
                                       cv = 3,
                                       n_jobs=n_jobs)
    
    #trains and optimizes the model
    random_search.fit(dataset, labels)
    
    #recover the best model
    return random_search.best_estimator_



class transformData:
    """
    Pipeline for data transformations
    
    """
    
    def __init__(self, df):
        self.df = df
        
        # Extract feature names
        paramset = df.columns.values
        
        # Categorize numerical values
        self.allIndex = []        
        for col in paramset:
            (self.df, index) = self.changeColumn(self.df, col)
            self.allIndex += [index]
            
        self.scaler = MinMaxScaler()
            

    def changeColumn(self, df1, col):
        """
        Group col datas by intervals and create a new feature containing the label of theses intervals.
    
        Parameters
        ----------
        df1 : Pandas DataFrame
        col : Name of a df1 feature (string)
    
        Returns
        -------
        Pandas DataFram with a Categorical column instead of col.
    
        """
        df1[col][np.isnan(df1[col])] = 0.0
        df1[col] = df1[col].astype(int)
            
        df1['Categorical_'+col] = pd.cut(df1[col], 5, labels=[i for i in range(5)])
        
        # store the index for the prediction step
        index = df1[col].value_counts(bins=5, sort=False).index
        
        return df1.drop([col],axis=1), index
    
    
    def transform(self, X_test):
        """
        Reproduce the same feature operations.

        Returns
        -------
        Normalized and transformized data.

        """
        X_trf = np.zeros(len(X_test))
        for i, value in enumerate(X_test):
            for j, index in enumerate(self.allIndex[i]):
                if value in index:
                    X_trf[i] = j        
        X = self.scaler.transform(X_trf)
        return X
        
    
    def fit(self):
        """
        Normalize the dataset.

        Returns
        -------
        X : dataset normalized

        """
        X = self.scaler.fit_transform(self.df.values)
        return X




class newModel:
    """
    main class call by the api or the app
    
    """
    
    def __init__(self):
        # Load dataset
        df = pd.read_csv(".\\nba_logreg.csv")
        
        # Extract duplicated rows with both prediction
        # That decreases recall but increase accuracy
        #paramset = df.drop(['TARGET_5Yrs','Name'],axis=1).columns.values
        #df = df[df[paramset].duplicated(keep=False).apply(lambda row : not(row))]
        
        # extract names, labels
        self.labels = df['TARGET_5Yrs'].values # labels
    

        # Plot the Pearson correlation table between features
        #correlation_heatmap(df)
        
        # We drop labels, name.
        # But the correlation_heatmap function shows that some features are similars.
        # It is true that FGM, FGA et FG% are linked : FG% = 100*FGM/FGA
        # We drop FGA, and same idea for 3PA, FTA and REB.
        # Finally correlation_heatmap function shows that FGM is similar to PTS
        rm_features = ['TARGET_5Yrs','Name'] + ['FGA','3PA', 'FTA', 'REB'] + ['FGM']
        df = df.drop(rm_features,axis=1)
        self.paramset = df.columns.values # features

    
        # Categorize numerical values
        trf = transformData(df)
        
        # Normalize the dataset
        self.X = trf.fit()
    
        # Search the best RandomForestClassifier
        #rfc = seachForestClassifier(X, labels)
        #score_classifier(X,rfc,labels)
        
        # Using the model given by the function seachForestClassifier, we can
        # maximize the recall score returned by the score_classifier function
        self.rfc = RandomForestClassifier(n_estimators=245,
                                          max_depth=1,
                                          min_samples_split=23,
                                          max_features="log2",
                                          random_state=50)
        
        
    def trainModel(self):
        """
        Allow to train the model by using the score_classifier function.

        Returns
        -------
        score : Recall score.
        train_time : Duration of the training.

        """
        t0 = time.time()
        score = score_classifier(self.X, self.rfc, self.labels)
        train_time = time.time() - t0
        
        return score, train_time
    
    
    def predictModel(self, X_test):
        """
        Allow to predict a result using our trained model.

        Parameters
        ----------
        X_test : TYPE
            DESCRIPTION.

        Returns
        -------
        pred : TARGET_5Yrs prediction.
        pred_time : Duration of the prediction.

        """
        t0 = time.time()
        pred = self.rfc.predict(X_test)
        pred_time = time.time() - t0
        
        return pred, pred_time



if __name__ == '__main__':
    
    mod = newModel()
    mod.trainModel()