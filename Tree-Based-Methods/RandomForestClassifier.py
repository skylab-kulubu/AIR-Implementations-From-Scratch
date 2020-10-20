import numpy as np
import pandas as pd
import scipy.stats as st
from DecisionTree import DecisionTreeClassifier

class RandomForestClassifier:
    def __init__(self, 
                 n_trees=100, 
                 tol=0.1, 
                 max_depth=None, 
                 min_members=10, 
                 criterion='entropy', 
                 split_method='nary', 
                 max_features=None):
        self.n_trees = n_trees
        self.tol = tol
        self.max_depth = max_depth
        self.min_members = min_members
        self.criterion = criterion
        self.split_method = split_method
        self.max_features = max_features

    def fit(self, X, y):
        self.classifiers_ = []
        X_ = self.__get_values(X)
        y_ = self.__get_values(y)
        for _ in range(self.n_trees):
            sample = self.__get_sample(X.shape[0])
            model = DecisionTreeClassifier(
                self.tol, 
                self.max_depth, 
                self.min_members, 
                self.criterion, 
                self.split_method, 
                self.max_features
            )
            model.fit(X_[sample], y_[sample])
            self.classifiers_.append(model)

    def predict(self, X):
        all_predictions = np.zeros((self.n_trees, X.shape[0]))
        for index, classifier in enumerate(self.classifiers_):
            all_predictions[index] = classifier.predict(X)
        
        majority_predictions = st.mode(all_predictions, axis=0)[0][0]
        return majority_predictions
    
    def score(self, X, y):
        pred = self.predict(X)
        return pred[y == pred].size / pred.size

    def __get_sample(self, sample_size):
        return np.random.choice(sample_size, size=sample_size)
    
    def __get_values(self, data):
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return data.values
        return data