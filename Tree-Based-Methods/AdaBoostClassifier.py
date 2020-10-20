import numpy as np
import pandas as pd
from DecisionTree import DecisionTreeClassifier

class AdaBoostClassifier:
    def __init__(self, 
                 n_learners=10, 
                 tol=0.1, 
                 max_depth=2, 
                 min_members=10, 
                 criterion='entropy', 
                 split_method='nary', 
                 max_features=None):
        self.n_learners = n_learners
        self.tol = tol
        self.max_depth = max_depth
        self.min_members = min_members
        self.criterion = criterion
        self.split_method = split_method
        self.max_features = max_features

    def fit(self, X, y):
        self.classifiers_ = []
        self.alphas_ = np.zeros(self.n_learners)
        self.n_outputs_ = np.unique(y)
        X_ = self.__get_values(X)
        y_ = self.__get_values(y)
        weights = np.full(len(X_), 1/len(X_))
        for i in range(self.n_learners):
            model = DecisionTreeClassifier(
                self.tol, 
                self.max_depth, 
                self.min_members, 
                self.criterion, 
                self.split_method, 
                self.max_features
            )
            model.fit(X_, y_, weights)
            self.classifiers_.append(model)
            y_pred = model.predict(X_)
            wrong_pred = y_ != y_pred
            weighted_error = np.sum(weights[wrong_pred]) / np.sum(weights)
            alpha = np.log((1-weighted_error)/weighted_error + 10e-8)
            weights[wrong_pred] *= np.exp(alpha)
            weights /= np.sum(weights)
            self.alphas_[i] = alpha

    def predict(self, X):
        all_predictions = np.zeros((X.shape[0], self.n_learners))
        all_linear_combs = np.zeros((X.shape[0], len(self.n_outputs_)))

        for index, classifier in enumerate(self.classifiers_):
            all_predictions[:, index] = classifier.predict(X)

        for i, y_i in enumerate(self.n_outputs_):
            is_y_i = np.asarray(all_predictions == y_i).astype('int')
            is_y_i_sum = is_y_i @ self.alphas_
            all_linear_combs[:, i] = is_y_i_sum.reshape(-1)
        
        pred_indices = np.argmax(all_linear_combs, axis=1)
        pred = np.zeros(pred_indices.shape)
        for i, y_i in enumerate(self.n_outputs_):
            pred[pred_indices == i] = y_i

        return pred
    
    def score(self, X, y):
        pred = self.predict(X)
        return pred[y == pred].size / pred.size
        
    def __get_values(self, data):
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return data.values
        return data