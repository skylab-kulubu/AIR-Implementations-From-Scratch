import numpy as np
import pandas as pd
import scipy.stats as st
from DecisionTree import DecisionTreeRegressor

class GradientBoostingClassifier:
    def __init__(self, 
                 n_learners=100,
                 n_iters_stop=None,
                 loss_tol=10e-4,
                 alpha=1.0, 
                 tol=None, 
                 max_depth=None, 
                 min_members=10, 
                 split_method='nary', 
                 max_features=None):
        self.n_learners = n_learners
        self.n_iters_stop = n_iters_stop
        self.loss_tol = loss_tol
        self.alpha = alpha
        self.tol = tol
        self.max_depth = max_depth
        self.min_members = min_members
        self.split_method = split_method
        self.max_features = max_features

    def fit(self, X, y):
        X_ = self.__get_values(X)
        y_ = self.__get_values(y)
        self.regressors_ = []
        self.unique_outputs_ = np.unique(y_)
        self.n_outputs_ = len(self.unique_outputs_)
        estimations = np.zeros((self.n_outputs_, len(y_)))
        self.loss_ = np.inf
        stable_loss_count = 0
        for _ in range(self.n_learners):
            prob_exp = np.exp(estimations)
            probs = prob_exp / (np.sum(prob_exp, axis=0))
            curr_regressor_group = []
            if self.n_iters_stop:
                curr_loss = self._loss(y_, probs)
                if abs(curr_loss - self.loss_) <= self.loss_tol:
                    stable_loss_count += 1
                    if stable_loss_count == self.n_iters_stop:
                        self.loss_ = curr_loss
                        break 
                else:
                    stable_loss_count = 0

                self.loss_ = curr_loss

            for k, y_k in enumerate(self.unique_outputs_):
                model = DecisionTreeRegressor(
                    tol=self.tol, 
                    max_depth=self.max_depth, 
                    min_members=self.min_members, 
                    criterion='mse', 
                    split_method=self.split_method, 
                    max_features=self.max_features
                )

                y_i = y_ == y_k
                residuals = y_i - probs[k]
                model.fit(X_, residuals)

                res_pred = self.__eval_leaves(model.tree_, X_, residuals)
                estimations[k] += res_pred * self.alpha
                curr_regressor_group.append(model)
            self.regressors_.append(curr_regressor_group)
    
    def __eval_leaves(self, tree, X, residuals):
       pred = np.full(residuals.shape, 0.0)
       self._eval_leave(tree, X, residuals, pred, np.arange(residuals.shape[0]))
       return pred

    def _eval_leave(self, node, X, residuals, pred, indices):
        if node.leaf:
            res_leaf = residuals[indices]
            res_abs = np.abs(res_leaf)
            res_update = np.sum(res_leaf) / np.sum(res_abs * (1-res_abs))
            node.label = res_update * (self.n_outputs_-1) / self.n_outputs_
            pred[indices] = node.label
            return
            
        branches = node.get_split_indices(X, indices)
        for index, branch in enumerate(branches):   
            self._eval_leave(node.children[index], X, residuals, pred, branch)
    
    def _loss(self, y, probs):
        losses = np.zeros((self.n_outputs_, y.shape[0]))
        for k, y_k in enumerate(self.unique_outputs_):
            y_i = y == y_k
            losses[k] = y_i * probs[k]
        return -np.sum(losses)

    def predict(self, X):
        estimations = np.zeros((self.n_outputs_, len(X)))
        for _, regressor_group in enumerate(self.regressors_):
            for index_k, regressor in enumerate(regressor_group):
                res_pred = regressor.predict(X)
                estimations[index_k] += res_pred * self.alpha
        
        prob_exp = np.exp(estimations)
        probs = prob_exp / (np.sum(prob_exp, axis=0))
        predict_indices = probs.argmax(axis=0)
        pred = np.zeros(X.shape[0])
        for i, y_i in enumerate(self.unique_outputs_):
            pred[predict_indices == i] = y_i
        
        return pred
    
    def score(self, X, y):
        pred = self.predict(X)
        return pred[y == pred].size / pred.size
    
    def __get_values(self, data):
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return data.values
        return data
    