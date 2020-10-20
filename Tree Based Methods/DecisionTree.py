import numpy as np
import pandas as pd
import scipy.stats as st

class Node:
    def __init__(self, feature=-1, split=None, impurity=np.inf):
        # Split feature
        self.feature = feature
        # Split criterion
        self.split = split
        self.impurity = impurity
        self.children = []
        self.leaf = False
        self.label = None
        self.depth = 0
    
    def get_depth(self):
        if self.leaf or len(self.children) == 0:
            return 1
        return 1 + max([child.get_depth() for child in self.children])
        
    def get_split_indices(self, X, intersect_with=None):
        X_feature = X[:, self.feature]

        # Categorical feature
        if isinstance(self.split, np.ndarray):
            splitted_data = []
            for value in self.split:
                indices = np.asarray(X_feature == value).nonzero()[0]
                if intersect_with is not None:
                    indices = np.intersect1d(indices, intersect_with, assume_unique=True)
                splitted_data.append(indices)
                    
            return splitted_data
        
        # Numerical feature
        indices_left = np.asarray(X_feature < self.split).nonzero()[0]
        indices_right = np.asarray(X_feature >= self.split).nonzero()[0]

        if intersect_with is not None:
            indices_left = np.intersect1d(indices_left, intersect_with, assume_unique=True)
            indices_right = np.intersect1d(indices_right, intersect_with, assume_unique=True)

        return [indices_left, indices_right]

    def __get_probs(self, y):
        unique_y = np.unique(y)
        probs = np.zeros(len(unique_y))
        y_len = len(y)
        for i, y_i in enumerate(unique_y):
            probs[i] = len(y[y == y_i]) / y_len
        
        return probs

    def calc_impurity(self, y, criterion, store=False):
        impurity = None

        if criterion == 'entropy':
            probs = self.__get_probs(y)
            impurity = self.__calc_entropy(probs)
        elif criterion == 'gini':
            probs = self.__get_probs(y)
            impurity = self.__calc_gini(probs)
        elif criterion == 'mse':
            impurity = self.__mean_squared_err(y)
        elif criterion == 'mae':
            impurity = self.__mean_absolute_err(y)
        
        if store:
            self.impurity = impurity
        return impurity
            
    def impurity_for_split(self, X, y, weights, criterion):
        splitted_indices = self.get_split_indices(X)  
        impurities = np.zeros(len(splitted_indices))
        if weights is not None:
            total_weight = np.sum(weights)
        else:
            total_weight = len(y)
        for index, branch_indices in enumerate(splitted_indices):
            y_branch = y[branch_indices]
            if weights is not None:
                total_branch_weight = np.sum(weights[branch_indices])
            elif criterion == 'mse' or criterion == 'mae':
                total_branch_weight = 1
            else:
                total_branch_weight = len(y_branch)
            impurities[index] = self.calc_impurity(y_branch, criterion) * total_branch_weight
        return np.sum(impurities) / total_weight
    
    def __calc_entropy(self, probs):
        entropy = -np.sum(probs * np.log(probs + 10e-10))
        return entropy
    
    def __calc_gini(self, probs):
        gini = 1 - np.sum(probs ** 2)
        return gini
    
    def __mean_squared_err(self, y):
        y_m = np.mean(y)
        return np.sum((y - y_m)**2)
    
    def __mean_absolute_err(self, y):
        y_m = np.median(y)
        return np.abs(y - y_m)

    def __str__(self):
        return 'Node(leaf={})'.format(self.leaf)
    
    def __repr__(self):
        return 'Node(leaf={})'.format(self.leaf)


class BaseDecisionTreeEstimator:
    def __init__(self, 
                 tol, 
                 max_depth, 
                 min_members, 
                 criterion, 
                 split_method, 
                 max_features):
        self.tol = tol
        self.max_depth = max_depth
        self.min_members = min_members
        self.criterion = criterion
        self.split_method = split_method
        self.max_features = max_features

    def fit(self, X, y, weights=None):
        self.tree_ = Node()
        X_ = self._get_values(X)
        y_ = self._get_values(y)
        self.weights_ = weights

        if self.split_method == 'binary':
            feature_types = None
        elif self.split_method == 'nary':
            feature_types = [self.__check_type(X_[:, column]) for column in range(X.shape[1])]
        else:
            raise ValueError('parameter split_method must be binary or nary')
        
        self.__generate_tree(self.tree_, X_, y_, weights, feature_types)
        
    def __generate_tree(self, tree, X, y, weights, feature_types):
        if len(y) <= self.min_members:
            self._label_node(tree, y)
            return

        if self.tol and tree.calc_impurity(y, store=True, criterion=self.criterion) < self.tol:
            self._label_node(tree, y)
            return
        
        if self.max_depth and self.tree_.get_depth() >= self.max_depth:
            self._label_node(tree, y)
            return
                
        best_feature_split = self.__split_attribute(tree, X, y, weights, feature_types)        
        tree.feature = best_feature_split[0]
        tree.split = best_feature_split[1]
        
        if tree.feature is None or tree.split is None:
            self._label_node(tree, y)
            return
        
        splitted_data = tree.get_split_indices(X)
        num_branches = len(splitted_data)
        if num_branches < 2:
            self._label_node(tree, y)
            return
        elif num_branches == 2:
            if len(splitted_data[0]) == 0 or len(splitted_data[1]) == 0:
                self._label_node(tree, y)
                return
                
        for branch_indices in splitted_data:
            new_node = Node()
            tree.children.append(new_node)
            branch_weights = weights[branch_indices] if weights is not None else None
            self.__generate_tree(new_node, X[branch_indices], y[branch_indices], branch_weights, feature_types)
        
    
    def __split_attribute(self, tree, X, y, weights, feature_types=None):
        min_impurity = np.inf
        impurity = min_impurity
        best_feature = None
        best_split_value = None

        features = []
        if not self.max_features:
            features = np.arange(X.shape[1])    
        elif self.max_features == 'auto':
            features = np.random.choice(X.shape[1], size=np.sqrt(X.shape[1]).astype('int'), replace=False)
        else:
            features = np.random.choice(X.shape[1], size=self.max_features, replace=False)

        for feature in features:
            tree.feature = feature
            X_feature = X[:, feature]
            if feature_types is not None and feature_types[feature] == 'cat':
                tree.split = np.unique(X_feature)
                if len(tree.split) < 2:
                    continue
                impurity = tree.impurity_for_split(X, y, weights, criterion=self.criterion)
                if impurity < min_impurity:
                    min_impurity = impurity
                    best_feature = feature
                    best_split_value = tree.split
            else:
                X_feature_sorted_indices = np.argsort(X_feature)
                X_feature_sorted = X_feature[X_feature_sorted_indices]
                y_sorted = y[X_feature_sorted_indices]
                thresholds = (X_feature_sorted[1:] + X_feature_sorted[:-1])/2
                thresholds_len = len(thresholds)
                for value_index, value in enumerate(thresholds):
                    if (value_index < thresholds_len - 1) and (y_sorted[value_index] == y_sorted[value_index+1] or thresholds[value_index] == thresholds[value_index+1]):
                        continue
                    
                    tree.split = value
                    impurity = tree.impurity_for_split(X, y, weights, criterion=self.criterion)
                    if impurity < min_impurity:
                        min_impurity = impurity
                        best_feature = feature
                        best_split_value = tree.split
                
        return best_feature, best_split_value, min_impurity
    
    def _label_node(self, node, y):
        pass

    def _decide(self, node, X, pred, indices):
        if node.leaf:
            pred[indices] = node.label
            return
            
        branches = node.get_split_indices(X, indices)
        for index, branch in enumerate(branches):
            self._decide(node.children[index], X, pred, branch)

    def __check_type(self, data):
        try:
            number_data = data.astype(np.number)
            if np.all(np.mod(number_data, 1) == 0):
                return 'cat' if len(np.unique(data)) / len(data) <= 0.05 else 'num'
            return 'num'
        except ValueError:
            return 'cat'
        
    def _get_values(self, data):
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return data.values
        return data

class DecisionTreeClassifier(BaseDecisionTreeEstimator):
    def __init__(self, 
                 tol=None, 
                 max_depth=None, 
                 min_members=10, 
                 criterion='entropy', 
                 split_method='nary', 
                 max_features=None):
        super().__init__(tol, max_depth, min_members, criterion, split_method, max_features)
    
    def _label_node(self, node, y):
        most_frequent = st.mode(y)[0]
        rand = np.random.randint(len(most_frequent))
        node.leaf = True
        node.label = most_frequent[rand]
    
    def predict(self, X):
        X_ = self._get_values(X)
        pred = np.full(X_.shape[0], -1)
        self._decide(self.tree_, X_, pred, np.arange(X_.shape[0]))
        return pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return y_pred[y == y_pred].size / y_pred.size
    
    def __str__(self):
        return "DecisionTreeClassifier(tol={}, max_depth={}, min_members={}, criterion={}, split_method={}, max_features={})".format(self.tol, self.max_depth, self.min_members, self.criterion, self.split_method, self.max_features)
    
    def __repr__(self):
        return "DecisionTreeClassifier(tol={}, max_depth={}, min_members={}, criterion={}, split_method={}, max_features={})".format(self.tol, self.max_depth, self.min_members, self.criterion, self.split_method, self.max_features)


class DecisionTreeRegressor(BaseDecisionTreeEstimator):
    def __init__(self, 
                 tol=None, 
                 max_depth=None, 
                 min_members=10, 
                 criterion='mse', 
                 split_method='nary', 
                 max_features=None):
        super().__init__(tol, max_depth, min_members, criterion, split_method, max_features)
    
    def _label_node(self, node, y):
        node.leaf = True
        if self.criterion == 'mse':
            node.label = np.mean(y)
        elif self.criterion == 'mae':
            node.label = np.median(y)

    def predict(self, X):
        X_ = self._get_values(X)
        pred = np.full(X_.shape[0], 0.0)
        self._decide(self.tree_, X_, pred, np.arange(X_.shape[0]))
        return pred

    def score(self, X, y):
        y_pred = self.predict(X)
        y_m = np.mean(y)
        ss_reg = np.sum((y_pred - y_m)**2)
        ss_tot = np.sum((y - y_m)**2)
        return ss_reg / ss_tot
    
    def __str__(self):
        return "DecisionTreeRegressor(tol={}, max_depth={}, min_members={}, criterion={}, split_method={}, max_features={})".format(self.tol, self.max_depth, self.min_members, self.criterion, self.criterion, self.split_method)
    
    def __repr__(self):
        return "DecisionTreeRegressor(tol={}, max_depth={}, min_members={}, criterion={}, split_method={}, max_features={})".format(self.tol, self.max_depth, self.min_members, self.criterion, self.criterion, self.split_method)
