import numpy as np
import pandas as pd


class GaussianDiscriminantAnalysis:
    def __init__(self):
        pass

    def initialize_data(self, X_, y_):
        self.n_features_ = X_.shape[1]

        self.classes_ = np.unique(y_)
        self.classes_.sort()

        self.means_ = np.zeros((self.classes_.size, self.n_features_))
        self.priors_ = np.zeros(self.classes_.size)

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def proba_score(self, X, mean, prior):
        pass

    def get_values(self, data):
        if (isinstance(data, pd.DataFrame) or isinstance(data, pd.Series)):
            return data.values
        return data


class LinearDiscriminantAnalysis(GaussianDiscriminantAnalysis):
    def fit(self, X, y):
        X_ = self.get_values(X)
        y_ = self.get_values(y)

        self.initialize_data(X_, y_)

        for index, y_class in enumerate(self.classes_):
            self.means_[index] = np.mean(X_[y_ == y_class], axis=0)
            self.priors_[index] = y_[y_ == y_class].size / y_.size

        self.cov_matrix_ = self.calc_single_covariance(X_, y_)
        self.cov_matrix_det_ = np.sqrt(np.linalg.det(self.cov_matrix_))
        self.cov_matrix_inv = np.linalg.pinv(self.cov_matrix_)

    def predict(self, X):
        X_ = self.get_values(X)
        probs = np.zeros((X_.shape[0], self.priors_.size))
        for index, _ in enumerate(self.classes_):
            probs[:, index] = self.proba_score(
                X_, self.means_[index], self.priors_[index])

        probs_arg_max = np.argmax(probs, axis=1)
        return probs_arg_max

    def proba_score(self, X, mean, prior):
        Xm = X - mean
        Xm_cov = (Xm @ self.cov_matrix_inv) * Xm
        Xm_cov_sum = Xm_cov.sum(axis=1)
        return -0.5*Xm_cov_sum + np.log(prior)

    def calc_single_covariance(self, X, y):
        cov = np.zeros(shape=(X.shape[1], X.shape[1]))
        for i, y_class in enumerate(np.unique(y)):
            X_class_members = X[y == y_class]
            cov += (X_class_members -
                    self.means_[i]).T @ (X_class_members - self.means_[i])

        cov /= X.shape[0]
        return cov


class QuadraticDiscriminantAnalysis(GaussianDiscriminantAnalysis):
    def fit(self, X, y):
        X_ = self.get_values(X)
        y_ = self.get_values(y)

        self.initialize_data(X_, y_)

        self.cov_matrices_ = np.zeros(
            (self.classes_.size, self.n_features_, self.n_features_))
        for index, y_class in enumerate(self.classes_):
            has_y_class = (y_ == y_class)
            self.means_[index] = np.mean(X_[has_y_class], axis=0)
            self.priors_[index] = y_[has_y_class].size / y_.size
            self.cov_matrices_[index] = self.calc_class_covariance(
                X_[has_y_class], self.means_[index])

    def predict(self, X):
        X_ = self.get_values(X)
        probs = np.zeros((X_.shape[0], self.priors_.size))
        for index, _ in enumerate(self.classes_):
            probs[:, index] = self.proba_score(X_, self.means_[index], self.priors_[
                                               index], self.cov_matrices_[index])
        probs_arg_max = np.argmax(probs, axis=1)
        return probs_arg_max

    def proba_score(self, X, mean, prior, cov_matrix):
        cov_matrix_det = np.linalg.det(cov_matrix)
        cov_matrix_inv = np.linalg.pinv(cov_matrix)

        Xm = X - mean
        Xm_cov = (Xm @ cov_matrix_inv) * Xm
        Xm_cov_sum = Xm_cov.sum(axis=1)
        return -0.5*Xm_cov_sum - 0.5*np.log(cov_matrix_det) + np.log(prior)

    def calc_class_covariance(self, X_members_i, mean_i):
        if (X_members_i.shape[0] == 0):
            return np.zeros((X_members_i.shape[1], X_members_i.shape[1]))
        return (X_members_i - mean_i).T @ (X_members_i - mean_i) / X_members_i.shape[0]