#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 21:44:41 2020

@author: safak
"""

#%%
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
import numpy as np

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
            self._verts3d = xs, ys, zs

    def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
            FancyArrowPatch.draw(self, renderer)



class PCA(object):
    
    """
    This algorithm will be more flrexible.
    Just for now, the algorithm works for the data that has three features.
    But you can change the number below algorithm and do it flexiblee
    """
    
    """
    Let self.all_data be our feature vector's transpose.
    
    We have to find its covariance matrix to do PCA.
    When we find the Covariance matrix, the next step is find eigenvalues and its eigenvectors
    The eigenvalues of covariance matrix (Σ) are roots of the characteristic equation:
        
                        det(Σ - λ I) = 0
    
    Then, sort the eigenvectors by decreasing eigenvalues and choose k eigenvectors with
    the largest eigenvalues to form a nxk dimensional matrix w, where n is shape of feature vector.
    
    We started with goal of to reduce the dimensionality of our feature space, projecting the feature
    space via PCA onto a smaller subspace, where the eigenvectors will from the axes of this new feature
    subspace. However, the eigenvectors only define the directions of the new axis, since they have all 
    the same unit length 1.
    Roughly speaking, the eigenvectors with the lowest eigenvalues bear the least information about the
    distribution of the data, and those are we want the drop out. Now choose k eingenvector, in
    decreasing order. And w is the tensor of has the k vector inside.
    
    The next step is transform the samples onto new subspace.
    In last step, we use the nxk dimensional natrix w that we just computed to transform our samples
    onto the new subspace via the equation:
            
                        z = w.T @ x
    
    SO lastly, we have computed our k principal component abd projected the data points onto the new
    subspace.
    """

    def __init__(self,x_data1,x_data2):
        
        self.x_data1       =    x_data1
        self.x_data2       =    x_data2
        self.all_data      =    np.concatenate((self.x_data1, self.x_data2), axis=1)
        
    def Plot_How_Distributed(self):
        
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        plt.rcParams['legend.fontsize'] = 10   
        ax.plot(self.x_data1[0,:], self.x_data1[1,:], self.x_data1[2,:], 'o',
                markersize=8, color='blue', alpha=0.5, label='Data 1')
        ax.plot(self.x_data2[0,:], self.x_data2[1,:], self.x_data2[2,:], '^',
                markersize=8, alpha=0.5, color='red', label='Data 2')

        plt.title('Samples for Data 1 and Data 2')
        ax.legend(loc='upper right')

        plt.show()
        plt.savefig("how-distributed.png")
        
    
    def Mean(self):
        
        self.mean_x1          = np.mean(self.all_data[0,:])
        self.mean_x2          = np.mean(self.all_data[1,:])
        self.mean_x3          = np.mean(self.all_data[2,:])
        self.mean_vector = np.array([[self.mean_x1],[self.mean_x2],[self.mean_x3]])

    def Covariance_Scatter(self):
        
        self.covariance_matrix = np.zeros((3,3))
        for i in range(self.all_data .shape[1]):
            self.covariance_matrix += (self.all_data [:,i].reshape(3,1) - self.mean_vector).dot(
                    (self.all_data [:,i].reshape(3,1) - self.mean_vector).T)
        self.scatter_matrix    = self.covariance_matrix
        self.covariance_matrix = self.covariance_matrix / (39)
        print('Covariance Matrix:\n', self.covariance_matrix)
        print('Covariance Matrix:\n', self.scatter_matrix)
   
    def Eigenvectors_and_Eigenvalues(self):
        
        # eigenvectors and eigenvalues for the from the scatter matrix
        self.eig_val_sc, self.eig_vec_sc   = np.linalg.eig(self.scatter_matrix)
        # eigenvectors and eigenvalues for the from the covariance matrix
        self.eig_val_cov, self.eig_vec_cov = np.linalg.eig(self.covariance_matrix)
        
        for i in range(len(self.eig_val_sc)):
            self.eigvec_sc = self.eig_vec_sc[:,i].reshape(1,3).T
            self.eigvec_cov = self.eig_vec_cov[:,i].reshape(1,3).T
            assert self.eigvec_sc.all() == self.eigvec_cov.all(), 'Eigenvectors are not identical'

            print('Eigenvector {}: \n{}'.format(i+1, self.eigvec_sc))
            print('Eigenvalue {} from scatter matrix: {}'.format(i+1, self.eig_val_sc[i]))
            print('Eigenvalue {} from covariance matrix: {}'.format(i+1, self.eig_val_cov[i]))
            print('Scaling factor: ', self.eig_val_sc[i]/self.eig_val_cov[i])
            print(40 * '-')
            
    def Visualize_Eigenvectors(self):
        
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.all_data[0,:], self.all_data[1,:], self.all_data[2,:], 'o',
                markersize=8, color='green', alpha=0.2)
        ax.plot([self.mean_x1], [self.mean_x2], [self.mean_x3], 'o', 
                markersize=10, color='red', alpha=0.5)
        
        for i in self.eig_vec_sc.T:
            a = Arrow3D([self.mean_x1, i[0]], [self.mean_x2, i[1]], [self.mean_x3, i[2]],
                        mutation_scale=20, lw=3, arrowstyle="-|>", color="r")
            ax.add_artist(a)
        
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('x3')

        plt.title('Eigenvectors')

        plt.show()
        plt.savefig("eigenvectors.png")
        
    def Sort_Eigenvectors_and_Choose_k(self):
        
        self.eig_pairs = [(np.abs(self.eig_val_sc[i]), self.eig_vec_sc[:,i]) for i in range(len(self.eig_val_sc))]
        self.eig_pairs.sort(key=lambda x: x[0], reverse=True)
        for i in self.eig_pairs:
            print(i[0]),
        
        self.matrix_w = np.hstack((self.eig_pairs[0][1].reshape(3,1), self.eig_pairs[1][1].reshape(3,1)))
        print('Matrix W:\n', self.matrix_w)
        
    def Transform_Samples_Onto_New_Subspace(self):
        
        fig = plt.figure(figsize=(7,7))
        self.transformed = self.matrix_w.T.dot(self.all_data )
        plt.plot(self.transformed[0,0:20], self.transformed[1,0:20], 'o', 
                 markersize=7, color='blue', alpha=0.5, label='Data 1')
        plt.plot(self.transformed[0,20:40], self.transformed[1,20:40], '^',
                 markersize=7, color='red', alpha=0.5, label='Data 2 ')
        plt.xlim([-4,4])
        plt.ylim([-4,4])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.title('Transformed Samples With Labels')

        plt.show()
        plt.savefig("transformed-labels.png")
        
    def PrincipalComponentAnalysis(self):
        model= {}
        self.Plot_How_Distributed()
        self.Mean()
        self.Covariance_Scatter()
        self.Eigenvectors_and_Eigenvalues()
        self.Visualize_Eigenvectors()
        self.Sort_Eigenvectors_and_Choose_k()
        self.Transform_Samples_Onto_New_Subspace()
        model['Covariance Matrix'] = self.covariance_matrix
        model['Scatter Matrix'] = self.scatter_matrix
        model['W'] =self.matrix_w
        model['Transformed'] = self.transformed 
        return(model)

if __name__ == "__main__":

    vec1 = np.array([0,0,0])
    cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
    data1 = np.random.multivariate_normal(vec1, cov_mat1, 20).T
    vec2 = np.array([1,1,1])
    cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
    data2 = np.random.multivariate_normal(vec2, cov_mat2, 20).T
    
    model= {}
    PCA=PCA(data1,data2)
    model=PCA.PrincipalComponentAnalysis()
        
    