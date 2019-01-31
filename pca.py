from read import read_files, standardize_data
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

def display_pca(results, cov):
    projection_matrix = project(cov, 2)
    z = np.matmul(results, projection_matrix)
    z = np.real(z)
    plt.scatter(z[:,0], z[:,1])
    plt.savefig("pca", bbox_inches="tight")
    plt.show()

def project(cov, k):
    w, v = LA.eig(cov)
    idx = np.argsort(w)[::-1][0:k]

    projection_matrix = []
    for i in idx:
        eig_vec = v[:,i]
        projection_matrix.append(eig_vec)

    return np.array(projection_matrix).transpose()

results = np.array(read_files())
results = standardize_data(results)
cov = np.cov(results, rowvar=False)

display_pca(results, cov)
