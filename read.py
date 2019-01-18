from PIL import Image
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def standardize_data(matrix):
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0, ddof=1)
    return (matrix - mean) / std

def project(cov):
    w, v = LA.eig(cov)
    idx = np.argsort(w)[::-1][0:2]

    projection_matrix = []
    for i in idx:
        eig_vec = v[:,i]
        projection_matrix.append(eig_vec)

    return np.array(projection_matrix).transpose()

results = []

for root, dirs, files in os.walk("./yalefaces/yalefaces/"):
    for name in files[1:]:
        filename = os.path.join(root, name)
        im = Image.open(filename).resize((40, 40))
        results.append(list(im.getdata()))


results = np.array(results)
results = standardize_data(results)
cov = np.cov(results, rowvar=False)

#test = np.array([[4,1,2], [2,4,0], [2,3,-8], [3,6,0], [4,4,0], [9,10,1], [6,8,-2], [9,5,1], [8,7,10], [10,8,-5]])
#new_test = standardize_data(test)
#cov = np.cov(new_test, rowvar=False)

projection_matrix = project(cov)
z = np.matmul(results, projection_matrix)
plt.scatter(z[:,0], z[:,1])
plt.show()
