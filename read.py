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

def get_k_eig(cov):
    w, v = LA.eig(cov)
    eig_sum = np.sum(w)
    idx = np.argsort(w)[::-1]

    projection_matrix = []
    k_sum = 0
    for i in idx:
        eig_vec = v[:,i]
        projection_matrix.append(eig_vec)

        k_sum += w[i]
        if (k_sum / eig_sum) > 0.95:
            break

    return np.array(projection_matrix).transpose()

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

#POC
#test = np.array([[4,1,2], [2,4,0], [2,3,-8], [3,6,0], [4,4,0], [9,10,1], [6,8,-2], [9,5,1], [8,7,10], [10,8,-5]])
#new_test = standardize_data(test)
#cov = np.cov(new_test, rowvar=False)
#k = get_k_eig(cov)
#pc1 = k[:,0]
#print("HELLO", pc1.shape)

results = np.array(results)
results = standardize_data(results)
cov = np.cov(results, rowvar=False)
projection_matrix = get_k_eig(cov)
pc1 = projection_matrix[:,0]

pc1 -= pc1.min()
pc1 /= pc1.max()/255.0
uint8 = np.uint8(pc1).reshape((40,40))

test_image = Image.fromarray(uint8)

plt.imshow(test_image)
plt.show()

#test = np.reshape(pc1, (40, 40))
#print("okay," pc1.shape)
#print(projection_matrix[0:]
#projection_matrix = project(cov)

#z = np.matmul(results, projection_matrix)

#plt.scatter(z[:,0], z[:,1])
#plt.show()
