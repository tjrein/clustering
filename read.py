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

def display_pca(results, cov):
    projection_matrix = project(cov)
    z = np.matmul(results, projection_matrix)
    plt.scatter(z[:,0], z[:,1])
    plt.show()

def display_pc1(results, cov):
    projection_matrix = get_k_eig(cov)
    pc1 = projection_matrix[:,0]
    pc1 -= pc1.min()
    pc1 /= pc1.max()/255.0
    uint8 = np.uint8(pc1).reshape((40,40))
    test_image = Image.fromarray(uint8)
    plt.imshow(test_image)
    plt.show()

def reconstruct_face(results, cov):
    projection_matrix = get_k_eig(cov)
    z = np.matmul(results, projection_matrix)
    obj1 = z[0]
    reconstruction = np.matmul(obj1, projection_matrix.transpose())
    reconstruction -= reconstruction.min()
    reconstruction /= reconstruction.max()/255.0
    uint8 = np.uint8(reconstruction).reshape((40, 40))
    test_image = Image.fromarray(uint8)
    plt.imshow(test_image)
    plt.show()


results = []
for root, dirs, files in os.walk("./yalefaces/yalefaces/"):
    for name in files[1:]:
        filename = os.path.join(root, name)
        im = Image.open(filename).resize((40, 40))
        results.append(list(im.getdata()))


results = np.array(results)
results = standardize_data(results)
cov = np.cov(results, rowvar=False)

#display_pca(results, cov)
#display_pc1(results, cov)
#reconstruct_face(results, cov)
