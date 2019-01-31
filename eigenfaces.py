from PIL import Image
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from read import read_files, standardize_data

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

def display_pc1(results, cov):
    projection_matrix = get_k_eig(cov)
    pc1 = projection_matrix[:,0]
    img = convert_to_img(pc1)
    plt.imshow(img)
    plt.show()

def convert_to_img(vector):
    vector -= vector.min()
    vector /= vector.max()/255.0
    uint8 = np.uint8(vector).reshape((40,40))
    return Image.fromarray(uint8)


def reconstruct_face(results, cov):
    projection_matrix = get_k_eig(cov)
    z = np.matmul(results, projection_matrix)
    obj1 = z[0]
    reconstruction = np.matmul(obj1, projection_matrix.transpose())
    img = convert_to_img(reconstruction)
    plt.imshow(img)
    plt.show()


results = np.array(read_files())
results = standardize_data(results)
cov = np.cov(results, rowvar=False)

display_pc1(results, cov)
#reconstruct_face(results, cov)
