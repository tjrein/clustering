from PIL import Image
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from read import read_files, standardize_data

def project(cov, k):
    w, v = LA.eig(cov)
    idx = np.argsort(w)[::-1][0:k]

    projection_matrix = []
    for i in idx:
        eig_vec = v[:,i]
        projection_matrix.append(eig_vec)

    return np.array(projection_matrix).transpose()

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
    fig = plt.figure(1)
    projection_matrix = get_k_eig(cov)
    pc1 = projection_matrix[:,0].copy()
    img = convert_to_img(pc1)
    plt.imshow(img)

def convert_to_img(vector):
    vector -= vector.min()
    vector /= vector.max()/255.0
    uint8 = np.uint8(vector).reshape((40,40))
    return Image.fromarray(uint8)

def reconstruct_face(results, cov):
    fig = plt.figure(2)

    pc1 = project(cov, 1)
    projection_matrix = get_k_eig(cov)

    a = fig.add_subplot(131)
    a.set_title("Original Image")
    face_1 = results[0].copy()
    plt.imshow(convert_to_img(face_1))

    a = fig.add_subplot(132)
    a.set_title("Single PC")
    pc1_projection = np.matmul(results, pc1)
    obs_1 = pc1_projection[0].copy()
    pc1_reconstruction = np.matmul(obs_1, pc1.transpose())
    plt.imshow(convert_to_img(pc1_reconstruction))

    a = fig.add_subplot(133)
    a.set_title("k PC")
    z = np.matmul(results, projection_matrix)
    obj1 = z[0].copy()
    reconstruction = np.matmul(obj1, projection_matrix.transpose())
    img = convert_to_img(reconstruction)
    plt.imshow(img)


results = np.array(read_files())
results = standardize_data(results)
cov = np.cov(results, rowvar=False)

display_pc1(results, cov)
reconstruct_face(results, cov)

plt.show()