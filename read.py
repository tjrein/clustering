from PIL import Image
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random

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

def convert_to_img(vector):
    vector -= vector.min()
    vector /= vector.max()/255.0
    uint8 = np.uint8(vector).reshape((40,40))
    return Image.fromarray(uint8)

def display_pc1(results, cov):
    projection_matrix = get_k_eig(cov)
    pc1 = projection_matrix[:,0]
    img = convert_to_img(pc1)
    plt.imshow(img)
    plt.show()

def reconstruct_face(results, cov):
    projection_matrix = get_k_eig(cov)
    z = np.matmul(results, projection_matrix)
    obj1 = z[0]
    reconstruction = np.matmul(obj1, projection_matrix.transpose())
    img = convert_to_img(reconstruction)
    plt.imshow(img)
    plt.show()

results = []
for root, dirs, files in os.walk("./yalefaces/yalefaces/"):
    for name in files[1:]:
        filename = os.path.join(root, name)
        im = Image.open(filename).resize((40, 40))
        results.append(list(im.getdata()))



t_shirts = np.array([[61, 120],
                     [65, 130],
                     [72, 250],
                     [63, 120],
                     [62, 195],
                     [62, 120],
                     [60, 100],
                     [70, 140],
                     [70, 160],
                     [65, 132],
                     [48, 75],
                     [72, 175],
                     [67, 167],
                     [69, 140],
                     [96, 285],
                     [70, 172],
                     [70, 185],
                     [71, 168],
                     [70, 180],
                     [69, 170],
                     [70, 150],
                     [70, 170],
                     [71, 144],
                     [66, 140],
                     [67, 175],
                     [67, 165],
                     [72, 175]])

standard_tshirts = standardize_data(t_shirts)
num_obs = standard_tshirts.shape[0]

inds = random.sample(range(0, num_obs), 2)

ref1 = standard_tshirts[inds[0]]
ref2 = standard_tshirts[inds[1]]


#plt.scatter(standard_tshirts[:,0], standard_tshirts[:,1,], c="r", marker='x', s=10)


for i in standard_tshirts:
    distance_to_1 = LA.norm(i-ref1)
    distance_to_2 = LA.norm(i-ref2)

    color = 'b'

    if distance_to_1 < distance_to_2:
        color = 'r'

    plt.scatter(i[0], i[1], c=color, marker='x', s=10)


plt.scatter(ref1[0], ref1[1], c="r", marker="o", s=80)
plt.scatter(ref2[0], ref2[1], c="b", marker="o", s=80)

plt.show()
#Theory part 1
#test_data = np.array( [ [-2, 1], [-5, -4], [-3, 1], [0, 3], [-8, 11], [-2, 5], [1, 0], [5, -1], [-1, -3], [6, 1] ] )
#test_data = standardize_data(test_data)
#a_cov = np.cov(test_data, rowvar=False)
#w,v = LA.eig(a_cov)


#Theory part 1 example from slides
#other_test = np.array([ [4,1,2], [2,4,0], [2,3,-8], [3,6,0], [4,4,0], [9,10,1], [6,8,-2], [9,5,1], [8,7,10], [10,8,-5] ])
#other_test = standardize_data(other_test)
#b_cov = np.cov(other_test, rowvar=False)
#x,y = LA.eig(b_cov)




#results = np.array(results)
#results = standardize_data(results)
#cov = np.cov(results, rowvar=False)

#display_pca(results, cov)
#display_pc1(results, cov)
#reconstruct_face(results, cov)
