from PIL import Image
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

results = []

def standardize_data(matrix):
    mean = np.mean(matrix, axis=0)
    print("mean", mean)
    std = np.std(matrix, axis=0, ddof=1)
    print("std", std)
    return (matrix - mean) / std

for root, dirs, files in os.walk("./yalefaces/yalefaces/"):
    for name in files[1:]:
        filename = os.path.join(root, name)
        im = Image.open(filename).resize((40, 40))
        results.append(im.getdata())

test = np.array([[4,1,2], [2,4,0], [2,3,-8], [3,6,0], [4,4,0], [9,10,1], [6,8,-2], [9,5,1], [8,7,10], [10,8,-5]])
new_test = standardize_data(test)
cov = np.cov(new_test, rowvar=False)
w, v = LA.eig(cov)

print("new_test", new_test[0])
print("test", v[:,0])

print(np.matmul(new_test, v[:,0]))

#results = np.array(results)
#results = standardize_data(results)
#print("results", results.shape)
#cov = np.cov(results)
#print("uhhhh", cov.shape)
