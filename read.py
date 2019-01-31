import os
from PIL import Image
import numpy as np

def read_files():
    results = []
    for root, dirs, files in os.walk("./yalefaces/yalefaces/"):
        for name in files:
            if name == "Readme.txt":
                continue

            filename = os.path.join(root, name)
            im = Image.open(filename).resize((40, 40))
            results.append(list(im.getdata()))

    return results

def standardize_data(matrix):
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0, ddof=1)
    return (matrix - mean) / std
