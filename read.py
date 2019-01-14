from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

results = []

for root, dirs, files in os.walk("./yalefaces/yalefaces/"):
    for name in files[1:]:
        filename = os.path.join(root, name)
        im = Image.open(filename).resize((40, 40))
        results.append(im.getdata())


print(np.array(results).shape)



#plt.imshow(test)
