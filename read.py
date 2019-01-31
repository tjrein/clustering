import os
from PIL import Image

def read_files():
    results = []
    for root, dirs, files in os.walk("./yalefaces/yalefaces/"):
        print("files", files)
        for name in files:
            if name == "Readme.txt":
                continue

            filename = os.path.join(root, name)
            im = Image.open(filename).resize((40, 40))
            results.append(list(im.getdata()))

    return results
