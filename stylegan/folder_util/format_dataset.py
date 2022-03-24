import os

# Was getting issues with dataset having special characters
# This just converts everything to a number

dataset_name = "destgun"
root = "./datasets/" + dataset_name + "/" + dataset_name + "_png/"
paths = [root + path for path in os.listdir(root)]

_, ext = os.path.splitext(paths[0])
os.rename(paths[0], root + "0" + ext)

for i, path in enumerate(paths):
    _, ext = os.path.splitext(paths[0])
    assert ext == ".png" or ext == ".jpg"
    os.rename(paths[i], root + str(i) + ext)
