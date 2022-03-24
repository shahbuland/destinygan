import os

root = "./staging/"
paths = os.listdir(root)
paths = [root + path for path in paths]

# Remove gparams
for path in paths:
    if path.find("gparams.pt") != -1:
        os.remove(path)

def get_key(path):
    first_g = path.find("gemaparams.pt")
    return int(path[len(root):first_g])

# Only keep every 1000th
for path in paths:
    k = get_key(path)
    if k % 1000 != 0: os.remove(path)
