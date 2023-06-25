import os
from PIL import Image
img_path = "./test.jpg"
img = Image.open(img_path)
print(img.shape[:])
#print(img.height)