# 将分割图和原图合在一起
from PIL import Image
import matplotlib.pyplot as plt
import os
import time
import json

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from src import fcn_resnet50

imagelist1 = os.listdir('D:/fcn/VOCdevkit/Test/')
rootdir1 = "D:/fcn/VOCdevkit/Test/"
imagelist2 = os.listdir('D:/fcn/result/')
rootdir2 = "D:/fcn/result/"
# image1 原图
# image2 分割图
print(rootdir2 + imagelist2[0])
for i in range(len(imagelist1)):
    image1 = Image.open(rootdir1 + imagelist1[i])
    print(rootdir1 + imagelist1[i])
    print(image1)
    image2 = Image.open(rootdir2 + imagelist2[i])
    print(rootdir2 + imagelist2[i])
    print(image2)
    # image2.show()
    # image1 = Image.open("test.jpg")
    # image2 = Image.open("test_result.png")
    width = min(image2.width, image1.width)
    height = min(image1.height, image2.height)
    image2 = image2.resize((width, height))
    image1 = image1.resize((width, height))
    image1 = image1.convert('RGBA')
    image2 = image2.convert('RGBA')

    image = Image.blend(image1, image2, 0.5)
    image.save("D:/fcn/chonghe/duibi{}.png".format(i))
    # image.show()

    # plt.show()
