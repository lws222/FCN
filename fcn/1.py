import os
import time
import json

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from src import fcn_resnet50


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    aux = False  # inference time not need aux_classifier
    classes = 4
    weights_path = "F:/jieguo/fcn/resnet50/4classes/fcn 4 1000 50.pth"
    # img_path = "./test.jpg"
    palette_path = "./palette.json"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    # assert os.path.exists(img_path), f"image {img_path} not found."
    assert os.path.exists(palette_path), f"palette {palette_path} not found."
    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = fcn_resnet50(aux=aux, num_classes=classes + 1)

    # delete weights about aux_classifier
    weights_dict = torch.load(weights_path, map_location='cpu')['model']
    for k in list(weights_dict.keys()):
        if "aux" in k:
            del weights_dict[k]

    # load weights
    model.load_state_dict(weights_dict)
    model.to(device)

    # 读取images文件夹下所有文件的名字
    imagelist = os.listdir('D:/fcn/VOCdevkit/Test/')

    rootdir = "D:/fcn/VOCdevkit/Test/"
    print(rootdir + imagelist[0])
    for i in range(len(imagelist)):
        img = Image.open(rootdir + imagelist[i])
        print(rootdir + imagelist[i])
        print(img)
        # load image
        original_img = img

        # from pil image to tensor and normalize
        data_transform = transforms.Compose([transforms.Resize(520),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                  std=(0.229, 0.224, 0.225))])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()  # 进入验证模式
        with torch.no_grad():
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            output = model(img.to(device))
            t_end = time_synchronized()
            print("inference+NMS time: {}".format(t_end - t_start))

            prediction = output['out'].argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            mask = Image.fromarray(prediction)
            print(mask)
            mask.putpalette(pallette)
            mask.save("D:/fcn/result/temp{:0>#3}.png".format(i))


if __name__ == '__main__':
    main()
