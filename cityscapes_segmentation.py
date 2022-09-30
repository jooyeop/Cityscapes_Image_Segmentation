import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import os
from tqdm import tqdm
import warnings
import seaborn as sns

warnings.filterwarnings("ignore")
plt.style.use("seaborn-darkgrid")
sns.set_context('paper', font_scale=1.5)

Batch_size = 32

id_map = {
    0: (0, 0, 0), # unlabelled
    1: (111, 74,  0), #static
    2: ( 81,  0, 81), #ground
    3: (128, 64,127), #road
    4: (244, 35,232), #sidewalk
    5: (250,170,160), #parking
    6: (230,150,140), #rail track
    7: (70, 70, 70), #building
    8: (102,102,156), #wall
    9: (190,153,153), #fence
    10: (180,165,180), #guard rail
    11: (150,100,100), #bridge
    12: (150,120, 90), #tunnel
    13: (153,153,153), #pole
    14: (153,153,153), #polegroup
    15: (250,170, 30), #traffic light
    16: (220,220,  0), #traffic sign
    17: (107,142, 35), #vegetation
    18: (152,251,152), #terrain
    19: ( 70,130,180), #sky
    20: (220, 20, 60), #person
    21: (255,  0,  0), #rider
    22: (  0,  0,142), #car
    23: (  0,  0, 70), #truck
    24: (  0, 60,100), #bus
    25: (  0,  0, 90), #caravan
    26: (  0,  0,110), #trailer
    27: (  0, 80,100), #train
    28: (  0,  0,230), #motorcycle
    29: (119, 11, 32), #bicycle
    30: (  0,  0,142) #license plate 
}

category_map = {
    0: 0,
    1: 0,
    2: 0,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 2,
    8: 2,
    9: 2,
    10: 2,
    11: 2,
    12: 2,
    13: 3,
    14: 3,
    15: 3,
    16: 3,
    17: 4,
    18: 4,
    19: 5,
    20: 6,
    21: 6,
    22: 7,
    23: 7,
    24: 7,
    25: 7,
    26: 7,
    27: 7,
    28: 7,
    29: 7,
    30: 7
} # 0: unlabelled, 1: road, 2: building, 3: pole, 4: vegetation, 5: sky, 6: person, 7: vehicle

num_classes = len(id_map.keys()) # 31

def preprocess(path) :
    img = Image.open(path)
    img1 = img.crop((0, 0, 256, 256)).resize((128, 128)) # crop의 기능 = 이미지를 자르는 기능 왼쪽, 위, 오른쪽, 아래
    img2 = img.crop((256, 0, 512, 256)).resize((128, 128)) # crop을 하는 이유 = 이미지의 크기를 줄이기 위해서
    img1 = np.array(img1)/ 255. # 255로 나누는 이유 = 이미지의 픽셀값을 0~1사이의 값으로 바꾸기 위해서
    img2 = np.array(img2)
    mask = np.zeros(shape=(img2.shape[0], img2.shape[1]), dtype=np.uint32)
    for row in range(img2.shape[0]) :
        for col in range(img.shape[1]) :
            a = img2[row, col, :] # a = img2의 픽셀값
            final_key = None
            for key, value in id_map.items():
                d = np.sum(np.sqrt(pow( a - value, 2))) # d = a와 value의 차이 (픽셀값의 차이)
                if final_key == None:
                    final_d = d
                    final_key = key
                elif d < final_d:
                    final_d = d
                    final_key = key
            mask[row, col] = final_key # mask = img2의 픽셀값을 id_map의 픽셀값으로 바꾼 값
    mask = mask.reshape((mask.shape[0], mask.shape[1], 1))
    del img2
    return img1, mask

def prepare_tensor_dataset(train_path, val_path) :
    