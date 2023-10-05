
import numpy as np
import pandas as pd

import torch
from torchvision import models
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torchvision

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import os
import json
import re
import random
from math import sqrt

!pip install -U torch torchvision

import torch
import numpy as np

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

# Commented out IPython magic to ensure Python compatibility.
import pickle
import numpy as np
import pandas as pd
import random
from skimage import io

from tqdm import tqdm, tqdm_notebook
from PIL import Image, ImageFont
from pathlib import Path

from torchvision import transforms
from multiprocessing.pool import ThreadPool
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from matplotlib import colors, pyplot as plt
import seaborn as sns
# %matplotlib inline


import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from google.colab import drive
drive.mount('/content/gdrive/')

!unzip -q /content/gdrive/MyDrive/archive.zip -d mask_dataset

from matplotlib import colors, pyplot as plt

# this code is needed to visualize an example of images, bounding boxes and labels:

with open('/content/mask_dataset/annotations/maksssksksss0.xml') as f:
    reader = f.read()

img = Image.open('/content/mask_dataset/images/maksssksksss0.png')

xmin = [j for j in range(3)]
xmax = [j for j in range(3)]
ymin = [j for j in range(3)]
ymax = [j for j in range(3)]

labels = []

print(re.findall('(?<=<name>)[BG | without_mask | with_mask | mask_weared_incorrect]+?(?=</name>)', reader))

for i in range (3):
  xmin[i] = int(re.findall('(?<=<xmin>)[0-9]+?(?=</xmin>)', reader)[i])
  xmax[i] = int(re.findall('(?<=<xmax>)[0-9]+?(?=</xmax>)', reader)[i])
  ymin[i] = int(re.findall('(?<=<ymin>)[0-9]+?(?=</ymin>)', reader)[i])
  ymax[i] = int(re.findall('(?<=<ymax>)[0-9]+?(?=</ymax>)', reader)[i])

  label = re.findall('(?<=<name>)[BG | without_mask | with_mask | mask_weared_incorrect]+?(?=</name>)', reader)[i]
  labels.append(label)


length = len(re.findall('(?<=<xmin>)[0-9]+?(?=</xmin>)', reader))

origin_img = img.copy()
draw = ImageDraw.Draw(origin_img)


for k in range (length):

    draw.rectangle(xy=[(xmin[k],ymin[k]), (xmax[k],ymax[k])])

    text = labels[k]
    w, h = 70, 10  #font.getsize(text)
    draw.rectangle([(xmin[k], ymin[k] + h), (xmin[k] + w, ymin[k])],  fill = 'red')   #[box[0], box[1], box[0] + w, box[1] + h], fill = 'red')
    draw.text((xmin[k], ymin[k]), text, font = None, fill = 'white')


# Then I try to make the image of size (300, 300)

origin_img

dims = (300, 300)
old_dims = torch.FloatTensor([img.width, img.height, img.width, img.height]).unsqueeze(0)
print(old_dims)
new_box = []

for k in range (3):
  box1 = torch.FloatTensor([xmin[k], ymin[k], xmax[k], ymax[k]])
  new_box1 = box1 / old_dims
  new_dims1 = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
  new_box1 = new_box1 * new_dims1
  new_box.append(new_box1)


# Draw the image with modified size

img = transforms.Resize(dims)(img)
draw = ImageDraw.Draw(img)
draw.rectangle(xy=[tuple(new_box[0].tolist()[0])[:2], tuple(new_box[0].tolist()[0])[2:]])
draw.rectangle(xy=[tuple(new_box[1].tolist()[0])[:2], tuple(new_box[1].tolist()[0])[2:]])
draw.rectangle(xy=[tuple(new_box[2].tolist()[0])[:2], tuple(new_box[2].tolist()[0])[2:]])
img

label_map = {'BG': 3,
    'without_mask': 2,
    'with_mask': 1,
    'mask_weared_incorrect': 0
}

rev_label_map = {'BG': 3,
    'without_mask': 2,
    'with_mask': 1,
    'mask_weared_incorrect': 0
}

all_img_name = []
img_folder_path = Path('/content/mask_dataset/images')

all_img_name += list(map(lambda x: '/'+ x, os.listdir(img_folder_path)))

# Make a function new_boxes to resize the images and boxes 

def new_boxex(img, annotation):
  dims = (300, 300)
  old_dims = torch.FloatTensor([img.width, img.height, img.width, img.height]).unsqueeze(0)
  new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)

  count = len(re.findall('(?<=<xmin>)[0-9]+?(?=</xmin>)', reader))

  xmin = [j for j in range(count)]
  xmax = [j for j in range(count)]
  ymin = [j for j in range(count)]
  ymax = [j for j in range(count)]

  origin_img = img.copy()
  draw = ImageDraw.Draw(origin_img)

  for i in range (count):
    xmin[i] = int(re.findall('(?<=<xmin>)[0-9]+?(?=</xmin>)', reader)[i])
    xmax[i] = int(re.findall('(?<=<xmax>)[0-9]+?(?=</xmax>)', reader)[i])
    ymin[i] = int(re.findall('(?<=<ymin>)[0-9]+?(?=</ymin>)', reader)[i])
    ymax[i] = int(re.findall('(?<=<ymax>)[0-9]+?(?=</ymax>)', reader)[i])

    draw.rectangle(xy=[(xmin[i],ymin[i]), (xmax[i],ymax[i])])

  return [xmin, xmax, ymin, ymax]