import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils
import MaUtilities as mu
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
import math
from torch.utils.data import Dataset, DataLoader
import torch
from torch.autograd import Variable
from torchvision.models import vgg
from torchvision import datasets, models, transforms
import torch.utils.model_zoo as model_zoo
import cv2

plt.ion()
features = torch.load("features")
print(len(features))

frame = 3
conv = 0

a = features[frame][conv].cpu().numpy()[0, ...]
mu.show_detail(a)
special_feature = [cv2.threshold(features[frame][1].cpu().numpy()[0][47], 0.5, 1, cv2.THRESH_BINARY)[1] for frame in range(220)]
torch.save(special_feature, "SpecialLayerFeature")

for frame in range(220):
    current_feature = features[frame][1].cpu().numpy()[0][47]
    (T, current_feature) = cv2.threshold(current_feature, 0.5, 1, cv2.THRESH_BINARY)#阈值化处理，阈值为：155
    original_image = Image.open("SmallImage/%d.png" % (frame+1))
    preprocessed_image = Image.open("P1_SmallImage/%d.png" % (frame+1))
    plt.ioff()
    plt.subplot(131)
    plt.imshow(original_image)
    plt.subplot(132)
    plt.imshow(preprocessed_image)
    plt.subplot(133)
    plt.imshow(current_feature)
    plt.title("%d" % frame)
    #plt.ion()
    plt.show()
    plt.pause(0.05)

assert 0
#mu.display(a[0, 0, ...])
for index in range(a.shape[1]):
    plt.imshow(a[index, ...])
    plt.title("%d" % index)
    plt.pause(0.5)
mu.display()
print(a.shape)