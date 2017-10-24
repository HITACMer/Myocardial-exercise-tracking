import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import MaUtilities as mu
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision.models import vgg
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch.utils.model_zoo as model_zoo

import time
import os
from PIL import Image
import math

class Raw_Dataset(Dataset):
    def __init__(self, transform=None):
        super(Raw_Dataset, self).__init__()
        self.transform = transform

    def __getitem__(self, item):
        image = Image.open("SmallImage/%d.png" % (item + 1))
        image = image.resize((224, 224))
        image = np.asarray(image)
        image = image.reshape(224, 224, 3)
        #mu.show_detail(image)
        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(os.listdir("SmallImage/"))

# class MyVGG(nn.Module):
#     '''
#     Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     ReLU (inplace)
#     MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
#     Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     ReLU (inplace)
#     MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
#     Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     ReLU (inplace)
#     Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     ReLU (inplace)
#     MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
#     Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     ReLU (inplace)
#     Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     ReLU (inplace)
#     MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
#     Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     ReLU (inplace)
#     Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     ReLU (inplace)
#     MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
#     '''
#     def __init__(self, features, num_classes=1000):
#         super(MyVGG, self).__init__()
#         self.features = features
#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, num_classes),
#         )
#         self._initialize_weights()
#
#     def forward(self, x):
#         for layer in self.features:
#             print(layer)
#             x = layer(x)
#
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 n = m.weight.size(1)
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()

class MyVGG(vgg.VGG):
    def __init__(self, features):
        super(MyVGG, self).__init__(features, num_classes=1000)
        self.features = features

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, torch.nn.modules.conv.Conv2d):
                print(x.data.shape)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

transformer = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]
)
a = Raw_Dataset()
dataloader = DataLoader(Raw_Dataset(transformer), batch_size=1)


pretrained_model = models.vgg11(pretrained=True).cuda()
my_model = MyVGG(vgg.make_layers(vgg.cfg['A'])).cuda()
my_model.load_state_dict(model_zoo.load_url(vgg.model_urls['vgg11']))


for i, (data) in enumerate(dataloader):
    data = Variable(data).cuda()
    #print(data.data)
    output = my_model(data)
    print(i)
