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

class Raw_Dataset(Dataset):
    def __init__(self, transform=None):
        super(Raw_Dataset, self).__init__()
        self.transform = transform

    def __getitem__(self, item):
        image = Image.open("P1_SmallImage/%d.png" % (item + 1))
        image = image.resize((224, 224))
        image = np.asarray(image)
        #image = mu.copy_2D_to_3D(image, 3)
        image = image.reshape(224, 224, 3)
        #mu.show_detail(image)
        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(os.listdir("P1_SmallImage/"))

transformer = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    #transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]
)

class MyVGG(vgg.VGG):
    # Do NOT change variable name "features", because pretrained weights link to "features"
    def __init__(self, features):
        super(MyVGG, self).__init__(features, num_classes=1000)
        self.features = features
        # collect all feature images
        self.feature_images = []

    def forward(self, x):
        current_feature_images = []
        # Split VGG's sequential up to alone layers so that we can identify conv layers
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, torch.nn.modules.conv.Conv2d):
                # get features
                print(x.data.shape)
                current_feature_images.append(x.data)
        self.feature_images.append(current_feature_images)
        #x = x.view(x.size(0), -1)
        #x = self.classifier(x)
        return x

    def get_features(self):
        return self.feature_images
'''
my_model = MyVGG(vgg.make_layers(vgg.cfg['A'])).cuda()
my_model.load_state_dict(model_zoo.load_url(vgg.model_urls['vgg11']))

dataloader = DataLoader(Raw_Dataset(transformer), batch_size=1)

for i, (data) in enumerate(dataloader):
    data = Variable(data).cuda()
    output = my_model(data)

features = my_model.get_features()
print(len(features))
#print(features[2][3])


torch.save(features, "features")
'''
#plt.ion()
features = torch.load("SpecialLayerFeature")
top_features = [f[0: 33, ...] for f in features]
mid_features = [f[33: 66, ...] for f in features]
bottom_features = [f[66: 99, ...] for f in features]
diff = []
last_feature = None
for feature in bottom_features:
    if last_feature is not None:
        diff.append(np.sum(abs(feature - last_feature)))

    #plt.imshow(feature)
    #plt.pause(0.05)
    last_feature = feature
average = np.asarray(diff).sum() / len(diff)
#diff = [i - average for i in diff]


index = [i for i in range(len(diff))]
plt.title('Top area movement')
plt.plot(index, diff)
plt.legend(bbox_to_anchor=[0.3, 1])

plt.grid()
plt.show()











