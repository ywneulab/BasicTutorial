---
title: PyTorch官方教程(二)-DataLoadingAndProcessing
sitemap: true
categories: PyTorch
date: 2018-07-05 21:50:33
tags:
- PyTorch
---


对于一个新的机器/深度学习任务, 大量的时间都会花费在数据准备上. PyTorch提供了多种辅助工具来帮助用户更方便的处理和加载数据. 本示例主要会用到以下两个包:
- scikit-image: 用于读取和处理图片
- pandas: 用于解析csv文件

导入下面的包
```py
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode
```

本示例使用的是人脸姿态的数据集, 数据集的标注信息是由68个landmark点组成的, csv文件的格式如下所示:
```
image_name,part_0_x,part_0_y,part_1_x,part_1_y,part_2_x, ... ,part_67_x,part_67_y
0805personali01.jpg,27,83,27,98, ... 84,134
1084239450_e76e00b7e7.jpg,70,236,71,257, ... ,128,312
```

利用如下代码可以快速的读取CSV文件里面的标注信息, 并且将其转换成 (N,2) 的数组形式, 其中, N 为 landmarks 点的个数
```py
landmarks_frame = pd.read_csv('faces/face_landmarks.csv')

n = 65
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
landmarks = landmarks.astype('float').reshape(-1, 2)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))
```

利用下面的函数可以将图像和标注文件中的点显示出来, 方便观察:
```py
def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

plt.figure()
show_landmarks(io.imread(os.path.join('faces/', img_name)),
               landmarks)
plt.show()
```

# Dataset class

`torch.utils.data.Dataset`实际上是一个用来表示数据集的虚类, 我们可以通过集成该类来定义我们自己的数据集, 在继承时, 需要重写以下方法:
- `__len__`: 让自定义数据集支持通过`len(dataset)`来返回dataset的size
- `__getitem__`: 让自定义数据集支持通过下标`dataset[i]`来获取第 $i$ 个数据样本.

接下来, 尝试创建人脸姿态的自定义数据集. 我们将会在`__init__`函数中读取csv文件, 但是会将读取图片的逻辑代码写在`__getitem__`方法中. 这么做有助于提高内存使用效率, 因为我们并不需要所有的图片同时存储在内存中, 只需要在用到的时候将指定数量的图片加载到内存中即可.

我们的数据集样本将会是字典形式: `{'image': image, 'landmarks':landmarks}`.  我们的数据集将会接受一个可选参数`transform`, 以便可以将任何需要的图片处理操作应用在数据样本上. 使用`transform`会使得代码看起来异常整洁干净.
```py
class FaceLandmarksDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        # 参数:
        # csv_file(string): csv标签文件的路径
        # root_dir(string): 所有图片的文件夹路径
        # transform(callable, optioinal): 可选的变换操作
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks.astype("float").reshape(-1,2)
        sample = {"image": image, "landmarks":landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
```

接下来, 让我们对这个类进行初始化
```py
face_dataset = FaceLandmarksDataset(csv_file="faces/face_.csv", root_dir="faces/")
fig = plt.figure()

for i in range(len(face_dataset)):
    sample = face_dataset[i]
    print(i, sample["image"].shape, sample["landmarks"])
    ax = plt.subplot(1,4,i+1)
    plt.tight_layout()
    ax.set_title("Sample")
    ax.axis("off")
    show_landmarks(**sample)
    if i==3:
        plt.show()
        break
```

# Transforms

尝试以下三种常见的转换操作:
- Rescale: 改变图片的尺寸大小
- RandomCrop: 对图片进行随机剪裁(数据增广技术)
- ToTensor: 将numpy图片转换成tensor数据

我们将会把这些操作写成可供调用的类, 而不仅仅是一个简单的函数, 这样做的主要好处是不用每次都传递transform的相关参数. 为了实现可调用的类, 我们需要实现类的 `__call__` 方法, 并且根据需要实现 `__init__` 方法. 我们可以像下面这样使用这些类:
```py
tsfm = Transform(params)
transformed_sample = tsfm(sample)
```
具体实现如下:
```py
class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h>w:
                new_h, new_w = self.output_size*h/w, self.out_size
            else:
                new_h, new_w = self.output_size, self.output_size*w/h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        landmarks = landmarks*[new_w/w, new_h/h]

        return {"image":img, "landmarks": landmarks}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}
```

# Compose transforms

接下来, 需要将定义好的转换操作应用到具体的样本上, 我们首先将特定的操作组合在一起, 然后利用`torchvision.transforms.Compose`方法直接将操作应用到对应的图片上.
```py
scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])

# Apply each of the above transforms on sample.
fig = plt.figure()
sample = face_dataset[65]
for i, tsfrm in enumerate([scale, crop, composed]):
    transformed_sample = tsfrm(sample)

    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**transformed_sample)

plt.show()
```

# Iterating through the dataset

总结一下对数据采样的过程:
- 从文件中读取一张图片
- 将transforms应用到图片上
- 由于transforms是随机应用的, 因此起到了一定的增广效果.

可以利用 `for i in range`循环操作来对整个数据集进行transforms
```py
transformed_dataset = FaceLandmarksDataset(csv_file='faces/face_landmarks.csv',root_dir='faces/',
                    transform=transforms.Compose([Rescale(256),RandomCrop(224),ToTensor()]))

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]

    print(i, sample['image'].size(), sample['landmarks'].size())

    if i == 3:
        break
```

# Afterword: torchvision
```py
import torch
from torchvision import transforms, datasets

data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
        ])
hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train',
                    transform=data_transform)
dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
                            batch_size=4, shuffle=True,
                            num_workers=4)
```
