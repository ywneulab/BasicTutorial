---
title: PyTorch 踩坑
sitemap: true
categories: PyTorch
date: 2018-06-19 22:24:13
tags:
- PyTorch
- 踩坑记录
---

# 模型与参数的类型不符
Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same

TensorIterator expected type torch.cuda.FloatTensor but got torch.FloatTensor

要么需要在每一处新建立的tensor上将其手动移动到 cuda 或者 cpu 上, 要么利用下面的语句设置默认设备和类型
```py
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
```


https://www.zhihu.com/question/67209417/answer/250909765

# reshape 和 view 的不同

view 只能作用在连续的内存空间上. 并且不会对 tensor 进行复制. 当它作用在非连续内存空间的 tensor 上时, 会产生报错.
reshape 可以作用在任何空间上, 并且会在需要的时候创建 tenosr 的副本.

# module 'torchvision.datasets' has no attribute 'VOCDetection'

这是因为 `VOCDetection` 还没有添加到最新的 release 版本的导致的错误, 我们可以通过源码的方式重新安装 `torchvision`. 方法如下:

首先查看当前虚拟环境的 `torchvision` 的安装位置:
```py
import torchvision as tv

print(tv.__file__)

# /home/zerozone/.pyenv/versions/a3py3.5/lib/python3.5/site-packages/torchvision/__init__.py
```

然后进入上面的文件夹, 删除旧的 `torchvision`
```py
cd /home/zerozone/.pyenv/versions/a3py3.5/lib/python3.5/site-packages/

rm -rf torchvision*
```

然后下载最新版本的 `torchvision` 并安装(注意不要更换安装路径)

```py
git clone https://github.com/pytorch/vision.git

python setup.py install
```

最后查看新安装的 `torchvision` 中是否包含 `VOCDetection`:

```py
>>> import torchvision as tv
>>> print(dir(tv.datasets))
# ['CIFAR10', 'CIFAR100', 'CocoCaptions', 'CocoDetection', 'DatasetFolder', 'EMNIST', 'FakeData', 'FashionMNIST', 'Flickr30k', 'Flickr8k', 'ImageFolder', 'LSUN', 'LSUNClass', 'MNIST', 'Omniglot', 'PhotoTour', 'SBU', 'SEMEION', 'STL10', 'SVHN', 'VOCDetection', 'VOCSegmentation', '__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', 'cifar', 'coco', 'fakedata', 'flickr', 'folder', 'lsun', 'mnist', 'omniglot', 'phototour', 'sbu', 'semeion', 'stl10', 'svhn', 'utils', 'voc']
```

可以看到, 新包含了 `'VOCDetection', 'VOCSegmentation', 'voc'` 等名称, 说明安装成功, 此时可以正常使用 `VOCDetection` 了.

# CUDA driver version is insufficient for CUDA runtime version

考虑可能是 cuda 或者 显卡驱动的版本不匹配, 也可能是 PyTorch 的版本过低导致的, 建议提升 PyTorch 版本至最新的稳定版.


# Differences between .data and .detach

【链接】PyTorch中tensor.detach()和tensor.data的区别
https://blog.csdn.net/DreamHome_S/article/details/85259533

https://github.com/pytorch/pytorch/issues/6990

detach 的作用: https://blog.csdn.net/qq_39709535/article/details/80804003

# Variable 在 PyTorch 1.0 中如何替换

# 将 PyTorch 模型转换成 Caffe 模型
https://github.com/longcw/pytorch2caffe


# DataLoader

**挂起:**
当`num_worker`参数大于1时, 会出现一些奇怪的现象, 表现为无法迭代(挂起, 死锁), 进而无法使用上面的进度条模块.

https://github.com/pytorch/pytorch/issues/1355
(仍然没有完全解决)

现在这种情况多出现在用户手动终止程序后, 再次启动程序时发生, 目前的解决方法是要在`dataloader`处使用异常捕获, 来避免进程之间的资源死锁


# RuntimeError: invalid argument 0: Sizes of tensors must match except in dimension 0. Got 37 and 1 in dimension 1

在 pytorch 中, dataloader 会自动将 datasets 中的数据组织成 tensor 的形式, 因此, 这就要求 batch 中的每一项元素的 shape 都要相同. 但是在目标检测中, 每一张图片所具有的 box 的数量是不同的, 因此, 需要自己实现`collate_fn`来构建 mini-batch 中每一个 samples. 如下所示(ssd代码):
```
def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets
```
