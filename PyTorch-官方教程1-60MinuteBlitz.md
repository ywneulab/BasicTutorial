---
title: PyTorch官方教程(一)-A 60 Minute Blitz
sitemap: true
categories: PyTorch
date: 2018-07-04 20:50:28
tags:
- PyTorch
---

# What is PyTorch?

一个基于Python的科学计算包, 设计目的有两点:
- numpy在GPUs实现上的替代品
- 具有高度灵活性和速度的深度学习研究平台

## Tensors

Tensors可以理解成是Numpy中的ndarrays, 只不过Tensors支持GPU加速计算.

```py
x = torch.empty(5,3)
print(x) # 输出 5×3 的未初始化的矩阵, 矩阵元素未初始化, 所以可能是浮点类型的任何职

x = torch.rand(5,3)

x = torch.zeros(5,4,dtype=torch.long)

x = torch.tensor([5.5, 3]) # 直接用常数来初始化一个Tensor

x.size() # Tensor的size
```

## Operations

PyTorch支持多种语法实现相同的操作.

**加法:**
```py
x = torch.rand(5,3)
y = torch.rand(5,3)

z1 = x + y
z2 = torch.add(x,y)

z3 = torch.empty(5,3)
torch.add(x,y,out=z3)

# in-place
y.add_(x) # _ 代表原地加法 也就是 y = y+x


# 可以想numpy数组一样使用tensor:
print(x[:,-1])

# Resizing, 利用torch.view来对tensor执行reshape/resize操作
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1,8) # -1代表自动推断维度
print(x.size(), y.size(), z.size()) # torch.Size([4,4]) torch.Size([16]) torch.Size([2,8])

# item()可以获得只有一个元素的tensor的值
x = torch.randn(1)
print(x.item())
```

## Tensor与Numpy Array

**从tensor转换成numpy数组:**

```py
a = torch.ones(5)
print(type(a)) # <class 'torch.Tensor'>
b = a.numpy()
print(type(b)) # <class 'numpy.ndarray'>
```

**注意, 此时a和b共享内存, 即a和b指向的都是同一个数据, 也就是说, 如果改变a的值, 那么b的值也会随之改变!!**
```py
print(a.add_(1)) # tensor([2., 2., 2., 2., 2])
print(b) # [2., 2., 2., 2., 2]
```

**从numpy数组转换成tensor**

```py
a = np.ones(5)
b = torch.from_numpy(a)
```
**同样, a和b是共享内存的**

所有位于CPU上的Tensor (除了CharTensor) 都支持转换成对应的numpy数组并且再转换回来.

## CUDA Tensors

Tensors可以利用`.to`方法移动到任何设备上去
```py
if torch.cuda.is_avaiable():
    device = torch.device("cuda") # 创建了一个cuda device对象
    y = torch.ones_like(x, device=device) # 直接从GPU上创建tensor
    x = x.to(device) # 将x移到gpu上, 也可以直接用字符串指明: x = x.to("cuda")
    z = x+y
    z.to("cpu", torch.double)
```

# Neural Networks

可以利用`torch.nn`包来创建神经网络, `nn`依靠`autograd`来定义模型并且对其计算微分. 从`nn.Module`类派生的子类中会包含模型的layers, 子类的成员函数`forward(input)`会返回模型的运行结果.

经典的训练神经网络的过程包含以下步骤:
- 定义具有一些可学习参数(权重)的神经网络
- 在数据集上创建迭代器
- 将数据送入到网络中处理
- 计算loss
- 对参数进行反向求导
- 更新参数: $weight = weight - lr*gradient$


## 定义一个简单的网络

```py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)
```
输出如下
```py
Net(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
```

当定义好模型的`forward()`函数以后, `backward()`函数就会自动利用`autograd`机制定义, 无需认为定义.

可以通过`net.parameters()`函数来获取模型中可学习的参数

```py
params = net.parameter() # params的类型为 <class 'Iterator'>
print(len(list(params))) # 具有可学习参数的层数
print(list(params)[0].size()) # conv1 的参数
```

根据网络结构接受的输入, 想网络中传输数据并获取计算结果
```py
input = torch.randn(1,1,32,32) # 四个维度分别为 (N,C,H,W)
out = net(input) # 自动调用forward函数进行计算并返回结果
print(out)  #tensor([[ 0.1246, -0.0511, 0.0235, 0.1766,  -0.0359, -0.0334, 0.1161, 0.0534, 0.0282, -0.0202]], grad_fn=<ThAddmmBackward>)
```

下面的代码可以清空梯度缓存并计算所有需要求导的参数的梯度
```py
net.zero_grad()
out.backward(torch.randn(1,10)) # 正如前面所说, 当定义了forward函数以后, 就会自动定义backward函数, 因此可以直接使用
```

**需要注意的是, 整个`torch.nn`包只支持mini-batches, 所以对于单个样本, 也需要显示指明batch size=1, 即input第一个维度的值为1**

也可以对单个样本使用`input.unsqueeze(0)`来添加一个假的batch dimension.

## Loss Function

一个损失函数往往接收的是一对儿数据 `(output, target)`. 然后根据相应规则计算`output`和`target`之间相差多远, 如下所示:

```py
output = net(input)
target = torch.randn(10)
target = target.view(1,-1) # 令target和output的shape相同.
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss) # tensor(1.3638, grad_fn=<MseLossBackward>)
```

利用`.grad_fn`属性, 可以看到关于loss的计算图:
```py
print(loss.grad_fn) # 返回MseLossBackward对象
#input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
#      -> view -> linear -> relu -> linear -> relu -> linear
#      -> MSELoss
#      -> loss
```

因此, 当调用`loss.backward()`时, 就会计算出所有(`requires_grad=True`的)参数关于loss的梯度, 并且这些参数都将具有`.grad`属性来获得计算好的梯度


## BackProp

再利用`loss.backward()`计算梯度之前, 需要先清空已经存在的梯度缓存(因为PyTorch是基于动态图的, 每迭代一次就会留下计算缓存, 到一下次循环时需要手动清楚缓存), 如果不清除的话, 梯度就换累加(注意不是覆盖).

```py
net.zero_grad()  # 清楚缓存
print(net.conv1.bias.grad) # tensor([0., 0., 0., 0., 0., 0.])

loss.backward()

print(net.conv1.bias.grad) # tensor([ 0.0181, -0.0048, -0.0229, -0.0138, -0.0088, -0.0107])
```

## Update The Weights

最简单的更新方法是按照权重的更新公式:
```py
learning_rate = 0.001
for f in net.parameters():
    f.data.sub_(learning_rate*f.grad.data)
```

当希望使用一些不同的更新方法如SGD, Adam等时, 可以利用`torch.optim`包来实现, 如下所示:
```py
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.01) # 创建优化器
optimizer.zero_grad() # 清空缓存
output = net(input)
loss = criterion(output, target)
loss.backward() # 计算梯度
optimizer.step() # 执行一次更新
```

# Train A Classifier


## What About Data?

通常情况下, 在处理数据的时候可以使用标准的Python包(opencv, skimage等), 并将其载入成Numpy数组的形式, 然后可以很方便的将其转换成`torch.*Tensor`数据.

对于图像数据来说, PyTorch提供了`torchvision`包, 它包含许多常见数据集(Imagenet, CIFAR10, MNIST等等)的加载器, 同时还包含其他一些针对图片的数据转换(data transformers)函数. 对于CIFAR10来说, 它的数据集中图片尺寸为 3×32×32, 总共具有10个不同的类别. 下面就来看一下如何训练一个分类器将这10个类别进行分类.

## Training An Image Classifier

接下来主要包括以下步骤:
- 使用`torchvision`加载并归一化CIFAR10的训练数据集和测试数据集.
- 定义一个卷积神经网络
- 定义损失函数
- 在traing data上训练网络
- 在test datauh测试网络

**Loading and normalizing CIFAR10:**

导入相关的包
```py
import torch
import torchvision
import torchvision.transforms as transforms
```
`torchvision`的输出类型是 PILImage. 我们需要将其转换成 Tensors, 并对其进行归一化, 使其数值处于 [-1, 1] 之间.
```py
# 将多个transforms链接(chained)起来
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

利用下面的代码可以查看CIFAR10中的训练图片样本:
```py
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
```

**定义卷积神经网络:**
```py
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Model):

    def __init__(self):
        super(self, Net).__init__
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2) # 两个max pooling的参数是一样的, 所以定义一个就行, 可以重复使用
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input):
        x = self.pool(F.relu(self.conv1(input)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5) # 第一个维度为batch size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output

net = Net()
```

**Define a Loss function and optimizer:**

损失函数使用交叉熵, 优化器使用带动量的SGD
```py
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

**训练网络:**

训练网络的时候, 我们需要简单的在数据迭代器上进行循环操作就可以, 只需要注意不断想网络中送入新的数据即可.
```py
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
```

**Test the network on the test data**

在测试集上获取模型的准确率, 只需要利用`outputs = net(images)`即可获得预测的类别概率, 取最大者为预测的类别结果.

```py
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
```

利用下面的代码可以看到每个类别的准确率:

```py
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
```

## Training on GPU
上面的代码是在CPU上训练的, 那么如何利用PyTorch在GPU上进行训练呢? 实际上, 只需要将模型转移到GPU上即可. 首先定义一个device对象:
```py
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device) # 输出 cdua:0
```

接下来, 利用`.to()`方法将模型转移到GPU上面(同时所有的参数和梯度缓存也会转移到GPU上)
```py
net.to(device) # 也可以直接写成 net.to(device), 但是这样会缺少了设备检查, 不够健壮
```

接下来, 再向模型投喂数据之前, 就需要先将数据转移到GPU上
```py
inputs, labels = inputs.to(device), labels.to(device)
```

其余代码均与上面的训练代码相同.

## Training on multiple GPUs

//TODO
