---
title: PyTorch官方教程(三)-Learning PyTorch with Examples
sitemap: true
categories: PyTorch
date: 2018-07-07 20:05:23
tags:
- PyTorch
---

# Tensors

## Warm-up: numpy

对于numpy来说, 它对计算图, 深度学习, 梯度等等概念几乎是不知道的, 但是, 如果我们了解简单神经网络的具体结构, 那么我们就可以很轻易的用numpy来实现这个简单网络, 对此, 我们通常需要自己来实现前向计算和反向计算的逻辑, 下面我们来实现一个具有两层隐藏层的简单网络:

```py
import numpy as np

# N 为batch size, D_in 为输入维度
# H 为隐藏层的维度, D_out 为输出的维度
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建随机的输入和输出数据
x = np.random.randn(N, D_in) # N × D_in 的矩阵
y = np.random.randn(N, D_out) # N × D_out 的矩阵

# 对两个隐藏层w1,w2进行初始化
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

# 设置学习率
learning_rate = 1e-6
for t in range(500):
    # 前向传播: 计算预测结果 y_pred
    h = x.dot(w1) # x维度为64 × 1000, w1维度为 1000 × 100, 计算完以后, h维度为 64 × 100
    h_relu = np.maximum(h,0)
    y = h_relu.dot(w2) # h_relu维度为 64×100, w2维度为100×10, y的维度为64×10

    # 计算损失
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # 反向传播根据loss更新w1和w2的值
    grad_y_pred = 2.0*(y_pred - y) # 对y_pred求导
    grad_w2 = h_relu.T.dot(grad_y_pred) # 对w2求导, 微分矩阵应该与w2的size相同
    grad_h_relu = grad_y_pred.dot(w2.T) # 对h_relu求导
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = grad_h_relu # 经过relu, 将小于0的梯度归0
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 = w1 - learning_rate * grad_w1
    w2 = w2 - learning_rate * grad_w2

```
在执行上述代码以后, `w1`和`w2`的值会是的预测出来的`pred_y`与`y`之间的平方损失越来越小.

## PyTorch: Tensors
用PyTorch实现一个简单的神经网络
在神经网络的实现中, 较麻烦的是梯度的计算过程, 下面利用PyTorch的自动求导来实现一个简单的神经网络(两层隐藏层)



```py
import torch

dtype = torch.float
device = torch.device("cpu")

# N为batch size, D_in为input dimension
# H为hidden dimension, D_out为output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建输入和输出的Tensors
# requires_grad的值默认为False 指明无需计算x和y的梯度
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_in, device=device, dtype=dtype)

# 初始化两个隐藏层的参数, 注意要将requires_grad的值设置为True
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    ## torch.mm / torch.Tensor.mm : matrix multiplication
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    loss = (y_pred - y).pow(2).sum().item()
    print(t, loss)

    # BackProp
    grad_y_pred = 2*(y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h<0] = 0 # 将grad_h_relu中小于0的都置为0, 即为relu的反向传播公式(因为小于0的梯度为0, 大于0的梯度为1)
    grad_w1 = x.t().mm(grad_h)

    w1 -= lr * grad_w1
    w2 -= lr * grad_w2
```

# Autograd

## 自定义一个具有自动求导功能的PyTorch函数

上面的例子是使用手动的方式求梯度的, 当模型参数变多时, 这样的方式显然很不方便. 不过, 借助PyTorch的`autograd`模块, 可以方便的求取任意参数的导数.
在使用PyTorch的自动推导模块`autograd`时, 前向传播过程会被定义成一个计算图, 图中的节点是Tensors, 图中的边是一些函数, 用于根据 input Tensors 来生成 output Tensors. 比如当 `x` 是一个Tensor, 并且拥有属性`x.requires_grad=True`, 那么`x.grad`就是另一个Tensor, 它持有loss相对于`x`的梯度.



```py
import torch

dtype = torch.float
device = torch.device("cpu")

# N为batch size, D_in为input dimension
# H为hidden dimension, D_out为output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建输入和输出的Tensors
# requires_grad的值默认为False 指明无需计算x和y的梯度
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_in, device=device, dtype=dtype)

# 初始化两个隐藏层的参数, 注意要将requires_grad的值设置为True
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # 定义前向计算过程, mm为两个矩阵相乘, clamp可以将数据限定在某一范围内, 实现relu的功能
    y_pre = x.mm(w1).clamp(min=0).mm(w2)

    # 计算loss, loss.item()可以得到loss的矢量值
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # 只需要调用一条语句, 即可计算出所有requires_grad设置为True的参数的梯度, 可以通过w1或者w2的grad属性来访问各自的梯度.
    loss.backward() # 所有的ops, 如conv, relu等的backward方法已经在PyTorch内部实现

    # 手动更新参数(面对大型网络时, 可以通过调用torch.optim.SGD来自动更新)
    # 将参数放在 torch.no_grad() 管理环境当中, 这是因为我们无需对grad进行跟踪, 因此, 也需要在更新完参数以后, 将grad重新置为0 , 以便下一次更新
    with torch.no_grad():
        w1 -= learning_rate*w1.grad
        w2 -= learning_rate*w2.grad
        # 为了避免当前求取的梯度的值累加到下一次迭代当中, 调用zero_原地清空grad. 对于nn.Module, 可以调用`zero_grad`来清空所有Module中的参数的梯度.
        w1.grad.zero_()
        w2.grad.zero_()

```

## Defining new autograd functions

在PyTorch中, 每一个具有自动求导功能的operator都由两个作用在Tensors上的函数实现, 分别是用于计算输出的 **前向函数** , 以及用于计算梯度的 **反向函数**. 因此, 我们在可以在PyTorch中通过继承父类`torch.autograd.Function`, 并实现其中的`forward` 和 `backward` 函数来定义自己的自定义autograd functions

```py
import torch

class MyReLU(torch.autograd.Function):
    @staticmethod # 将该方法变成静态方法, 使得不用实例化也可以调用, 当前实例化也可以调用
    def forward(ctx, input):
        # ctx 是一个上下文管理器, 它可以利用`ctx.save_for_backward`把任何需要在backward用到的对象都存储起来
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output 为从下游传回来的梯度
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input>0] = 0
        return grad_input

# 使用方法: 使用.apply方法来应用自定义的ops
y_pred = MyReLU.apply(x.mm(w1)).mm(w2)

# 为了方便, 也可以先对MyReLU重命名, 然后调用更简洁的别名
relu = MyReLU.apply
y_pred = relu(x.mm(w1)).mm(w2)
```

# Static Graphs
PyTorch采用动态计算图, 而TensorFlow采用静态计算图

**静态计算图:** 只对计算图定义一次, 而后会多次执行这个计算图.
好处:
- 可以预先对计算图进行优化, 融合一些计算图上的操作, 并且方便在分布式多GPU或多机的训练中优化模型


**动态计算图:** 每执行一次都会重新定义一张计算图.
- 控制流就像Python一样, 更容易被人接受, 可以方便的使用for, if等语句来动态的定义计算图, 并且调试起来较为方便.

# nn Module

## nn

对于大型网络模型来说, 直接使用`autograd`有些太过底层(too low-level). 为此在搭建神经网络时, 我们经常会将计算放置到 **`layers`**上 , 这些 **`layers`** 中的可学习参数会在训练中就行更新. 在TF中, Keras, TF-Slim等提高了封装性更高的高层API, 在PyTorch中, `nn` 包可以提供这些功能. 在`nn`包中, 定义了一系列的 **Modules** , 可以类比为神经网络中的不同层. 一个 Module 会接受一组 input Tensors, 并计算出对应的 output Tensors, 同时会持有一些内部状态(如可学习的权重参数). 在`nn`包中还定义了一系列有用的 loss functins 可供使用. 下面尝试用 `nn` 包来实现上面的两层网络:

```py
import

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N,D_in)
y = torch.randn(N,D_out)

# 由于当前的两层网络是序列的, 因此可以使用 torch.nn.Sequential 来定义一个Module, 该 Module 中包含了一些其它的 Modules (如Linear, ReLU等),
# Sequential Module会序列化的执行这些 Modules, 并且自动计算其output和grads.
# 注意因为是序列化执行的, 因此无需自定义 forward. 这是与 nn.Module 的区别之一.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)

loss_fn = torch.nn.MESLoss(reduction="sum")

lr  = 1e04
for t in range(500):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # 在获取梯度前, 先清空梯度缓存
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        for param in model.parameters():
            param -= lr * param.grad
```

## optim

可以看到, 上面在更新参数时, 我们仍采取的是手动更新的方式, 对于简单的优化算法来说, 这并不是什么难事, 但是如果我们希望使用更加复杂的优化算法如AdaGrad, Adam时, 采用 `optim` 包提供的高层API可以方便的使用这些优化算法.
```py
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
for t in range(500):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad() # 已经将待优化参数model.parameters()传给优化器了
    loss.backward()
    optimizer.step() # 执行一次参数优化操作(是不是很简单?)
```

## Custom nn Modules

有时候, 我们需要定义一些相比于序列化模型更加复杂的模型, 此时, 我们可以通过继承`nn.Module`,同时定义`forward`前向计算函数来自定义一个 Module. 下面我们就用这种方式来自定义一个具有两层网络的 Module.
```py

class TwoLayerNet(torch.nn.Module):

    def __init__(self, D_in, H, D_out):
        # 通常我们将具有参数的层写在__init__函数中, 将不具有参数的ops写在forward中
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, input):

        h_relu = self.linear1(input).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred
N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = TwoLayerNet(D_in, H, D_out)
loss_fn = torch.nn.MSELoss(reduction="sum")
optim = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)

    optim.zero_grad()
    loss.backward()
    optim.step()
```
# Control Flow + Weight Sharing

为了更好的演示动态图和权重共享的特点, 我们会在下面实现一个非常奇怪的模型: 一个全连接的ReLU网络, 中间会随机的使用1~4层隐藏层, 并且重复利用相同的权重来计算最深层的隐藏层输出.

在PyTorch中, 我们可以通过for循环来实现这种动态模型, 并且通过重复调用同一个Module就可以很容易使用多层之间的参数共享, 如下所示:
```py
import random
import torch

class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        # 实现三个 nn.Linear 实例, 意味着在模型中只有 三个 nn.Linear 的参数
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        在PyTorch中, 我们可以通过for循环来随机的选择中间层的层数, 使得每一次
        执行forward函数时, 都具有不同的中间层层数. 而这些中间层都来自于同一个Module实例, 因而具有共享的权重参数.
        """
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0,3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred;

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
model = DynamicNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. Training this strange model with
# vanilla stochastic gradient descent is tough, so we use momentum
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
