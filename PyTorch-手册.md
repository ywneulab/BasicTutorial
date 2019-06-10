---
title: PyTorch 手册
sitemap: true
categories: PyTorch
date: 2018-06-02 18:43:48
tags:
- PyTorch
---

PyTorch主要提供以下两大特色:
- 支持强力GPU加速的Tensor计算能力
- 基于tape的具有自动微分求导能力的深度神经网络框架

PyTorch 主要包含以下组成要素:

| 组成要素 | 描述说明 |
| --- | --- |
| torch  | 一个类似于numpy的tensor哭, 提供强力的GPU支持 |
| torch.autograd  | 一个基于tape的具有自动微分求导能力的库, 可以支持几乎所有的tesnor operatioin |
| torch.nn | 一个神经网络库, 与autograd深度整合, 可以提供最大限度的灵活性 |
| torch.multiprocessing | Python的多线程处理, 可以提供torch Tensors之间的内存共享, 对于加载数据和Hogwild training来说十分有用 |
| torch.utils | 一些功能类和函数, 如DataLoader, Trainer等等 |
| torch.legacy(.nn/.optim) | 为了兼容性而存在的一些代码和实现 |

Pytorch通常可以作为以下用途使用:
- 为了使用GPUs性能的numpy替代品
- 可以提供强大灵活力和速度优势的深度学习平台.




# torch

## backends.cudnn

```py
torch.backends.cudnn.benchmark = True
```
上述设置可以让内置的`cudnn`的`auto-tuner`自动寻找最合适当前配置的搞笑算法, 来达到优化运行效率的目标, 在使用时, 应该遵循以下两个准则:
1. 如果网络的输入数据维度或类型上变化不大, 则该设置可以增加运行效率
2. 如果网络的输入数据在每次的`iteration`中都变化的话, 会导致`cudnn`每次都寻找一遍最优配置, 这样反而 **会降低** 运行效率.

## torch.cat()
```py
torch.cat(seq, dim=0, out=None) # 返回连接后的tensor
```
将给定的 tensor 序列 `seq` 按照维度连接起来. 默认维度为0, 说明会将其在第 0 个维度上进行拼接.(最后的结果是第 0 维度增大, 例如三个2行3列的 tensor 按照第0维度拼接, 最后得到的 tensor 维度为6行3列)


## clamp()/clamp_()
```py
torch.clamp(input, min, max, out=None) -> Tensor
```
将input里面元素全部划分到[min,max]区间内, 小于min的置为min, 大于max的置为max. 如果不指定`min`或者`max`,则认为无下界或上界

**其他调用形式:**
```py
torch.Tensor(min, max) # 调用tensor为input, 返回值为out
```


## device()
```py
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```
## gather()
```py
torch.gather(input, dim, index, out=None) -> Tensor
```
沿着`dim`指定的轴按着`index`指定的值重新组合成一个新的tensor.

```py
out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
```

即假设`input`是一个 n 维的tensor, 其 size 为 $(x_0, x_1, ..., x_{i-1}, x_i, x_{i+1},..., x_{n-1})$, 若`dim=i`, 则 `index` 必须也是一个 n 维的tensor, 其 size 为 $(x_0, x_1, ..., x_{i-1}, y, x_{i+1},..., x_{n-1})$, 其中 $y\geq 1$, 而返回的 tensor `out`  的 size 和 `index` 的 size 相同.
一句来说 gather 的作用就是, 在指定的维度上筛选给给定下标`index`指示的值, 其他值舍弃.

**一个例子说明:**
scores是一个计算出来的分数，类型为[torch.FloatTensor of size 5x1000]
而y_var是正确分数的索引，类型为[torch.LongTensor of size 5]
容易知道，这里有1000个类别，有5个输入图像，每个图像得出的分数中只有一个是正确的，正确的索引就在y_var中，这里要做的是将正确分数根据索引标号提取出来。
```py
scores = model(X_var)  # 分数
scores = scores.gather(1, y_var.view(-1, 1)).squeeze()  #进行提取
```
提取后的scores格式也为[torch.FloatTensor of size 5]
这里讲一下变化过程：
1、首先要知道之前的scores的size为[5,1000]，而y_var的size为[5]，scores为2维，y_var为1维不匹配，所以先用view将其展开为[5,1]的size，这样维数n就与scroes匹配了。
2、接下来进行gather，gather函数中第一个参数为1，意思是在第二维进行汇聚，也就是说通过y_var中的五个值来在scroes中第二维的5个1000中进行一一挑选，挑选出来后的size也为[5,1]，然后再通过squeeze将那个一维去掉，最后结果为[5].


**Tensor形式:**
```py
torch.Tensor.gather(dim, index) -> Tensor
```

## torch.ge()

## torch.gt()
```py
torch.gt(input, other, out=None) # -> Tensor
```
根据 input 和 other 的值返回一个二值 tensor, 如果满足大于条件则为1, 不满足则为0.
other 可以是能够转换成 input size 的tensor, 也可以是一个 `float` 标量.

## torch.index_select()
```py
torch.index_select(input, dim, index, out=None) # -> Tensor
```
返回在 `dim` 维度上的 `index` 指明的下标组成的 tensor.
返回的 tensor 的维度的数量和 `input` 是相同的, 但是第 `dim` 维度的 size 会和 `index` size大小相同. 其他维度的 size 保持不变.

## torch.le()
```py
torch.le(input, other, out=None) # ->Tensor
```
按元素计算 $input \leq other$.

## max()
```py
torch.max(input) # 返回一个Tensor, 代表所有元素中的最大值

torch.max(input,dim,keepdim=False,out=None) # 返回一个元组:(Tensor, LongTensor)
```
第二种形式会返回一个元组, 元组内元素类型为: (Tensor, LongTensor), 其中, 前者代表对应 dim 上 reduce 后的最大值, 后者代表最大值在维度 `dim` 中对应的下标.
如果`keepdim=True`, 则输出的 tensor 的 size 会和输入的相同, 只不过对应 `dim` 维度上的size为1. 否则, 对应 `dim` 维度会被 squeeze/reduce, 使得输出的维度比输入的维度少1.
```py
>>> a = torch.randn(4, 4)
>>> a
tensor([[-1.2360, -0.2942, -0.1222,  0.8475],
        [ 1.1949, -1.1127, -2.2379, -0.6702],
        [ 1.5717, -0.9207,  0.1297, -1.8768],
        [-0.6172,  1.0036, -0.6060, -0.2432]])
>>> torch.max(a, 1)
(tensor([ 0.8475,  1.1949,  1.5717,  1.0036]), tensor([ 3,  0,  0,  1]))
```

## mm()
**注意, 没有`torch.mm_`版本**
```py
torch.mm(mat1, mat2, out=None) # 返回值为Tensor, 也可以使用out记录返回值
```
两矩阵相乘, 矩阵的size需要满足乘法规则

**其他调用形式:**
```py
torch.Tensor(mat2) # 调用者为mat1
```



## norm()
返回输入tensor的p-norm标量
```py
torch.norm(input, p=2) # 返回一个标量tensor
```

## numel()
```py
torch.numel(input)  #返回一个int值
```
返回 `inpput` tensor 中的元素的总个数
```py
a = torch.randn(1,2,3,4,5)
print(torch.numel(a)) # 120
```

## ones()

## randn()
标准正太分布随机基础, 传入参数为维度信息

## torch.sort()
```py
torch.sort(input, dim=None, descending=False, out=None) # 返回 (Tensor, LongTensor)
```
如果没有给定维度 `dim`, 则会默认选择最后一个维度.

## sum()
```py
torch.sum(input, dtype=None) # 返回求和后的Tensor(只有一个元素)

torch.sum(input, dim, keepdim=False, dtype=None) # 返回在dim上reduce的sum和, 如果dim包含多个维度, 则都进行reduce求和.
# reduce这个词很形象, 因为返回的Tensor的维度刚好没有了dim指示的那些维度
```

**其他形式:**
```py
torch.Tensor.sum()
```

## torch.t()
```py
torch.t(input) # 返回转置后的Tensor
```

**其他形式:**
```py
torch.Tensor.t()
```

## unsqueeze()
在指定维度上插入一个 singleton 维度(一般用于将单一数据处理用 batch 的形式)
```py
torch.unsqueeze(input, dim, out=None) # -> Tensor
```
**返回的tensor与input tensor 共享数据**

dim 的取值范围在 [-input.dim()-1, input.dim()+1] 之间, 如果为负值, 则相当于 `dim = dim + input.dim() + 1`.

## zeros()

# torch.cuda

## torch.cuda.empty_cache()

释放所有未使用的 GPU 内存, 使用这些内存可以被其他 GPU 应用使用, 并且可以被 `nvidia-smi` 查到.

`empty_cache()` 并不会强制提升供 PyTorch 使用的显卡内存的大小, 查看[Memory management](https://pytorch.org/docs/master/notes/cuda.html#cuda-memory-management)

# torch.Tensor

`torch.Tensor` 是默认类型 `torch.FloatTensor` 的别名, 使用 `torch.Tenosr` 的构造函数创建 tensor 变量时, 传入的是维度信息(注意与 `torch.tensor()` 的区别):
```py
t = torch.Tensor(2,3,4) # 里面的数值未初始化, 是随机的
print(t.size()) # torch.Size([2,3,4])
```
`torch.LongTesnor` 使用方法相似, 只不过数据类型是长整型.

## troch.tensor()
创建tensor
```py
torch.tensor(data, dtype=None, device=None, requires_grad=False)
```
可以利用`torch.tensor`从python的list数据或者其他序列数据中创建tensor对象
```py
torch.tensor([[1,-1],[1,-1]])
torch.tensor(np.array([[1,2,3],[4,5,6]]))
```
**注意, `torch.tensor()`函数总是会对数据进行复制操作, 因此, 如果你仅仅是想将数据的`requires_grad`标志改变, 那么就应该使用`required_grad_()`或者`detach()`函数来避免复制. 同时, 对numpy数组使用`torch.as_tensor()`将其转换成tensor而无需复制**

## torch.Tensor.cpu()
```py
torch.Tensor.cpu()
z = x.cpu()
```
将tensor移动到cpu上, 注意返回值`z`是cpu上的数据, tensor`x`本身的device属性不变

## torch.Tensor.cuda()
```py
torch.Tensor.cuda()
z = x.cuda()
```

## torch.Tensor.dim()
```py
torch.Tensor.dim() -> int
```
返回 tensor 的维度的个数.


## torch.Tensor.max()
```py
torch.Tensor.max(dim=None, keepdim=False) -> Tensor or (Tensor, Tensor)
```
详情见 `torch.max()`


## torch.Tensor.numel()
```py
torch.Tensor.numel()
```
详见 `torch.numel()`

## torch.Tensor.to()
```py
torch.Tensor.to(*args, *kwargs)
```
返回一个转移后的tensor, 而自身维持不变
```py
t = torch.randn(2,3)
t.to(torch.float64)
t.to(device)
t.to("cuda:0")
```

将tensor移动到gpu上, 注意返回值`z`是gpu上的数据, tensor`x`本身的device属性不变

## torch.Tensor.numpy()
tensor与numpy数组的转换

```py
torch.Tensor.numpy() # 返回tensor对应的numpy数组

torch.from_numpy(ndarray) # 将numpy数组ndarray转换成对应的tensor并返回.
```

**torch.Tensor** 实际上是 **torch.FloatFensor** 的别名

## torch.Tensor.permute()
重新排列tensor的维度
```py
torch.Tensor.permute(*dims) # 返回一个重新排列维度后的 tensor
```

## torch.Tensor.unsqueeze()

详细可见`torch.unsqueeze`

## torch.Tensor.expand()
```py
torch.Tensor.expand(*sizes) # 返回 tensor
```
将 tensor 中的 singleton 维度扩展到一个更大的 size.
参数 `-1` 意味着不改变原始的维度
新增的维度的元素被被添加到前头, size不能设置为-1.
expand 并没有申请新的内存, 而仅仅是在当前已经存在的 tensor 上面创建了新的视图(view), 使得 singleton 维度被扩展成了一个更大的尺寸.
Any dimension of size 1 can be expanded to an arbitrary value without new memory.
```py
x = torch.tensor([1],[2],[3])
print(x.size())  # torch.Size([3,1])
print(x.expand(3,4)) # torch.Size([3,4]) # 将维度为1的扩展到任意尺寸
print(x.expand(-1,4)) # torch.Size([3,4]) # -1 代表不改变维度
```
注意, 只能对 singleton 的维度进行扩展, 如果强行对其他维度扩展, 则会报错.

## torch.Tensor.expand_as()
```py
torch.Tensor.expand_as(other) # 返回 tensor
```
将当前 tensor 扩展到和 `other` 一样的size.
`self.expand_as(other)` 与 `self.expand(other.size())` 等价.

## torch.Tensor.index_fill_()
```py
torch.Tensor.index_fill_(dim, index, val) # 返回tensor
```
在给定的维度 `dim` 上, 用 `val` 将该维度上的 `index` 坐标的值填充.
```py
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
index = torch.tensor([0, 2])
x.index_fill_(1, index, -1)
print(x)
#tensor([[-1.,  2., -1.],
#       [-1.,  5., -1.],
#       [-1.,  8., -1.]])
```

## torch.Tensor.contiguous()
返回一个连续的tensor, 数据内容不变
```py
torch.Tensor.contiguous() # 如果tensor本身就是连续的, 那么就会返回tensor本身
```
这里的 `contiguous` 指的是内存上的连续, 由于在 PyTorch 中, `view` 只能用在 `contiguous` 的 tensor 上面, 而如果在 `view` 之前使用了 `transpose`, `permute`等操作后, 就需要使用 `contiguous` 来返回一个 contiguous tensor.

**在 PyTorch 0.4 版本以后, 增加了 `torch.reshape()`, 这与 `numpy.reshape()` 的功能类似, 它大致相当于 `tensor.contiguous().view()` ?**


## torch.Tensor.item()

当Tensor中只包含一个元素时, 可以利用该函数返回这个元素的标量

## torch.Tensor.tolist()

可以将Tensor转换成列表

## torch.Tensor.zero_()

```py
torch.Tensor.zero_()
```
将当前的 tensor 变量全部置为0(原地)



# torch.autograd

## set_grad_enabled()
```py
class torch.autograd.set_grad_enabled(mode)
```
用来控制梯度计算的开关(依据bool类型参数`mode`决定), 可以当做上下文管理器使用, 也可以当做函数使用
```py
# 当做上下文管理器
with torch.set_grad_enabled(is_train): # 注意, 这里省略了autograd
    loss.backward()
    optimizer.step()

# 当做函数使用
w1 = torch.Tensor([1], requires=True)
torch.set_grad_enabled(True)
print(w1.requires_grad) # True
torch.set_grad_enabled(False)
print(w1.requires_grad) # False
```

## no_grad()
```py
class torch.autograd.no_grad
```
用于禁用梯度计算的上下文管理器.
在测试阶段, 当你确信你不会调用`Tensor.backward()`时,禁用梯度计算十分有用. 这会降低计算使用内存消耗.
```py
x = torch.tensor([1.0], requires_grad=True)
with torch.no_grad(): # 省略了autograd
    print(x.requires_grad) # True, 虽然为True, 但在该上下文中, 会无视掉requires_grad参数, 一律做False处理
    y = x*2  
    print(y.requires_grad) # False, 在当前上下文产生的tensor的requires_grad属性为False
print(x.requires_grad) # True
```

## torch.autograd.Function
```py
class torch.autograd.Function
```
为可微分的 ops 记录 operation history, 同时定义计算公式.

每一个作用在 tensor 上的 operatin 都会创建一个新的 function 对象, 它会执行计算过程并记录相关信息. 这些信息可以从一个由 functions 组成的有向图中获得. 当 `backward()` 方法被调用时, 就会利用这些信息在 function 上进行反向传播, 并将梯度传给下一个 Funtion.
通常情况下, 当用于需要自定义可自动求导的 ops 时, 可以实现一个 Function 的子类.
```py
# Example
class Exp(Function):

    @staticmethod
    def forward(ctx, i):
        result = i.exp()
        ctx.save_for_backward(result)

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output*result
```

**static forward(ctx, *args, **kwargs):**
定义前向计算的逻辑.

**static backward(ctx, *grad_outputs):**
定义反向传导的逻辑, 如果确定不会使用到反向传播, 则可以不实现该函数.


# torch.nn

## Module

```py
class torch.nn.Module
```
所有神经网络Module的基类, 自定义的模型也应该是它的子类.
Modules可以包含其他Module(如Linear, Conv2d等等).

### parameters()
```py
for param in model.parameters():
    print(param.data, param.size())
```

**state_dict**:
```py
torch.nn.Module.state_dict(destination=None,prefix="",keep_vars=False)
```
以字典形式返回整个module的状态

**train**
```py
torch.nn.Module.train(mode=True)
```
将module的模式设置为train, 这只对部分module有效, 如Dropout, BatchNorm等, 详细请查看官网.
返回值: torch.nn.Module

**training**
```py
torch.nn.Module.training # 属性, 返回一个bool值, 指示当前的模式是否为train
```

**eval**
```py
torch.nn.Module.eval() # 注意, 和train不同, eval为无参函数
```
将module的mode设置为evaluation, 同样, 只对部分module起效.

## Linear
```py
torch.nn.Linear(in_features, out_features, bias=True)
```
全连接层的实现. 输入的shape为 $(N, ..., in_features)$, 输出的shape为 $(N,..., out_features)$, 可以看出, 除了最后一维不同外, 其他维度都相同. (通常在使用Linear之前, 会将输入变成二维的矩阵, 其中第一维为batch size, 第二维为特征向量).

**in_features** 和 **out_features** 可以当做属性用`.`来获取.

## Conv2d
```py
class torch.nn.Conv2的(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
```

- **in_channels(`int`)**:
- **out_channels(`int`)**:
- **kernel_size(`int`or`tuple`)**:
- **stride(`int`or`tuple`, optional)**:

## MaxPool2d

## Softmax()
```py
class torch.nn.Softmax(dim=None)
```
`dim`指明了需要进行 softmax 的维度, 在这个维度上的值, 加起来和为1.

## ReLU
```py
torch.nn.ReLU(inplace=False)
```
输入输出的shape是相同的, 执行relu函数


## torch.nn.Sequential
```py
class torch.nn.Sequential(*args)
```

## torch.nn.MSELoss
```py
class torch.nn.MSELoss(size_average=None, reduce=None, reduction="elementwise_mean")
```
- **size_average(bool, optional):** 弃用(见reduction参数). 默认情况下, loss会计算在每个样本上的平均误差. 如果将size_average置为False, 则计算平方误差总和. **当reduce参数为False时, 忽视该参数**
- **reduce(bool, optional):** 弃用(见reduction参数). reduce参数顾名思义, 就是是否让MSELoss函数返回值的维度减少, 默认为True, 即会将任意维度的输入计算loss后, 返回一个标量(平均or总和取决于size_average), 如果为False, 则说明返回值维度不应该发生变化, 故而返回值就是对每个元素单独进行平方损失计算.
```py
y = torch.tensor([1,2,3,4], dtype=torch.float)
pred_y = torch.tensor([1,1,1,1], dtype=torch.float)
loss_fn1 = torch.nn.MSELoss()
loss1 = loss_fn1(y, pred_y)
loss_fn2 = torch.nn.MSELoss(size_average=False)
loss2 = loss_fn2(y, pred_y)
loss_fn3 = torch.nn.MSELoss(reduce=False)
loss3 = loss_fn3(y, pred_y)
print(loss1,loss2,loss3)
# tensor(3.5000) tensor(14.) tensor([0., 1., 4., 9.])
```
- **reduction(string, optional):** 用字符串来替代上面两个参数的作用: "elementwise_mean"(默认) | "sum" | "none" (不进行reduce).

## torch.nn.functional

### conv1d()

### conv2d()

### relu()
```py
torch.nn.functional.relu(input, inplace=True) # 返回 一个 Tenosr
```

### relu_()
```py
torch.nn.functional.relu_(input) # relu() 的原地版本
```

# torch.optim

## lr_scheduler

**StepLR**
```py
class torch.optim.lr_schedulr.StepLR(optimizer,step_size,gamma=0.1,last_epoch=-1)
```
每经过step_size次epoch之后, lr就会衰减gamma倍(new_lr=lr×gamma), 初始的lr来自于optimizer中的lr参数.
```py
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
```
**ExponentialLR**
```py
class torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma,last_epoch=-1)
```

**CosineAnnealingLR**
```py
```

## Adam
```py
class torch.optim.Adam(params,lr=0.001,betas=(0.9,0.999),eps=1e-08,weight_decay=0,amsgrad=False)
```



## conv2d

# torch.utils.data

## DataLoader
```py
class torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False,sampler=None,batch_sampler=None,num_workers=0,collate_fn=<function default_collate>,pin_memory=False,drop_last=False,timeout=0,worker_init_fn=None)
```
数据加载器, 将数据集和采样器结合起来, 并且提供单/多线程的迭代器.
- `dataset(utils.data.Dataset)`:
- `batch_size(int,optional)`: batch中的样本个数
- `shuffle(bool,optional)`
- `num_worker(int,optional)`: 加载数据的线程个数, 0意味着只有一个主线程.


**方法:**
- `__iter__(self)`: 可以当做迭代器使用, 如`inputs,class_ids=next(iter(dataloaders))`, 其中, `input`的shape为 $(N, C, H, W)$, `class_ids`的shape为 $(N)$.
- `__len__(self)`: 返回数据集的类别数目

# torchvision

## torchvision.utils

### make_grid
```py
torchvision.utils.make_grid(tensor,nrow=8,padding=2,normalize=False,range=None,scale_each=False,pad_value=0)
```
制作一个关于image的grid, 返回值依然是一个tensor, 只不过尺度变成了3D, 相当于把多个图片拼接在一起了, 直接通过`plt.imshow(grid)`即可输出网格化以后的图片.

- `tensor(Tensor/list)`: 4D的 mini-batch Tensor, Shape为 $(N×C×H×W)$, 或者是同维度的list.



## torchvision.transforms

### torchvision.transforms.Compose
```py
class torchvision.transforms.Compose(transforms)

# 使用
trans.Compose([
    transforms.CenterCrop(10),
    transforms.ToTensor(),
])
```
将多个transforms操作组合起来, 注意参数是列表形式

### Transforms on PIL Image

```py
# cv2 image to PIL Image

# skimage to PIL Image
```
**注意, 以下操作作用在PIL Image上的**

**CenterCrop**
```py
class torchvision.transform.CenterCrop(size)
```
`size`参数表示输出的图谱的大小, 如果只传入了一个数字, 则该数字既表示高度, 又表示宽度.

**Resize**
```py
class torchvision.transforms.Resize(size, interpolation=2)
```
- `size`: 期望的输出size.
- `interpolation`: 插值方法, 默认为双线性插值

**ToTensor**
```py
class torchvision.transforms.ToTensor
```
将一个`PIL Image`或者`numpy.ndarray` (H×W×C,[0, 255])转换成torch.FloatTensor (C×H×W, [0.0, 1.0]).

**RandomHorizontalFlip**
```py
transforms.RandomHorizontalFlip(p=0.5)
```
在给定概率下对PIL Image随机执行水平翻转操作

**RandomResizedCrop**
```py
torch.transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333), interpolation=2)
```
对PIL Image随机执行剪裁操作(按照scale和ratio的区间剪裁), 然后将剪裁后的图片放缩都期望的尺寸(默认插值为双线性插值)
- size: 期望得到的尺寸
- scale: 剪裁的面积比例(相对于原始图)
- ratio: 剪裁的宽高比
- interpolation: 默认为:PIL.Image.BILINEAR

## Transforms on torch.*Tensor

**注意, 以下操作是作用在tensor上的**

**Normalize**
```py
class torchvision.transforms.Normalize(mean, std)
```
将图片tensor按照均值mean和标准差std进行归一化, 对于n个channels, 有 mean=(M1, ..., Mn), std=(S1,...,Sn).
**注意, 这个归一化操作是原地进行的**

## torchvision.datasets

### ImageFolder

```py
class torchvision.datasets.ImageFolder(root, transform=None, target_transform=None, loader=<function default_loader>)
```
一个一般化的数据加载器, 主要针对如下数据排列格式:
```py
root/dog/x.png
root/dog/y.png
root/dog/z.png
...
root/cat/123.png
root/cat/nsdf3.png
root/cat/asd932_.png
```
- `root`: 根目录路径
- `transform(callable,optional)`: 对图片要做的变换操作
- `target_transform(callable,optional)`: 对target要做的变换操作
- `loader`: 用于加载给定路径图片的函数

**属性:**
- `classes(list)`: 返回类别的名字列表 class_names
- `class_to_idx(dict)`: 以字典的形式返回(class_name, class_index)
- `imgs(list)`: 返回元组列表: (image path, class_index)

**方法:**
- **__getitem__(index):** 根据index返回(sample,target)元组. 可以使用
- **`len(imagefolder)`** 返回类别数量


# sort()#
```py
sort(dim=None, descending=False)  # 默认为升序, 返回(Tensor, LongTensor)
```
详见 `torch.sort()`


# torch.distributed

## torch.distributed.reduce()


# inspect 模块

```py
inspect.signature() # 查看函数签名, python3.6以上
inspect.getargspec() # 查看函数签名, python3.6以上
inspect.getsource() # 获取模型的code
inspect.getabsfile() # 获取模块的路径

```

# un normalize
https://github.com/pytorch/vision/issues/528

```py
mean = torch.tensor([1, 2, 3], dtype=torch.float32)
std = torch.tensor([2, 2, 2], dtype=torch.float32)

normalize = T.Normalize(mean.tolist(), std.tolist())

unnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
```
