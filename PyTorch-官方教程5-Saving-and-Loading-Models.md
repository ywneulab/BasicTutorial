---
title: PyTorch官方教程(五)-Saving and Loading Models
sitemap: true
categories: PyTorch
date: 2018-07-09 21:34:33
tags:
- PyTorch
---

本篇教程提高了大量的用例来说明如何保存和加载PyTorch models. 在介绍细节之前, 需要先熟悉下面的三个函数:
- `torch.save`: 保存一个序列化对象(serialized object)到磁盘中. 该函数使用的是Python的`pickle`工具完成序列化的. Models, tensors, 以及由各种对象所组成的字典数据都可以通过该函数进行保存.
- `torch.load`: 使用`pickle`的解包工具(unpickling facilities)来反序列化 pickled object 到内存中. 该函数同样可以操作设备(device)来加载数据
- `torch.nn.Module.load_state_dict`: 利用非序列结构数据`state_dict`加载模型的参数字典.

# What is a `state_dict`?

在PyTorch中, `torch.nn.Module` 模型中的可更新的参数(weighs and biases)在保存在模型参数中(`model.parameters()`). 而`state_dict`是一个典型的python字典数据, 它将每一层和层中的参数tensor相互关联起来. 注意到, 只有那些具有可更新参数的层才会被保存在模型的`state_dict`数据结构中. 优化器对象(Optimizer object-`torch.optim`)同样也可以拥有`state_dict`数据结构, 它包含了优化器的相关状态信息(超参数等). 下面看一个`state_dict`的简单例子.

**调用时, 要使用 `.state_dict()` 来获得字典结构`**

```py
# Define model
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 个人建议最好将relu也写在__init__函数内, 否则无法通过模型获知到底使用了什么激活函数(只有通过forward函数才能知道)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = TheModelClass()

# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
```

# Saving & Loading Model for Inference

## Save/Load `state_dict`(Recommended)

**Save:**
```py
torch.save(model.state_dict(), PATH)
```

**Load:**
```py
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
```
当为inference阶段保存模型时, 仅仅保存训练好的模型的可更新参数即可. 利用`torch.save()`函数来保存模型的`state_dict`可以在之后恢复模型时提供极大的灵活性, 这也是我们推荐使用该方法来保存模型的原因.

**模型的保存文件通常以`.pt`或者`.pth`结尾**

请牢记在执行inference逻辑之前使用了`model.eval()`来将当前的模式转换成测试模式, 不然的话dropout层和BN层可能会产生一些不一致的测试结果.

**请注意, `load_state_dict()`函数接受的参数是一个字典对象, 而不是模型文件的保存路径. 这意味着你必须先将模型文件解序列成字典以后, 才能将其传给`load_state_dict()`函数**

## Save/Load Entire Model

**Save:**
```py
torch.save(model, PATH)
```

**Load:**
```py
# Model class must be defined somewhere
model = torch.load(PATH)
model.eval()
```
这段保存/加载的流程使用了最直观的语法以及最少的代码. 以这种方式保存模型时将会用`pickle`模块把 **整个** 模型序列化保存. 这种方法的坏处是序列化的数据会和特定的classes绑定, 以及模型保存时固定的目录结构(这句话啥意思?). 造成这种结果的原因在于`pickle`没有保存模型本身, 而是保存了一个包含类的文件的路径. 因此, 这样的代码会在之后应用到其他工程时以各种方式造成程序崩溃.

# Saving & Loading a General Checkpoint for Inference and/or Resuming Training

**Save:**
```py
torch.save({"epoch":epoch,
            "model_state_dict":model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(),
            "loss": loss,
            ...
            }, PATH)
```

**Load:**
```py
model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
epoch = checkpoint["epoch"]
loss = checkpoint["loss"]

model.eval()
# --or--
model.train()
```

当保存一个通用的 checkpoint 文件时, 我们不仅仅需要保存模型的 `state_dict` 信息, 还需要保存一些其他信息. 为此, 我们需要将这些信息组织成字典的形式, 然后利用 `torch.save()` 函数进行保存. 通常情况下, 在PyTorch中, 这些checkpoints文件使用 `.tar` 文件后缀.  在加载模型时, 首先要记得初始化模型, 然后利用 `torch.load()` 函数来你所需要的各项数据.

# Saving Multiple Models in One File

**Save:**
```py
torch.save({
            "modelA_state_dict": modelA.state_dict(),
            "modelB_state_dict": modelB.state_dict(),
            "optimizerA_state_dict": optimizerA.state_dict(),
            "optimizerB_state_dict": optimizerB.state_dict(),
            ...
            }, PATH)
```

**Load:**
```py
modelA = TheModelAClass(*args, **kwargs)
modelB = TheModelBClass(*args, **kwargs)
optimizerA = TheOptimizerAClass(*args, **kwargs)
optimizerB = TheOptimizerBClass(*args, **kwargs)

checkpoint = torch.load(PATH)
modelA.load_state_dict(checkpoint["modelA_state_dict"])
modelB.load_state_dict(checkpoint["modelB_state_dict"])
optimizerA.load_state_dict(checkpoint["optimizerA_state_dict"])
optimizerB.load_state_dict(checkpoint["optimizerB_state_dict"])

modelA.eval()
modelB.eval()
# - or -
modelA.train()
modelB.train()
```

当需要保存多个不同的模型时(如RNN, CNN), 可以用同样的方式将这些模型的 `state_dict` 信息保存起来, 并将它们组织成字典的形式, 然后利用`torch.save()`将他们序列化保存起来, 通常情况下文件以`.tar`后缀命名.

# Warmstarting Model Using Parameters from a Different Model

**Save:**
```py
torch.save(modelA.state_dict(), PATH)
```

**Load:**
```py
modelB = ThemodelBClass(*args, **kwargs)
modelB.load_state_dict(torch.load(PATH), strict=False)
```

Partially loading a model 或者 loading a partial model 在迁移学习或者训练一个复杂模型时是很常见的, 即使只是用很小一部分参数, 也可以起到训练过程的热启动效果, 进而可以帮助模型更快的收敛.
不论何时, 当你需要从 partial `state_dict` 中加载模型时, 都需要将参数 `strict` 设置为 False.

# Saving & Loading Model Across Devices

## Save on GPU, Load on CPU

**Save:**
```py
torch.save(model.state_dict(), PATH)
```

**Load:**
```py
device = torch.device("cpu")
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))
```

通过 `torch.load()` 函数的 `map_location` 参数来指定将模型的 `state_dict` 加载到哪个设备上.

## Save on GPU, Load on GPU

**Save:**
```py
torch.save(model.state_dict(), PATH)
```

**Load:**
```py
device = torch.device("cuda:0")
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)
```

使用 `.to` 来将模型中的参数tensor转移到GPU设备上, 需要注意的是, 在进行训练或者预测之间, 还需要调用 tensor 的 `.to()` 方法来将 tensor 也转移到 GPU 设备上, 另外, 注意, `mytensor.to(device)` 实际上是在 GPU 中创建了 `mytensor` 的副本, 而并没有改变 `mytensor` 的值, 因此, 需要写成这样的形式来是的 `mytensor` 的值改变: `my_tensor = my_tensor.to(torch.device("cuda"))`

## Save on CPU, Load on GPU

**Save:**
```py
torch.save(model.state_dict(), PATH)
```

**Load:**
```py
device = torch.device("cuda")
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))
model.to(device)
```

由于模型是在CPU上存储的, 因此在模型加载时, 需要设置 `torch.load()` 函数的 `map_location` 参数为 `cuda:0`. 然后, 还需要调用 `model` 的 `.to(device)` 方法来将model的参数 tensor 全部转移到 GPU 上去, 另外别忘了将数据也要转移到 GPU 上去, `my_tensor = my_tensor.to(torch.device("cuda"))`.

## Saving `torch.nn.DataParallel` Models

**Save:**
```py
torch.save(model.module.state_dict(), PATH)
```

**Load:**
```py
# Load to whatever device you want
```
