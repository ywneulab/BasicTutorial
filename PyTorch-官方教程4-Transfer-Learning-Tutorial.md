---
title: PyTorch官方教程(四)-Transfer_Learning_Tutorial
sitemap: true
categories: PyTorch
date: 2018-07-08 18:51:27
tags:
- PyTorch
---

通常情况下, 我们不会从头训练整个神经网络, 更常用的做法是先让模型在一个非常大的数据集上进行预训练, 然后将预训练模型的权重作为当前任务的初始化参数, 或者作为固定的特征提取器来使用. 既通常我们需要面对的是下面两种情形:
- **Finetuning the convnet:** 在一个已经训练好的模型上面进行二次训练
- **ConvNet as fixed feature extractor:** 此时, 我们会将整个网络模型的权重参数固定, 并且将最后一层全连接层替换为我们希望的网络层. 此时, 相当于是将前面的整个网络当做是一个特征提取器使用.

# Load Data

我们将会使用`torch.utils.data`包来载入数据. 我们接下来需要解决的问题是训练一个模型来分类蚂蚁和蜜蜂. 我们总共拥有120张训练图片, 具有75张验证图片.

```py
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # 注意转换成tensor后, 像素会变成[0,1]之间的浮点数
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
}

data_dir = "hymenoptera_data"
# from torchvision import datasets
image_datasets = {x:datasets.ImageFolder(root=os.path.join(data_dir, x),
                        transform=data_transforms[x])
                        for x in ["train", "val"]}
dataloaders = {x:torch.utils.data.DataLoader(image_datasets[x]), batch_size=4, shuffle=True, num_workers=4)
                            for x in ["train", "val"]}
dataset_sizes = {x:len(image_datasets[x]) for x in ["train", "val"]}
class_names = image_datasets["train"].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

## Visualize a few images
```py
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

inputs, class_ids = next(iter(dataloaders["train"])) # 获取一个batch
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in class_ids])
```

# Training the model

接下来, 让我们定义一个简单的函数来训练模型, 我们会利用LR scheduler对象`torch.optim.lr_scheduler`设置lr scheduler, 并且保存最好的模型.

```py
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(epoch)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs,1) # preds代表最大值的坐标, 相当于获取了最大值对应的类别
                    loss = criterion(outputs, labels)

                    if phase = "train": # 只有处于train模式时, 来更新权重
                        loss.backward()
                        optimizer.step()
                # 统计状态
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds==labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(phase, epoch_loss, epoch_acc)

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(time_elapsed)
    print(best_acc)

    # load best model weights
    model.load_state_dic(best_model_wts)
    return model
```

## Visualizing the model predictions

下面的代码用于显示预测结果

```py
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad(): # 不计算梯度
        for i, (inputs, labels) in enumerate(dataloaders["val"]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs,1)

            for j in range(inputs.size()[0]): # 或者batch size
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis("off")
                ax.set_title(class_names[preds[j]])
                imshow(inputs.cpu().data[j]) # 由于imshow不能作用在gpu的数据上, 因此需要先将其移动到cpu上.

                if images_so_far == num_images:
                    model.train(mode = was_training)
                    return
        model.train(mode=was_training)
```
# FineTuning the convnet

加载预训练模型, 并重置最后一层全连接层

```py
# from torchvisioin import models
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()


# 这里是让所有的参数都进行更新迭代
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
```

## Train and evaluate
调用刚刚定义的训练函数对模型进行训练
```py
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

visualize_model(model_ft)
```

# Convnet as Fixed Feature Extractor

假设我们需要将除了最后一层的其它层网络的参数固定(freeze), 为此, 我们需要将这些参数的`requires_grad`属性设置为`False`.

```py
model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# 将最后一层fc层重新指向一个新的Module, 其内部参数的requires_grad属性默认为True
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs,2)

model_conv = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
```

## Train and evaluate

```py
model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_eopch=25)
visualize_model(model_conv)
```
