# 猫狗图片分类程序

## 简介

一个对猫狗图片分类的简单程序，用于代码实践练习。


## 文件目录

目录结构如下：

```
.
├── data_process.py
├── dataset.py
├── model.py
├── README.md
├── train.py
└── test.py
```

## 环境要求

```
Pillow==10.4.0
torch==2.3.1+cu121
torchvision==0.19.0
torchvision==0.18.1+cu121
tqdm==4.65.0
```

## 使用说明

原网站的数据集只分为`train`和`test`两部分。运行`data_process.py`从`train`中取出一部分作为验证集`val`，并将这两个目录中所有图片的路径和类别（存为整数，0表示`cat`，1表示`dog`）保存到`train.txt`和`test.txt`中。

运行`train.py`训练模型，训练过程中每个周期的权重保存在`./models/`中。训练过程中每个batch的损失将被记录在`train_record.csv`中。每个epoch结束后对验证集进行测试，每个epoch的`val loss`将记录在`val_record.csv`中。

运行`test.py`对测试集进行测试，在终端输入图片编号（1~12500之间的整数）后输出判断结果。非法输入将结束程序。下为该程序运行示例：

```
=== Enter an image id (an integer between 1 and 12500) ===
250   # your input
Classified as a photo of a dog.
251   # your input
Classified as a photo of a dog.
252   # your input
Classified as a photo of a cat.
0     # your input
Invalid input.
```

## 数据集

数据集来源于[Kaggle - Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)。

## 模型结构

模型网络结构如下：

```
Model(
  (conv1_1): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (conv1_2): Sequential(
    (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (1): ReLU()
    (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (conv2_1): Sequential(
    (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (conv2_2): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (1): ReLU()
    (2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): ReLU()
    (2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (conv3_2): Sequential(
    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (1): ReLU()
    (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (linear_1): Linear(in_features=200704, out_features=128, bias=True)
  (linear_2): Linear(in_features=128, out_features=1, bias=True)
  (sigmoid): Sigmoid()
)
```

