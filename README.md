# CIFAR-10 MatteaNet

这是一个用于 CIFAR-10 图像分类的入门项目。模型 `MatteaNet` 不直接使用 ResNet、VGG 等现成网络，而是使用 PyTorch 基础模块自行搭建。

## 1. 任务

CIFAR-10 一共有 10 个类别，每张图片大小为 `32 x 32`，共有 3 个 RGB 通道。

模型输入：

```text
[B, 3, 32, 32]
```

模型输出：

```text
[B, 10]
```

其中 10 个输出分别对应 CIFAR-10 的 10 个类别。

## 2. 网络结构

MatteaNet 一共包含 3 个卷积阶段：

```text
Input: 3 x 32 x 32

Stage 1
Conv 3 -> 32
BatchNorm
ReLU
Conv 32 -> 32
BatchNorm
ReLU
MaxPool
        ↓
32 x 16 x 16

Stage 2
Conv 32 -> 64
BatchNorm
ReLU
Conv 64 -> 64
BatchNorm
ReLU
MaxPool
        ↓
64 x 8 x 8

Stage 3
Conv 64 -> 128
BatchNorm
ReLU
Conv 128 -> 128
BatchNorm
ReLU
MaxPool
        ↓
128 x 4 x 4

Adaptive Average Pooling
        ↓
128 x 1 x 1

Flatten
Dropout
Linear 128 -> 10
```

设计思路：

- 空间尺寸逐渐减小：`32 -> 16 -> 8 -> 4`
- 通道数量逐渐增大：`3 -> 32 -> 64 -> 128`
- 前面的卷积负责提取局部特征，后面的卷积学习更高级的特征
- 使用 BatchNorm 让训练更稳定
- 使用全局平均池化减少全连接层参数量
- 最后一层输出 10 个 logits，对应 10 个类别

## 3. 数据划分与实验规范

CIFAR-10 官方提供：

```text
50,000 张 training images
10,000 张 test images
```

本项目不会在训练过程中查看官方 test 集。

50,000 张官方 training images 会使用固定随机种子拆分为：

```text
45,000 train
 5,000 validation
```

默认随机种子：

```text
seed = 2026
```

训练过程中：

```text
train
  ↓
更新网络参数
  ↓
validation
  ↓
根据 val_acc 选择 best.pth
```

训练完成后才运行 `test.py`：

```text
best.pth
  ↓
official CIFAR-10 test set
  ↓
final test accuracy
```

这样可以避免用 test set 反复选择模型造成测试集泄漏。

训练集使用随机裁剪和随机水平翻转；validation 和 test 均不使用随机数据增强。

## 4. 项目结构

```text
.
├── model.py          # MatteaNet 网络定义
├── train.py          # train / validation 与 best checkpoint 选择
├── test.py           # 官方 test 集最终评估
├── requirements.txt
├── .gitignore
└── README.md
```

训练后会自动生成：

```text
checkpoints/best.pth
```

CIFAR-10 数据集会自动下载到：

```text
data/
```

`data/` 和 `checkpoints/` 默认不会提交到 GitHub。

## 5. 安装依赖

```bash
pip install -r requirements.txt
```

建议使用带 CUDA 的 PyTorch 环境训练。

## 6. 检查网络是否能正常前向传播

```bash
python model.py
```

正常情况下输出张量形状应为：

```text
Input shape : (4, 3, 32, 32)
Output shape: (4, 10)
```

## 7. 开始训练

```bash
python train.py
```

默认参数：

```text
epochs     = 50
batch size = 128
optimizer  = AdamW
lr         = 0.001
val size   = 5000
seed       = 2026
scheduler  = CosineAnnealingLR
```

也可以手动指定参数：

```bash
python train.py --epochs 100 --batch-size 128 --lr 0.001 --val-size 5000 --seed 2026
```

第一次运行时 torchvision 会自动下载 CIFAR-10。

训练启动时会打印实际划分：

```text
Dataset split: train=45000, val=5000, seed=2026
Official CIFAR-10 test set is reserved for test.py only.
```

每个 epoch 显示：

```text
train_loss
train_acc
val_loss
val_acc
```

当 validation accuracy 刷新时，会自动保存：

```text
checkpoints/best.pth
```

checkpoint 中记录：

```text
model weights
best_val_acc
epoch
val_size
seed
```

## 8. 最终测试

训练结束且模型选择完成后运行：

```bash
python test.py
```

程序会加载：

```text
checkpoints/best.pth
```

然后只在官方 CIFAR-10 test split 上进行最终评估，并输出：

```text
selected checkpoint epoch
best validation accuracy
final test loss
final test accuracy
```

## 9. 训练核心流程

```text
images
  ↓
MatteaNet forward
  ↓
10-class logits
  ↓
CrossEntropyLoss
  ↓
loss.backward()
  ↓
optimizer.step()
```

其中：

- `loss.backward()` 计算梯度
- `optimizer.step()` 根据梯度更新卷积核和其他可训练参数
- validation 只负责选择模型，不参与梯度更新
- official test set 只用于最终报告成绩

## 10. 后续可以继续尝试

完成基础版本后，可以继续进行消融实验，例如：

- 去掉 BatchNorm 比较准确率
- 去掉数据增强比较准确率
- 改变通道数量，例如 `16 -> 32 -> 64`
- 增加第四个卷积阶段
- 使用 SGD 替换 AdamW
- 加入自己实现的 Residual Block

这样可以观察网络结构和训练策略对 CIFAR-10 分类性能的影响。
