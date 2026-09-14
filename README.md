# CIFAR-10 MatteaNet

这是一个用于 CIFAR-10 图像分类的自定义 CNN 项目。模型 `MatteaNet` 不直接调用 ResNet、VGG 等现成网络，而是使用 PyTorch 基础模块自行搭建。当前版本为 MatteaNet V2，核心结构是 Residual Block + SE Channel Attention。

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

## 2. MatteaNet V2 网络结构

总体结构：

```text
Input: 3 x 32 x 32
        ↓
Stem Conv 3 -> 64
        ↓
Stage 1: 2 x Residual-SE Block
64 x 32 x 32
        ↓
Stage 2: 2 x Residual-SE Block
128 x 16 x 16
        ↓
Stage 3: 3 x Residual-SE Block
256 x 8 x 8
        ↓
Stage 4: 2 x Residual-SE Block
384 x 4 x 4
        ↓
Global Average Pooling
384 x 1 x 1
        ↓
Flatten + Dropout
        ↓
Linear 384 -> 10
```

每个 `ResidualSEBlock` 的主分支大致为：

```text
input
  ↓
3x3 Conv
BatchNorm
SiLU
  ↓
3x3 Conv
BatchNorm
  ↓
SE Channel Attention
  ↓
Dropout2d
  ↓
+ shortcut
  ↓
SiLU
```

如果输入输出通道不同，或者需要下采样，shortcut 会使用 `1x1 Conv + BatchNorm` 对齐形状。

### 为什么比基础 CNN 更高级

- 使用 Residual Connection，使深层网络更容易训练，梯度可以通过 shortcut 更顺畅地传播。
- 使用 SE Channel Attention，让网络根据当前图片自适应地调整不同通道特征的重要程度。
- 使用 SiLU 激活函数，相比普通 ReLU 更平滑。
- 使用 stride=2 的卷积完成下采样，不再单独依赖 MaxPool。
- 使用 Dropout2d 和分类头 Dropout 减少过拟合。
- 使用 Global Average Pooling，降低全连接层参数量。
- 通道数逐渐增加：`64 -> 128 -> 256 -> 384`。
- 空间尺寸逐渐减小：`32 -> 16 -> 8 -> 4`。

这个网络借鉴了现代 CNN 中常见的设计思想，但 `ResidualSEBlock`、stage 数量、通道数和整体组合均在本项目中自行定义，没有直接调用现成 ResNet 或 SE-ResNet 模型。

## 3. 数据划分与实验规范

CIFAR-10 官方提供：

```text
50,000 张 training images
10,000 张 test images
```

本项目不会在训练过程中查看官方 test 集。

50,000 张官方 training images 采用固定分层划分，不使用随机种子：

```text
每个类别 5,000 张
├── 前 500 张   -> validation
└── 后 4,500 张 -> train
```

10 个类别合计：

```text
45,000 train
 5,000 validation
```

因此 train 和 validation 中 10 个类别数量完全均衡，并且每次运行得到的划分完全一致。

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

注意：数据集划分本身不随机，但训练 DataLoader 仍设置 `shuffle=True`，因此每个 epoch 会打乱训练样本顺序。训练集使用随机裁剪和随机水平翻转；validation 和 test 均不使用随机数据增强。

## 4. 项目结构

```text
.
├── model.py          # MatteaNet V2 网络定义
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

程序会打印：

```text
Input shape
Output shape
Parameters
Trainable params
```

正常输出张量形状应为：

```text
Input shape  : (4, 3, 32, 32)
Output shape : (4, 10)
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
scheduler  = CosineAnnealingLR
```

也可以手动指定参数：

```bash
python train.py --epochs 100 --batch-size 128 --lr 0.001
```

第一次运行时 torchvision 会自动下载 CIFAR-10。

训练启动时会打印实际划分：

```text
Dataset split: train=45000, val=5000
Split rule: fixed stratified split, 500 validation samples per class and 4500 training samples per class.
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
split = fixed_stratified
val_per_class = 500
```

## 8. 最终测试

训练结束且模型选择完成后运行：

```bash
python test.py
```

程序会加载 `checkpoints/best.pth`，然后只在官方 CIFAR-10 test split 上进行最终评估。

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

## 10. 后续实验方向

可以继续进行消融实验：

- 去掉 SE Attention
- 去掉 Residual shortcut
- 将 SiLU 改回 ReLU
- 减少 Stage 3 的 block 数量
- 修改通道宽度
- AdamW 与 SGD 对比
- 调整数据增强强度

这样可以分析 MatteaNet V2 中不同设计对 CIFAR-10 分类性能的影响。
