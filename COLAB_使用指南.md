# 🚀 在 Colab 上成功运行的完整指南

## ✅ 已修复兼容性问题！

**好消息**：项目代码已经修复，现在**兼容最新版本的 PyTorch**！无需降级！

### 步骤 1：打开 Colab
1. 访问 https://colab.research.google.com/
2. 上传 `Colab_人体解析_完整版.ipynb`
3. **重要**: Runtime -> Change runtime type -> 选择 **GPU**

### 步骤 2：准备环境

```python
# 安装编译工具（Colab 已自带最新 PyTorch）
!pip install ninja
```

### 步骤 3：克隆您的项目

```python
!git clone https://github.com/likunpeng0127/salf.git
%cd salf
!mkdir -p checkpoints inputs outputs
```

### 步骤 4：下载预训练模型

```python
!pip install gdown
import gdown

# 选择数据集: 'lip', 'atr', 或 'pascal'
dataset = 'lip'

# 下载对应的模型
if dataset == 'lip':
    url = 'https://drive.google.com/uc?id=1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH'
elif dataset == 'atr':
    url = 'https://drive.google.com/uc?id=1ruJg4lqR_jgQPj-9K0PP-L2vJERYOxLP'
elif dataset == 'pascal':
    url = 'https://drive.google.com/uc?id=1E5YwNKW2VOEayK9mWCS3Kpsxf-3z04ZE'

gdown.download(url, 'checkpoints/final.pth', quiet=False)
```

### 步骤 5：上传测试图片

```python
from google.colab import files
import shutil

# 上传图片
uploaded = files.upload()

# 移动到 inputs 文件夹
for filename in uploaded.keys():
    shutil.move(filename, f'inputs/{filename}')
```

### 步骤 6：运行推理

```python
!python simple_extractor.py --dataset lip --model-restore checkpoints/final.pth --input-dir inputs --output-dir outputs
```

### 步骤 7：查看结果

```python
from PIL import Image
import matplotlib.pyplot as plt
import os

# 获取输出文件
output_files = [f for f in os.listdir('outputs') if f.endswith('.png')]
input_files = [f for f in os.listdir('inputs') if f.endswith(('.png', '.jpg'))]

# 显示对比
for inp, out in zip(sorted(input_files), sorted(output_files)):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(Image.open(f'inputs/{inp}'))
    axes[0].set_title('原图')
    axes[0].axis('off')
    
    axes[1].imshow(Image.open(f'outputs/{out}'))
    axes[1].set_title('解析结果')
    axes[1].axis('off')
    
    plt.show()
```

### 步骤 8：下载结果

```python
import shutil
from google.colab import files

# 打包并下载
shutil.make_archive('results', 'zip', 'outputs')
files.download('results.zip')
```

---

## 方法二：使用 PyTorch 1.5.1（最稳定）

如果上面的方法有问题，可以用项目原始版本：

```python
# 安装 PyTorch 1.5.1
!pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
!pip install ninja

# 其他步骤相同...
```

---

## 各数据集说明

### 📊 LIP (推荐，最全面)
- **20个标签**
- **mIoU: 59.36%**
- 最大数据集，50000+图片
- 标签：背景、帽子、头发、手套、太阳镜、上衣、连衣裙、外套、袜子、裤子、连体衣、围巾、裙子、脸、左臂、右臂、左腿、右腿、左鞋、右鞋

### 👔 ATR (适合时尚)
- **18个标签**
- **mIoU: 82.29%**
- 17000+图片，专注时尚AI
- 标签：背景、帽子、头发、太阳镜、上衣、裙子、裤子、连衣裙、腰带、左鞋、右鞋、脸、左腿、右腿、左臂、右臂、包、围巾

### 🏃 Pascal-Person-Part (简化版)
- **7个标签**
- **mIoU: 71.46%**
- 3000+图片，专注身体部位
- 标签：背景、头部、躯干、上臂、下臂、大腿、小腿

---

## 💡 常见问题

### Q: 为什么要降低 PyTorch 版本？
**A**: 代码使用了旧版 PyTorch API（如 `tensor.type()` 和 `tensor.data<T>()`），新版本（2.0+）已废弃这些 API，会导致编译失败。使用 1.5-1.7 版本完全兼容。

### Q: 可以在 Mac 本地运行吗？
**A**: 可以，但需要：
1. Mac 没有 CUDA，只能用 CPU（会很慢）
2. 或者用 Apple Silicon 的 MPS 加速（需要修改代码）
3. **推荐直接用 Colab 的免费 GPU**

### Q: 报错 "RuntimeError: CUDA out of memory"？
**A**: 图片太大或太多。解决方法：
- 减少同时处理的图片数量
- 降低图片分辨率
- 或在 Colab 升级到更大的 GPU（付费）

### Q: 可以批量处理吗？
**A**: 可以！把所有图片放在 `inputs` 文件夹，程序会自动批量处理。

---

## 🎯 完整的一键运行代码

复制以下代码到 Colab 新建的 notebook 中：

```python
# ============== 一键运行脚本 ==============

# 1. 安装依赖
!pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
!pip install ninja gdown

# 2. 克隆项目
!git clone https://github.com/likunpeng0127/salf.git
%cd salf
!mkdir -p checkpoints inputs outputs

# 3. 下载模型（LIP）
import gdown
gdown.download('https://drive.google.com/uc?id=1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH', 
               'checkpoints/final.pth', quiet=False)

# 4. 上传图片
from google.colab import files
import shutil
uploaded = files.upload()
for f in uploaded.keys():
    shutil.move(f, f'inputs/{f}')

# 5. 运行推理
!python simple_extractor.py --dataset lip --model-restore checkpoints/final.pth --input-dir inputs --output-dir outputs

# 6. 显示结果
from PIL import Image
import matplotlib.pyplot as plt
import os

for inp in sorted(os.listdir('inputs')):
    if inp.endswith(('.png', '.jpg', '.jpeg')):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(Image.open(f'inputs/{inp}'))
        ax[0].set_title('原图')
        ax[0].axis('off')
        
        out = inp.replace('.jpg', '.png').replace('.jpeg', '.png')
        ax[1].imshow(Image.open(f'outputs/{out}'))
        ax[1].set_title('解析结果')
        ax[1].axis('off')
        plt.show()

# 7. 下载结果
shutil.make_archive('results', 'zip', 'outputs')
files.download('results.zip')
```

---

## ✅ 总结

**最简单的方法**：
1. 在 Colab 新建 notebook
2. 选择 GPU 运行时
3. 复制上面的"一键运行代码"
4. 运行即可！

**预计时间**：
- 安装依赖：2-3分钟
- 下载模型：1-2分钟
- 处理图片：每张图 1-3秒

祝您使用愉快！🎉

