# 🚀 交互式服装检测 - 快速开始指南

## 🎯 您现在可以实现的效果

✅ **鼠标悬停自动高亮服装部位**  
✅ **点击锁定高亮效果**  
✅ **实时调节透明度和颜色**  
✅ **导出每个部位的独立掩码**  
✅ **获取详细的 JSON 元数据**  

---

## 📋 第一步：在 Colab 运行完整流程

### 1. 打开 Colab Notebook
```
https://colab.research.google.com/github/likunpeng0127/salf/blob/main/Colab_人体解析_完整版.ipynb
```

### 2. 选择 GPU
Runtime → Change runtime type → GPU

### 3. 依次运行所有单元格
- ✅ 步骤 1: 准备环境
- ✅ 步骤 2: 克隆项目（使用修复版代码）
- ✅ 步骤 3: 选择数据集（默认 LIP）
- ✅ 步骤 4: 下载模型
- ✅ 步骤 5: 上传图片
- ✅ 步骤 6: 运行推理
- ✅ **步骤 7.5: 提取服装部位** ⭐ **新功能**
- ✅ **步骤 7.6: 交互式高亮显示** ⭐ **新功能**
- ✅ 步骤 8: 下载结果

---

## 🎨 第二步：使用交互式控件

运行完步骤 7.6 后，您会看到：

### 📋 部位选择下拉菜单
```
选择部位: [无高亮 ▼]
          Hat (帽子) (2.3%)
          Hair (头发) (8.5%)
          Dress (连衣裙) (35.2%)
          Face (脸) (3.2%)
          Left-arm (左臂) (6.1%)
          ...
```

### 🎚️ 透明度滑块
```
透明度: [━━━━●━━━━━━] 0.6
```

### 🎨 颜色选择器
```
高亮颜色: 🟡 🔴 🟢 🔵 🔷 🟣
```

### 效果预览
- 选择 "Dress" → 连衣裙部分会用选定的颜色高亮
- 调节透明度 → 实时看到效果变化
- 换个颜色 → 高亮颜色立即改变

---

## 📂 第三步：查看输出文件

### 文件结构
```
parts_output/
├── batch_metadata.json          # 所有图片的元数据
└── c22b87643cfc465e8e4fcf233ea40bf1_parts/
    ├── metadata.json            # 单张图片的元数据
    ├── 02_Hair.png             # 头发掩码（黑白图）
    ├── 06_Dress.png            # 连衣裙掩码
    ├── 13_Face.png             # 脸部掩码
    ├── 14_Left-arm.png         # 左臂掩码
    ├── 15_Right-arm.png        # 右臂掩码
    ├── 16_Left-leg.png         # 左腿掩码
    ├── 17_Right-leg.png        # 右腿掩码
    ├── 18_Left-shoe.png        # 左鞋掩码
    └── 19_Right-shoe.png       # 右鞋掩码
```

### 掩码图片说明
- **白色区域**（255）= 该部位
- **黑色区域**（0）= 其他部位
- 可以直接用于图像处理、AI训练等

---

## 📊 第四步：使用元数据 (metadata.json)

### 示例内容
```json
{
  "image_name": "your_image",
  "dataset": "lip",
  "parts": [
    {
      "id": 6,
      "name": "Dress",
      "pixel_count": 125420,
      "percentage": 35.2,
      "mask_path": "your_image_parts/06_Dress.png"
    },
    {
      "id": 2,
      "name": "Hair",
      "pixel_count": 30250,
      "percentage": 8.5,
      "mask_path": "your_image_parts/02_Hair.png"
    }
  ],
  "clothing_parts": [
    // 只包含服装相关部位
    {
      "id": 6,
      "name": "Dress",
      "pixel_count": 125420,
      "percentage": 35.2,
      "mask_path": "your_image_parts/06_Dress.png"
    }
  ]
}
```

### 如何使用
```python
import json

# 读取元数据
with open('parts_output/batch_metadata.json') as f:
    data = json.load(f)

# 获取所有服装部位
for img in data:
    print(f"图片: {img['image_name']}")
    for part in img['clothing_parts']:
        print(f"  - {part['name']}: {part['percentage']:.1f}%")
```

---

## 🌐 第五步：在网页中使用

### 方式 1：下载并在本地打开

1. **下载所有文件**：
   ```bash
   # 在 Colab 中运行
   !zip -r results.zip inputs outputs parts_output
   
   from google.colab import files
   files.download('results.zip')
   ```

2. **解压后打开** `interactive_demo.html`

3. **修改代码加载您的数据**：
   ```javascript
   // 替换 demoData 为您的实际数据
   fetch('parts_output/batch_metadata.json')
       .then(response => response.json())
       .then(data => {
           // 使用数据
       });
   ```

### 方式 2：集成到您的网站

#### 基本示例
```html
<div class="clothing-viewer">
    <img id="original" src="your_image.jpg" />
    <canvas id="overlay"></canvas>
</div>

<div class="parts-list">
    <div data-part="6" onmouseenter="highlight(6)">连衣裙</div>
    <div data-part="14" onmouseenter="highlight(14)">左臂</div>
</div>

<script>
function highlight(partId) {
    // 1. 加载掩码图片
    const mask = new Image();
    mask.src = `parts_output/xxx_parts/${partId:02d}_*.png`;
    
    // 2. 在 canvas 上绘制高亮
    const canvas = document.getElementById('overlay');
    const ctx = canvas.getContext('2d');
    
    mask.onload = () => {
        ctx.globalAlpha = 0.6;
        ctx.fillStyle = 'rgba(255, 255, 0, 1)';
        // 根据掩码绘制
        ctx.drawImage(mask, 0, 0);
    };
}
</script>
```

---

## 💡 应用场景示例

### 场景 1：电商网站 - 服装详情页

```javascript
// 用户鼠标移动到连衣裙
onMouseMove: (event) => {
    const pixel = getPixelAt(event.x, event.y);
    const partId = parsing_result[pixel.y][pixel.x];
    
    if (partId === 6) {  // 连衣裙
        highlightPart(partId);
        showInfo({
            name: "优雅连衣裙",
            price: "¥299",
            material: "100% 棉"
        });
    }
}
```

### 场景 2：虚拟试衣间

```javascript
// 点击裙子区域
onClick: (partId) => {
    if (partId === 6) {  // 连衣裙
        // 显示可选颜色
        showColorPicker([
            { color: 'red', preview: 'dress_red.jpg' },
            { color: 'blue', preview: 'dress_blue.jpg' }
        ]);
    }
}
```

### 场景 3：AI 设计助手

```python
# 分析服装搭配
def analyze_outfit(metadata):
    clothing = metadata['clothing_parts']
    
    if 'Dress' in [p['name'] for p in clothing]:
        return "建议搭配高跟鞋和手包"
    elif 'Upper-clothes' in [p['name'] for p in clothing]:
        return "建议搭配裤子或裙子"
```

---

## 🔧 自定义和扩展

### 修改高亮颜色
```python
# 在 Colab notebook 中
color_dropdown.value = [0, 255, 128]  # 青绿色
```

### 合并多个部位
```python
# 合并上衣和外套
mask_upper = Image.open('parts_output/.../05_Upper-clothes.png')
mask_coat = Image.open('parts_output/.../07_Coat.png')

combined = np.logical_or(
    np.array(mask_upper), 
    np.array(mask_coat)
)
Image.fromarray(combined.astype(np.uint8) * 255).save('combined.png')
```

### 导出为视频
```bash
# 对视频每一帧处理
ffmpeg -i input.mp4 -vf fps=5 frames/%04d.jpg
python simple_extractor.py --input frames/ --output results/
# 再合成视频
```

---

## 🐛 常见问题

### Q: 步骤 7.5 报错 "No such file: extract_parts.py"
**A**: 确保运行了步骤 2（克隆项目）。或手动下载：
```bash
!wget https://raw.githubusercontent.com/likunpeng0127/salf/main/extract_parts.py
```

### Q: 掩码图片全黑
**A**: 该部位不存在于图片中。查看 metadata.json 的 pixel_count。

### Q: 交互式控件不显示
**A**: 确保先运行步骤 7.5 生成了 parts_output/ 目录。

### Q: 想要更精细的部位分割
**A**: 使用 ATR 数据集（18个部位），包含腰带、包包等：
```python
dataset = 'atr'  # 在步骤 3 修改
```

---

## 📚 完整文档

- **详细使用说明**: `INTERACTIVE_USAGE.md`
- **API 集成示例**: 见 INTERACTIVE_USAGE.md
- **HTML 模板**: `interactive_demo.html`
- **提取工具**: `extract_parts.py`

---

## 🎉 开始使用吧！

现在就打开 Colab，运行您的第一个交互式服装检测！

👉 **Colab 链接**: https://colab.research.google.com/github/likunpeng0127/salf/blob/main/Colab_人体解析_完整版.ipynb

有问题欢迎提 Issue: https://github.com/likunpeng0127/salf/issues

