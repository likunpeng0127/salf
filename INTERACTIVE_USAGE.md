# 🎨 交互式服装区域检测 - 使用指南

## 📋 功能概述

这套工具可以帮您实现：
1. ✅ **提取每个服装部位的单独掩码**
2. ✅ **生成元数据（JSON）**，包含每个部位的信息
3. ✅ **在 Colab 中交互式高亮显示**（下拉选择 + 实时预览）
4. ✅ **网页端交互式演示**（鼠标悬停高亮）

---

## 🚀 快速开始

### 方式 1：在 Colab 中使用

1. **运行完整的 notebook**：
   ```
   打开 Colab_人体解析_完整版.ipynb
   依次运行所有单元格
   ```

2. **步骤 7.5** 会自动提取所有部位掩码：
   ```python
   !python extract_parts.py --input outputs --output parts_output --dataset lip --batch
   ```

3. **步骤 7.6** 会显示交互式控件：
   - 下拉菜单选择部位
   - 滑块调节透明度
   - 颜色选择器

### 方式 2：命令行使用

#### 单张图片处理
```bash
python extract_parts.py \
    --input outputs/your_image.png \
    --output parts_output \
    --dataset lip
```

#### 批量处理
```bash
python extract_parts.py \
    --input outputs/ \
    --output parts_output \
    --dataset lip \
    --batch
```

---

## 📂 输出文件结构

运行后会生成以下文件：

```
parts_output/
├── batch_metadata.json              # 批量元数据
└── your_image_parts/                # 单张图片的部位文件夹
    ├── metadata.json                # 该图片的元数据
    ├── 00_Background.png            # 背景掩码
    ├── 02_Hair.png                  # 头发掩码
    ├── 06_Dress.png                 # 连衣裙掩码
    ├── 13_Face.png                  # 脸部掩码
    ├── 14_Left-arm.png              # 左臂掩码
    ├── 15_Right-arm.png             # 右臂掩码
    ├── 16_Left-leg.png              # 左腿掩码
    ├── 17_Right-leg.png             # 右腿掩码
    ├── 18_Left-shoe.png             # 左鞋掩码
    └── 19_Right-shoe.png            # 右鞋掩码
```

---

## 📊 元数据格式 (metadata.json)

```json
{
  "image_name": "c22b87643cfc465e8e4fcf233ea40bf1",
  "dataset": "lip",
  "parts": [
    {
      "id": 6,
      "name": "Dress",
      "pixel_count": 125420,
      "percentage": 35.2,
      "mask_path": "c22b87643cfc465e8e4fcf233ea40bf1_parts/06_Dress.png"
    },
    {
      "id": 2,
      "name": "Hair",
      "pixel_count": 30250,
      "percentage": 8.5,
      "mask_path": "c22b87643cfc465e8e4fcf233ea40bf1_parts/02_Hair.png"
    }
    // ... 更多部位
  ],
  "clothing_parts": [
    // 只包含服装相关部位
  ]
}
```

---

## 🌐 在网页中使用

### 方案 1：使用提供的 HTML 模板

1. **打开 `interactive_demo.html`**
2. **替换数据源**：
   ```javascript
   // 加载您的元数据
   fetch('parts_output/batch_metadata.json')
       .then(response => response.json())
       .then(data => {
           // 使用数据渲染界面
       });
   ```

3. **加载掩码图片**：
   ```javascript
   // 加载部位掩码
   const mask = new Image();
   mask.src = 'parts_output/your_image_parts/06_Dress.png';
   ```

### 方案 2：集成到现有项目

#### React 示例
```jsx
import { useState } from 'react';

function ClothingHighlight({ imageSrc, metadata }) {
    const [activePart, setActivePart] = useState(null);
    
    return (
        <div>
            <img src={imageSrc} />
            {activePart && (
                <HighlightOverlay 
                    maskSrc={activePart.mask_path} 
                    alpha={0.6} 
                />
            )}
            
            <PartsList 
                parts={metadata.clothing_parts}
                onHover={setActivePart}
            />
        </div>
    );
}
```

#### Vue 示例
```vue
<template>
  <div class="clothing-viewer">
    <img :src="imageSrc" />
    <canvas ref="overlay" />
    
    <div class="parts-list">
      <div 
        v-for="part in parts" 
        :key="part.id"
        @mouseenter="highlightPart(part)"
        @mouseleave="clearHighlight"
      >
        {{ part.name }}
      </div>
    </div>
  </div>
</template>

<script>
export default {
  methods: {
    highlightPart(part) {
      // 在 canvas 上绘制高亮
      const ctx = this.$refs.overlay.getContext('2d');
      // 加载掩码并绘制
    }
  }
}
</script>
```

---

## 💡 应用场景

### 1. 电商服装展示
- 鼠标悬停自动高亮服装部位
- 点击查看详情和价格
- 多角度切换

### 2. 虚拟试衣间
- 选择不同服装部位
- 实时替换颜色/款式
- AI 推荐搭配

### 3. 服装设计工具
- 快速选择编辑区域
- 自动分离不同材质
- 批量修改颜色

### 4. 图像标注工具
- 自动预标注
- 人工微调
- 导出训练数据

---

## 🎨 自定义高亮颜色

### 在 Python 中
```python
# 修改 extract_parts.py 中的颜色
highlight_color = [255, 200, 0]  # 橙色
alpha = 0.7
```

### 在 JavaScript 中
```javascript
// 修改 interactive_demo.html
highlightColor = [0, 255, 128];  # 青绿色
alpha = 0.5;
```

---

## 📱 响应式设计

HTML 模板已内置响应式支持：
- 桌面端：左右分栏显示
- 移动端：上下堆叠显示
- 自适应图片大小

---

## 🐛 常见问题

### Q: 掩码图片全黑？
**A**: 检查该部位是否存在于图片中。查看 `metadata.json` 的 `pixel_count`。

### Q: 如何合并多个部位？
**A**: 加载多个掩码图片，使用 OR 运算合并：
```python
combined_mask = mask1 | mask2 | mask3
```

### Q: 如何导出为 JSON API？
**A**: `metadata.json` 已经是标准 JSON 格式，可以直接作为 API 响应。

### Q: 支持实时视频吗？
**A**: 可以！对视频每帧运行推理，然后使用提取工具。建议降低帧率（如每秒5帧）。

---

## 🔗 API 集成示例

### RESTful API
```python
from flask import Flask, jsonify, send_file
import os

app = Flask(__name__)

@app.route('/api/parts/<image_id>')
def get_parts(image_id):
    # 读取元数据
    metadata_path = f'parts_output/{image_id}_parts/metadata.json'
    with open(metadata_path) as f:
        return jsonify(json.load(f))

@app.route('/api/mask/<image_id>/<part_id>')
def get_mask(image_id, part_id):
    # 返回掩码图片
    mask_path = f'parts_output/{image_id}_parts/{part_id:02d}_*.png'
    return send_file(mask_path, mimetype='image/png')
```

### GraphQL
```graphql
type Part {
  id: Int!
  name: String!
  pixelCount: Int!
  percentage: Float!
  maskUrl: String!
}

type Image {
  id: String!
  parts: [Part!]!
  clothingParts: [Part!]!
}

type Query {
  image(id: String!): Image
}
```

---

## 📚 更多资源

- **原始论文**: [Self-Correction for Human Parsing](https://arxiv.org/abs/1910.09777)
- **项目地址**: https://github.com/likunpeng0127/salf
- **Colab Demo**: [在线运行](https://colab.research.google.com/github/likunpeng0127/salf/blob/main/Colab_人体解析_完整版.ipynb)

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目遵循原始项目的许可证。

