#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
服装部位提取工具
提取人体解析结果中的每个部位，用于交互式服装检测
"""

import os
import numpy as np
from PIL import Image
import argparse
import json

# 数据集标签定义
DATASET_LABELS = {
    'lip': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
            'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
            'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe'],
    'atr': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
            'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf'],
    'pascal': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs']
}

# 服装相关的类别（可以高亮的部分）
CLOTHING_CATEGORIES = {
    'lip': {
        'Hat': 1, 'Upper-clothes': 5, 'Dress': 6, 'Coat': 7,
        'Socks': 8, 'Pants': 9, 'Jumpsuits': 10, 'Scarf': 11, 'Skirt': 12,
        'Left-shoe': 18, 'Right-shoe': 19
    },
    'atr': {
        'Hat': 1, 'Upper-clothes': 4, 'Skirt': 5, 'Pants': 6, 'Dress': 7, 'Belt': 8,
        'Left-shoe': 9, 'Right-shoe': 10, 'Bag': 16, 'Scarf': 17
    },
    'pascal': {
        'Torso': 2
    }
}


def extract_part_mask(parsing_result, part_id):
    """
    提取单个部位的二值掩码
    
    Args:
        parsing_result: 解析结果数组 (H, W)，每个像素值代表类别 ID
        part_id: 要提取的部位 ID
    
    Returns:
        二值掩码数组，该部位为 255，其他为 0
    """
    mask = np.zeros_like(parsing_result, dtype=np.uint8)
    mask[parsing_result == part_id] = 255
    return mask


def extract_all_parts(parsing_path, output_dir, dataset='lip'):
    """
    提取所有部位的掩码
    
    Args:
        parsing_path: 解析结果图片路径
        output_dir: 输出目录
        dataset: 数据集类型
    """
    # 读取解析结果
    parsing_img = Image.open(parsing_path)
    parsing_result = np.array(parsing_img)
    
    # 获取标签列表
    labels = DATASET_LABELS[dataset]
    clothing_cats = CLOTHING_CATEGORIES[dataset]
    
    # 获取图片名称
    img_name = os.path.basename(parsing_path).replace('.png', '')
    
    # 创建输出目录
    parts_dir = os.path.join(output_dir, img_name + '_parts')
    os.makedirs(parts_dir, exist_ok=True)
    
    # 保存元数据
    metadata = {
        'image_name': img_name,
        'dataset': dataset,
        'parts': [],
        'clothing_parts': []
    }
    
    # 提取每个存在的部位
    unique_ids = np.unique(parsing_result)
    print(f"📊 图片 {img_name} 包含 {len(unique_ids)} 个部位:")
    
    for part_id in unique_ids:
        if part_id >= len(labels):
            continue
            
        label = labels[part_id]
        
        # 提取掩码
        mask = extract_part_mask(parsing_result, part_id)
        
        # 计算该部位的像素数量
        pixel_count = np.sum(mask > 0)
        percentage = (pixel_count / mask.size) * 100
        
        print(f"  - {label} (ID: {part_id}): {pixel_count} 像素 ({percentage:.2f}%)")
        
        # 保存掩码
        mask_path = os.path.join(parts_dir, f'{part_id:02d}_{label.replace(" ", "_")}.png')
        Image.fromarray(mask).save(mask_path)
        
        # 添加到元数据
        part_info = {
            'id': int(part_id),
            'name': label,
            'pixel_count': int(pixel_count),
            'percentage': float(percentage),
            'mask_path': os.path.relpath(mask_path, output_dir)
        }
        metadata['parts'].append(part_info)
        
        # 标记服装类别
        if label in clothing_cats:
            metadata['clothing_parts'].append(part_info)
    
    # 保存元数据
    metadata_path = os.path.join(parts_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 完成！共提取 {len(metadata['parts'])} 个部位")
    print(f"   其中服装部位: {len(metadata['clothing_parts'])} 个")
    print(f"   输出目录: {parts_dir}")
    
    return metadata


def create_overlay_image(original_img_path, parsing_path, part_id, output_path, alpha=0.7):
    """
    创建叠加高亮的图片（用于交互式显示）
    
    Args:
        original_img_path: 原始图片路径
        parsing_path: 解析结果路径
        part_id: 要高亮的部位 ID
        output_path: 输出路径
        alpha: 透明度
    """
    # 读取图片
    original = Image.open(original_img_path).convert('RGBA')
    parsing_img = Image.open(parsing_path)
    parsing_result = np.array(parsing_img)
    
    # 调整解析结果大小到原图大小
    if parsing_result.shape != (original.height, original.width):
        parsing_result = np.array(Image.fromarray(parsing_result).resize(
            (original.width, original.height), Image.NEAREST
        ))
    
    # 创建高亮层
    highlight = np.zeros((original.height, original.width, 4), dtype=np.uint8)
    mask = parsing_result == part_id
    highlight[mask] = [255, 255, 0, int(255 * alpha)]  # 黄色高亮
    
    # 叠加
    highlight_img = Image.fromarray(highlight, 'RGBA')
    result = Image.alpha_composite(original, highlight_img)
    result.save(output_path)


def batch_extract(input_dir, output_dir, dataset='lip'):
    """
    批量处理目录中的所有解析结果
    
    Args:
        input_dir: 输入目录（包含解析结果 PNG）
        output_dir: 输出目录
        dataset: 数据集类型
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有 PNG 文件
    png_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    
    print(f"🔍 找到 {len(png_files)} 个解析结果文件\n")
    
    all_metadata = []
    for png_file in png_files:
        parsing_path = os.path.join(input_dir, png_file)
        print(f"处理: {png_file}")
        metadata = extract_all_parts(parsing_path, output_dir, dataset)
        all_metadata.append(metadata)
        print()
    
    # 保存批量元数据
    batch_metadata_path = os.path.join(output_dir, 'batch_metadata.json')
    with open(batch_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"🎉 全部完成！处理了 {len(png_files)} 张图片")
    print(f"批量元数据: {batch_metadata_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="服装部位提取工具")
    parser.add_argument('--input', type=str, required=True, help='解析结果图片或目录')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--dataset', type=str, default='lip', choices=['lip', 'atr', 'pascal'],
                        help='数据集类型')
    parser.add_argument('--batch', action='store_true', help='批量处理模式')
    
    args = parser.parse_args()
    
    if args.batch:
        batch_extract(args.input, args.output, args.dataset)
    else:
        extract_all_parts(args.input, args.output, args.dataset)

