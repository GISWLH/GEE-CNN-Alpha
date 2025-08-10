#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建紧凑的GEE参数文件
将模型参数转换为更紧凑的格式，便于在GEE中使用
"""

import torch
import numpy as np
import json

def create_compact_params():
    """
    创建紧凑的参数格式
    """
    model_path = "../model/water_cnn_alpha_model.pth"
    
    print("加载模型参数...")
    model_state = torch.load(model_path, map_location='cpu')
    
    # 提取参数并转换为列表
    params = {}
    
    # Conv1
    conv1_weight = model_state['conv1.weight'].numpy()
    conv1_bias = model_state['conv1.bias'].numpy()
    
    # Conv2  
    conv2_weight = model_state['conv2.weight'].numpy()
    conv2_bias = model_state['conv2.bias'].numpy()
    
    # Conv3
    conv3_weight = model_state['conv3.weight'].numpy()
    conv3_bias = model_state['conv3.bias'].numpy()
    
    # Conv4
    conv4_weight = model_state['conv4.weight'].numpy()
    conv4_bias = model_state['conv4.bias'].numpy()
    
    # Conv5
    conv5_weight = model_state['conv5.weight'].numpy()
    conv5_bias = model_state['conv5.bias'].numpy()
    
    # 创建紧凑的JavaScript代码
    js_lines = []
    
    js_lines.append("// CNN模型参数 - 紧凑格式")
    js_lines.append("// 从PyTorch模型提取，验证准确率: 97.45%")
    js_lines.append("")
    
    # Conv1参数
    js_lines.append("// Conv1: 15 -> 16, kernel_size=3x3")
    
    # 将conv1_weight转换为紧凑格式
    conv1_w_flat = []
    for out_ch in range(conv1_weight.shape[0]):
        for in_ch in range(conv1_weight.shape[1]):
            kernel = conv1_weight[out_ch, in_ch].flatten().tolist()
            conv1_w_flat.extend(kernel)
    
    js_lines.append(f"var conv1_weight_flat = {json.dumps(conv1_w_flat)};")
    js_lines.append(f"var conv1_bias = {json.dumps(conv1_bias.tolist())};")
    js_lines.append("")
    
    # Conv2参数
    js_lines.append("// Conv2: 16 -> 32, kernel_size=3x3")
    
    conv2_w_flat = []
    for out_ch in range(conv2_weight.shape[0]):
        for in_ch in range(conv2_weight.shape[1]):
            kernel = conv2_weight[out_ch, in_ch].flatten().tolist()
            conv2_w_flat.extend(kernel)
    
    js_lines.append(f"var conv2_weight_flat = {json.dumps(conv2_w_flat)};")
    js_lines.append(f"var conv2_bias = {json.dumps(conv2_bias.tolist())};")
    js_lines.append("")
    
    # Conv3参数
    js_lines.append("// Conv3: 47 -> 64, kernel_size=1x1")
    conv3_w_flat = conv3_weight.reshape(conv3_weight.shape[0], -1).tolist()
    js_lines.append(f"var conv3_weight = {json.dumps(conv3_w_flat)};")
    js_lines.append(f"var conv3_bias = {json.dumps(conv3_bias.tolist())};")
    js_lines.append("")
    
    # Conv4参数
    js_lines.append("// Conv4: 64 -> 128, kernel_size=1x1")
    conv4_w_flat = conv4_weight.reshape(conv4_weight.shape[0], -1).tolist()
    js_lines.append(f"var conv4_weight = {json.dumps(conv4_w_flat)};")
    js_lines.append(f"var conv4_bias = {json.dumps(conv4_bias.tolist())};")
    js_lines.append("")
    
    # Conv5参数
    js_lines.append("// Conv5: 128 -> 2, kernel_size=1x1")
    conv5_w_flat = conv5_weight.reshape(conv5_weight.shape[0], -1).tolist()
    js_lines.append(f"var conv5_weight = {json.dumps(conv5_w_flat)};")
    js_lines.append(f"var conv5_bias = {json.dumps(conv5_bias.tolist())};")
    js_lines.append("")
    
    # 添加重构函数
    js_lines.append("// 重构3x3卷积权重的辅助函数")
    js_lines.append("function reshapeConv3x3Weights(flatWeights, outChannels, inChannels) {")
    js_lines.append("  var weights = [];")
    js_lines.append("  var idx = 0;")
    js_lines.append("  for (var out = 0; out < outChannels; out++) {")
    js_lines.append("    var outWeights = [];")
    js_lines.append("    for (var inp = 0; inp < inChannels; inp++) {")
    js_lines.append("      var kernel = [];")
    js_lines.append("      for (var k = 0; k < 9; k++) {")
    js_lines.append("        kernel.push(flatWeights[idx++]);")
    js_lines.append("      }")
    js_lines.append("      outWeights.push(kernel);")
    js_lines.append("    }")
    js_lines.append("    weights.push(outWeights);")
    js_lines.append("  }")
    js_lines.append("  return weights;")
    js_lines.append("}")
    js_lines.append("")
    
    js_lines.append("// 重构权重")
    js_lines.append("var conv1_weight = reshapeConv3x3Weights(conv1_weight_flat, 16, 15);")
    js_lines.append("var conv2_weight = reshapeConv3x3Weights(conv2_weight_flat, 32, 16);")
    
    # 保存到文件
    output_file = "gee_params_compact.js"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(js_lines))
    
    print(f"紧凑参数已保存到: {output_file}")
    
    # 打印统计信息
    print(f"\n参数统计:")
    print(f"Conv1: {conv1_weight.size + conv1_bias.size} 个参数")
    print(f"Conv2: {conv2_weight.size + conv2_bias.size} 个参数") 
    print(f"Conv3: {conv3_weight.size + conv3_bias.size} 个参数")
    print(f"Conv4: {conv4_weight.size + conv4_bias.size} 个参数")
    print(f"Conv5: {conv5_weight.size + conv5_bias.size} 个参数")
    
    total_params = (conv1_weight.size + conv1_bias.size + 
                   conv2_weight.size + conv2_bias.size +
                   conv3_weight.size + conv3_bias.size +
                   conv4_weight.size + conv4_bias.size +
                   conv5_weight.size + conv5_bias.size)
    print(f"总参数数: {total_params}")

if __name__ == "__main__":
    create_compact_params()
