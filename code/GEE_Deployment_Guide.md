# GEE-CNN-Alpha 部署指南

## 概述

本指南介绍如何将训练好的CNN水体检测模型部署到Google Earth Engine (GEE) 进行大尺度云端计算。

## 模型信息

- **模型架构**: 5×5窗口CNN，结合AlphaEarth和Sentinel-2特征
- **验证准确率**: 97.45%
- **输入特征**: 15个通道（5个AlphaEarth + 10个Sentinel-2波段）
- **输出**: 二分类（水体/非水体）
- **总参数数**: 18,466个

## 文件说明

### 1. AlphaGEE.js
主要的GEE JavaScript代码文件，包含：
- 数据获取和预处理
- CNN模型架构实现
- 结果可视化和导出

### 2. model_params_for_gee.js
完整的模型参数文件（较大），包含所有卷积层的权重和偏置。

### 3. gee_params_compact.js
紧凑格式的参数文件，更适合在GEE中使用。

## 部署步骤

### 步骤1: 准备参数文件

1. 打开 `code/gee_params_compact.js` 文件
2. 复制所有参数定义到 `AlphaGEE.js` 中对应位置
3. 替换占位符参数

### 步骤2: 配置数据源

在 `AlphaGEE.js` 中修改以下参数：

```javascript
// 时间参数
var year = 2022;  // 修改为目标年份
var start = year + '-09-01';  // 修改起始日期
var end = year + '-12-31';    // 修改结束日期

// ROI设置
var points = ee.FeatureCollection("projects/ee-wang/assets/pnt");  // 修改为您的资产路径
```

### 步骤3: AlphaEarth数据配置

确认使用的AlphaEarth波段：
```javascript
.select(['A31', 'A63', 'A36', 'A47', 'A46'])  // 前5重要波段
```

### 步骤4: 在GEE中运行

1. 登录 [Google Earth Engine Code Editor](https://code.earthengine.google.com/)
2. 创建新脚本
3. 复制完整的 `AlphaGEE.js` 代码
4. 确保参数已正确填入
5. 运行脚本

## 模型架构详解

### 网络结构
```
输入: (batch, 15, 5, 5)
├── Conv1: 15→16, 3×3, padding=0 → (batch, 16, 3, 3)
├── Conv2: 16→32, 3×3, padding=0 → (batch, 32, 1, 1)
├── 中心1×1提取: (batch, 15, 1, 1)
├── 特征连接: (batch, 47, 1, 1)
├── Conv3: 47→64, 1×1 → (batch, 64, 1, 1)
├── Conv4: 64→128, 1×1 → (batch, 128, 1, 1)
└── Conv5: 128→2, 1×1 → (batch, 2)
```

### 特征说明

**AlphaEarth特征** (5个波段):
- A31, A63, A36, A47, A46
- 通过SHAP分析选出的最重要特征

**Sentinel-2特征** (10个波段):
- B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
- 标准多光谱波段

## 性能优化建议

### 1. 内存管理
- 对于大区域，建议分块处理
- 使用适当的scale参数（建议90m，匹配AlphaEarth分辨率）

### 2. 计算效率
- 限制处理区域大小
- 使用质量掩膜过滤无效像素
- 考虑使用 `maxPixels` 参数

### 3. 导出设置
```javascript
Export.image.toDrive({
  image: waterMask,
  description: 'CNN_Water_Detection_' + year,
  folder: 'GEE_CNN_Alpha',
  region: roi,
  scale: 90,  // AlphaEarth分辨率
  maxPixels: 1e9
});
```

## 故障排除

### 常见问题

1. **内存超限**
   - 减小处理区域
   - 增加scale参数
   - 使用分块处理

2. **参数加载错误**
   - 检查参数格式是否正确
   - 确认所有参数都已填入
   - 验证数组维度

3. **数据访问错误**
   - 确认AlphaEarth数据集访问权限
   - 检查资产路径是否正确
   - 验证时间范围设置

### 调试技巧

1. **逐步测试**
   ```javascript
   print('Input image bands:', combinedImage.bandNames());
   print('Conv1 result:', conv1_result.bandNames());
   ```

2. **可视化中间结果**
   ```javascript
   Map.addLayer(conv1_result.select(0), {min: 0, max: 1}, 'Conv1 Output');
   ```

3. **检查数据范围**
   ```javascript
   print('Image statistics:', combinedImage.reduceRegion({
     reducer: ee.Reducer.minMax(),
     geometry: roi,
     scale: 90,
     maxPixels: 1e6
   }));
   ```

## 扩展应用

### 1. 多时相分析
修改时间参数进行不同时期的水体检测对比。

### 2. 大尺度制图
结合GEE的并行计算能力进行区域或全球尺度的水体制图。

### 3. 实时监测
设置定期任务进行水体变化监测。

## 联系支持

如有问题，请参考：
- GEE官方文档: https://developers.google.com/earth-engine
- 项目GitHub: https://github.com/GISWLH/GEE-CNN-Alpha

## 更新日志

- v1.0: 初始版本，支持AlphaEarth + Sentinel-2水体检测
- 模型准确率: 97.45%
- 支持5×5窗口CNN架构
