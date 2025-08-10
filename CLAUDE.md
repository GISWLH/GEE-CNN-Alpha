---
description: 
globs: 
alwaysApply: true
---
# 项目规则与指南

## 总体规则

运行代码所需的环境在conda activate torch，因此运行py代码前先激活环境
timeout xx其中xx应该大于100分钟，因为代码运行较慢，可以多给一些时间
我不喜欢定义太过复杂的函数，并运行main函数，我是深度jupyter notebook用户，我喜欢直接的代码，简单的函数定义是可以接受的
使用matplotlib可视化，绘图使用Arial字体(在linux中手动增加我们的arial字体），绘图中的图片标记都用英文
数据都在data文件下，代码都在code文件下，图片都在image文件下

## 项目概述

本项目用于结合Sentinel-2的几个波段和最新AlphaEarth Embedding数据（高维特征数据集）来反演水体

使用CNN+GEE的方法，核心工作都是在本地完成的，我们先撰写geemap脚本提取多波段信息（Sentinel + AlphaEarth）

接下来用本地的水体标注点作为真值，提取的csv table的输入矩阵作为输入，输入到一个CNN卷积网络

再把CNN的参数传到GEE进行大尺度计算

## 预分析模式

由于Alpha Earth具有64个波段，因此先挑选出重要的波段来

这一步需要你根据全球的AlphaEarth（data\AlphaEarth），提取标注点（data/pnt.shp)上的Alpha1-Alpha64的值，导出一个csv （py脚本）

预分析模式第一步完成 ✅
已完成的任务：
✅ 成功读取了AlphaEarth数据（data/AlphaEarth/test2023.tif）
✅ 成功读取了标注点数据（data/pnt.shp）
✅ 在torch环境中正确运行了脚本
✅ 提取了12700个标注点上的Alpha1-Alpha64共64个波段的值
✅ 导出了CSV文件（data/alphaearth_extracted_values.csv）
生成的数据特点：
数据规模: 12700个样本点 × 131个特征列
AlphaEarth特征: Alpha1到Alpha64（64个波段）
原始属性: 包含landcover标签和其他遥感特征
坐标信息: 包含经纬度坐标
数据范围: AlphaEarth值在-0.42到0.38之间，数据质量良好
输出文件:  data/alphaearth_extracted_values.csv

第二步需要创建随机森林分类模型+SHAP分析找出最重要的变量，如Alpha1，Alpha6，。。只找出前5个即可。（py脚本）

  前5个最重要的Alpha波段：
  1. Alpha32 (SHAP重要性: 0.052501)
  2. Alpha64 (SHAP重要性: 0.042707)

  3. Alpha37 (SHAP重要性: 0.025922)
  4. Alpha48 (SHAP重要性: 0.020013)
  5. Alpha47 (SHAP重要性: 0.019665)

## 波段提取模式
接下来根据几个Sentinel-2的波段（["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"])
和AlphaEarth比较重要的波段[Alpha32, Alpha64, Alpha37, Alpha48, Alpha47]来提取数据(GEE是从A00-A63需要注意)
提取这几个波段的5*5窗口信息，具体提取代码在geeCNN_Water_sampleExport.ipynb
提取的结果保存在data下
注意Sentinel-2是日的，但AlphaEarth是年数据
此外，我们用asset的数据，但是数据较大，需要按random列提取，如(0,0.05)，(0.05, 0.1)。。。。提取20次

## GEE模式
接下来用GEE的大尺度计算来计算水体的大尺度计算
主要用code/code.js进行GEE云端运算，我们通过print(weights_and_biases['conv1.bias'])这种形式手动复制矩阵到GEE，这样就能在云端运行卷积了

## 2. 代码规范规则

- 使用Python脚本进行数据处理和自动化任务
- 代码文件命名规范：小写字母加下划线
- 模型参数保存在`model_parameters.txt`中
- 遵循PEP 8 Python代码风格指南
- 为复杂代码段添加清晰的注释和文档
- 代码尽量简单，避免复杂循环
- 不使用各种复杂函数定义方式，平铺撰写
- 读取遥感tif影像注意空值，mask=TRUE，注意避免极大极小值
- 使用rioxarray等高级库处理，而少用gdal等复杂的方式

## 3. 数据管理规则

- 原始数据存储在`data/`目录下
- 按类型组织处理后的数据
- 使用CSV格式存储表格数据
- 使用Shapefile格式存储空间数据


## 4. 文档管理规则

- 研究论文存储在`docs/`目录下
- 维护课程材料和教程文档
- 保持模型文档与代码同步
- 记录所有实验及其结果
- 在每个主要目录中包含README文件
- 维护重要更新的变更日志

## 5. 版本控制规则

- 使用Git进行版本控制
- 定期备份主要文件

## 6. 工作流程规则

### 6.1 数据准备
- 处理前验证数据质量
- 记录数据转换步骤

### 6.2 模型开发
- 将模型参数保存到`model_parameters.txt`
- 记录模型架构和超参数
- 跟踪模型性能指标

### 6.3 结果分析
- 使用Jupyter Notebook进行结果可视化
- 使用matplotlib可视化
- 绘图使用Arial字体
- 绘制图时图中内容用英文
- 生成评估指标和图表
- 记录分析方法与发现
- 与基准模型比较结果

### 6.4 文档更新
- 及时更新相关文档
- 记录实验过程和结果
- 维护研究日志
- 记录经验教训和最佳实践



















