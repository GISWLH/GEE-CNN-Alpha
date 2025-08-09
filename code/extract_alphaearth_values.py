#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取AlphaEarth数据在标注点位置的值
根据全球的AlphaEarth（data/AlphaEarth），提取标注点（data/pnt.shp)上的Alpha1-Alpha64的值，导出一个csv
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
from pathlib import Path

# 设置数据路径
data_dir = Path("data")
alphaearth_file = data_dir / "AlphaEarth" / "test2023.tif"
points_file = data_dir / "pnt.shp"
output_file = data_dir / "alphaearth_extracted_values.csv"

print("开始提取AlphaEarth数据...")
print(f"AlphaEarth文件: {alphaearth_file}")
print(f"标注点文件: {points_file}")

# 检查文件是否存在
if not alphaearth_file.exists():
    print(f"错误: AlphaEarth文件不存在: {alphaearth_file}")
    exit(1)

if not points_file.exists():
    print(f"错误: 标注点文件不存在: {points_file}")
    exit(1)

# 读取标注点数据
print("读取标注点数据...")
points_gdf = gpd.read_file(points_file)
print(f"标注点数量: {len(points_gdf)}")
print(f"标注点坐标系: {points_gdf.crs}")
print("标注点属性列:", list(points_gdf.columns))

# 如果标注点没有坐标系，根据lon/lat列判断应该是WGS84
if points_gdf.crs is None:
    print("标注点文件没有坐标系信息，根据lon/lat列设置为EPSG:4326")
    points_gdf = points_gdf.set_crs("EPSG:4326")

# 读取AlphaEarth数据
print("读取AlphaEarth数据...")
try:
    # 使用rioxarray读取，但不立即加载到内存，使用chunks进行分块处理
    alphaearth_data = rxr.open_rasterio(alphaearth_file, chunks={'band': 1, 'x': 1000, 'y': 1000})
    print(f"AlphaEarth数据形状: {alphaearth_data.shape}")
    print(f"AlphaEarth坐标系: {alphaearth_data.rio.crs}")
    print(f"波段数量: {alphaearth_data.sizes['band']}")

    # 检查数据范围，使用小样本避免内存问题
    print("检查数据基本信息...")
    sample_data = alphaearth_data.isel(x=slice(0, 100), y=slice(0, 100)).compute()
    # 将数据展平来检查有效值
    flat_data = sample_data.values.flatten()
    valid_data = flat_data[~np.isnan(flat_data)]
    if len(valid_data) > 0:
        print(f"样本数据范围: {float(valid_data.min())} 到 {float(valid_data.max())}")
    else:
        print("警告: 样本区域全部为NaN值，但继续处理...")

except Exception as e:
    print(f"读取AlphaEarth数据时出错: {e}")
    exit(1)

# 确保坐标系一致
if points_gdf.crs != alphaearth_data.rio.crs:
    print(f"转换标注点坐标系从 {points_gdf.crs} 到 {alphaearth_data.rio.crs}")
    points_gdf = points_gdf.to_crs(alphaearth_data.rio.crs)

# 提取点坐标
print("提取点坐标...")
x_coords = points_gdf.geometry.x.values
y_coords = points_gdf.geometry.y.values

print(f"坐标范围: X({x_coords.min():.2f}, {x_coords.max():.2f}), Y({y_coords.min():.2f}, {y_coords.max():.2f})")

# 在每个点位置提取AlphaEarth值
print("在标注点位置提取AlphaEarth值...")
extracted_values = []

for i, (x, y) in enumerate(zip(x_coords, y_coords)):
    if i % 100 == 0:
        print(f"处理进度: {i}/{len(x_coords)}")

    try:
        # 使用sel方法选择最近的像素点，并立即计算值
        point_values = alphaearth_data.sel(x=x, y=y, method='nearest').compute()

        # 检查是否有有效值
        if not np.isnan(point_values).all():
            # 转换为numpy数组并处理空值
            values = point_values.values
            # 将NaN替换为0或其他合适的值
            values = np.where(np.isnan(values), 0, values)
            extracted_values.append(values)
        else:
            # 如果所有值都是NaN，用0填充
            extracted_values.append(np.zeros(alphaearth_data.sizes['band']))

    except Exception as e:
        print(f"在点 ({x}, {y}) 处提取值时出错: {e}")
        # 用0填充
        extracted_values.append(np.zeros(alphaearth_data.sizes['band']))

print("提取完成，准备数据...")

# 转换为numpy数组
extracted_array = np.array(extracted_values)
print(f"提取的数据形状: {extracted_array.shape}")

# 创建列名 (Alpha1, Alpha2, ..., Alpha64)
num_bands = extracted_array.shape[1]
alpha_columns = [f"Alpha{i+1}" for i in range(num_bands)]

# 创建DataFrame
result_df = pd.DataFrame(extracted_array, columns=alpha_columns)

# 添加原始标注点的属性
for col in points_gdf.columns:
    if col != 'geometry':
        result_df[col] = points_gdf[col].values

# 添加坐标信息
result_df['x_coord'] = x_coords
result_df['y_coord'] = y_coords

print(f"最终数据形状: {result_df.shape}")
print("数据列:", list(result_df.columns))

# 保存为CSV
print(f"保存结果到: {output_file}")
result_df.to_csv(output_file, index=False)

print("数据提取完成!")
print(f"输出文件: {output_file}")
print(f"总共提取了 {len(result_df)} 个点的数据")
print(f"每个点包含 {num_bands} 个AlphaEarth波段值")

# 显示前几行数据作为示例
print("\n前5行数据预览:")
print(result_df.head())

# 显示数据统计
print("\nAlphaEarth数据统计:")
alpha_cols = [col for col in result_df.columns if col.startswith('Alpha')]
print(result_df[alpha_cols].describe())
