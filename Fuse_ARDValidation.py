# -*- coding: utf-8 -*-
# @Time    : 2025/11/2 19:22
# @Author  : Liu Kun
# @Email   : liukunjsj@163.com
# @File    : Fuse_ARDValidation.py
# @Software: PyCharm

"""
Describe: 读取大气大气校正后的影像数据，分别将每个波段的数据跟真实值做验证
"""
from osgeo import gdal
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

# 读取 Excel 文件的“哈尔滨陆表反射率数据”表
excel_path = r"C:\Users\liuku\Desktop\L-LSR-Z-HEB-M-20240903-样区12水体-V2.xlsx"
sheet_name = "哈尔滨陆表反射率数据"

df = pd.read_excel(excel_path, sheet_name=sheet_name)

# 提取波长与反射率
wavelengths = df.iloc[:, 0].values
reflectance = df.iloc[:, 1].values

# 定义高分二号影像的波段中心波长
gf2_bands = {
    "Blue": 485,
    "Green": 555,
    "Red": 660,
    "NIR": 830
}

# 计算匹配结果
results = []
for name, wl_center in gf2_bands.items():
    idx = np.argmin(np.abs(wavelengths - wl_center))
    results.append({
        "Band": name,
        "Center_Wavelength(nm)": wl_center,
        "Nearest_Wavelength(nm)": wavelengths[idx],
        "Reflectance": reflectance[idx]
    })

# 转换为 DataFrame
df_result = pd.DataFrame(results)
print(df_result)

# # 1. 打开tif文件
# tif_path = r"F:\BaiduNetdiskDownload\GF2\tif\GF2_PMS1_E93.5_N42.6_20250624_L1A14721219001_fuse.tif"
# dataset = gdal.Open(tif_path)
#
# # 2. 获取波段数
# band_count = dataset.RasterCount
# print(f"波段数量: {band_count}")
#
# # 3. 循环读取每个波段数据
# bands_data = []
# for i in range(1, band_count + 1):  # 波段从1开始计数
#     band = dataset.GetRasterBand(i)
#     data = band.ReadAsArray()
#     bands_data.append(data)
#     print(f"波段 {i} 的数据形状: {data.shape}, 数据类型: {data.dtype}")
#
# # 4. 将所有波段合并为一个三维数组 (bands, height, width)
# image_array = np.stack(bands_data)
# print(f"最终影像数组形状: {image_array.shape}")  # (波段数, 高, 宽)
#
# # 5. 如果需要读取地理信息
# geotransform = dataset.GetGeoTransform()
# projection = dataset.GetProjection()
# print("仿射变换参数:", geotransform)
# print("投影信息:", projection)
