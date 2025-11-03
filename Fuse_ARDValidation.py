# -*- coding: utf-8 -*-
# @Time    : 2025/11/2 19:22
# @Author  : Liu Kun
# @Email   : liukunjsj@163.com
# @File    : Fuse_ARDValidation.py
# @Software: PyCharm

"""
Describe: 读取大气大气校正后的影像数据，分别将每个波段的数据跟真实值做验证
"""
import glob
import os
import pandas as pd
import numpy as np
import warnings
from osgeo import gdal, osr
import random
warnings.filterwarnings("ignore")

def measured_reflectance(srf_file, excel_folder, output_folder):
    # ========== 1. 读取光谱响应函数 ==========
    srf_df = pd.read_excel(srf_file)
    srf_wl = srf_df.iloc[:, 0].values  # wavelength (nm)
    srf_matrix = srf_df.iloc[:, 1:].values
    band_names = list(srf_df.columns[1:])
    n_srf_bands = len(band_names)

    print(f"光谱响应函数波段: {band_names}")
    print(f"波长范围: {srf_wl.min()} - {srf_wl.max()} nm")

    # ========== 2. 创建输出文件夹 ==========
    os.makedirs(output_folder, exist_ok=True)
    results = []

    # ========== 3. 批量处理 Excel 文件 ==========
    all_files = glob.glob(os.path.join(excel_folder, "*.xlsx"))
    if not all_files:
        print("未找到任何 Excel 文件，请检查路径。")
        return None

    for xlsx in all_files:
        print(f"\n正在处理文件: {os.path.basename(xlsx)}")

        # ========== 3.1 读取 sheet 信息 ==========
        try:
            xl = pd.ExcelFile(xlsx)
            sheet_names = xl.sheet_names
        except Exception as e:
            print(f"读取 Excel 失败: {os.path.basename(xlsx)}，错误: {e}")
            continue

        # ========== 3.2 提取“基本信息”sheet ==========
        # 读取基本信息sheet
        basic_info = {}
        basic_sheet = next((sn for sn in sheet_names if "基本信息" in sn), None)
        if basic_sheet:
            try:
                # 尝试方法 1: 字段名在第一行，数据在第二行
                df_basic = pd.read_excel(xlsx, sheet_name=basic_sheet, header=0)
                if df_basic.shape[0] > 0:
                    basic_info = df_basic.iloc[0].to_dict()
                    # 转字符串，去掉 nan
                    basic_info = {str(k): str(v) for k, v in basic_info.items() if pd.notna(v)}
                else:
                    # 方法 2: 两列形式
                    df_basic2 = pd.read_excel(xlsx, sheet_name=basic_sheet, header=None)
                    for i in range(len(df_basic2)):
                        if df_basic2.shape[1] >= 2:
                            key = str(df_basic2.iloc[i, 0]).strip()
                            val = str(df_basic2.iloc[i, 1]).strip()
                            if key.lower() != "nan" and val.lower() != "nan":
                                basic_info[key] = val
            except Exception as e:
                print(f"读取基本信息失败: {e}")
        else:
            print("未找到‘基本信息’sheet。")

        # ========== 3.3 提取光谱数据sheet ==========
        try:
            candidate = None
            for sn in sheet_names:
                if any(k in sn for k in ["波长", "反射", "光谱"]):
                    candidate = sn
                    break
            if candidate:
                df_spec = pd.read_excel(xlsx, sheet_name=candidate)
            else:
                df_spec = pd.read_excel(xlsx, sheet_name=1 if len(sheet_names) > 1 else 0)
        except Exception as e:
            print(f"读取光谱sheet失败: {e}")
            continue

        # ========== 4. 提取波长与反射率 ==========
        try:
            wl = pd.to_numeric(df_spec.iloc[:, 0], errors='coerce').dropna().values
            refl = pd.to_numeric(df_spec.iloc[:, 1], errors='coerce').dropna().values

            min_len = min(len(wl), len(refl))
            wl = wl[:min_len]
            refl = refl[:min_len]

            valid = (wl >= srf_wl.min()) & (wl <= srf_wl.max())
            wl = wl[valid]
            refl = refl[valid]
        except Exception as e:
            print(f"解析波长/反射率错误: {e}")
            continue

        # ========== 5. 插值到 SRF 波长上 ==========
        refl_interp = np.interp(srf_wl, wl, refl, left=0, right=0)

        # ========== 6. 对每个波段进行加权积分 ==========
        MSR = []
        for b in range(n_srf_bands):
            resp = srf_matrix[:, b]
            num = np.trapz(refl_interp * resp, srf_wl)
            den = np.trapz(resp, srf_wl)
            simul_val = np.nan if den == 0 else num / den
            MSR.append(simul_val)
        MSR = np.array(MSR)

        # ========== 7. 保存结果 ==========
        result_entry = {"文件名": os.path.basename(xlsx)}
        result_entry.update({band_names[i]: MSR[i] for i in range(n_srf_bands)})
        result_entry.update(basic_info)  # 添加基本信息字段
        results.append(result_entry)

        print(f"{os.path.basename(xlsx)} -> {MSR}")

    # ========== 8. 导出所有结果 ==========
    if results:
        out_df = pd.DataFrame(results)
        out_path = os.path.join(
            output_folder,
            os.path.basename(srf_file).split(".")[0] + "_反射率结果_含基本信息.xlsx"
        )
        out_df.to_excel(out_path, index=False)
        print(f"\n所有文件计算完成，结果已保存至：\n{out_path}")
        return out_df
    else:
        print("未生成任何有效结果，请检查数据格式。")
        return None

def get_random_reflectance(tif_path, scale_factor=10000):
    """
    参数：
        tif_path (str): GeoTIFF 影像文件路径
        scale_factor (float): 反射率缩放因子（默认 10000，适用于 GF 系列）
    """
    # 打开影像
    dataset = gdal.Open(tif_path)
    if dataset is None:
        raise FileNotFoundError(f"无法打开影像 {tif_path}")

    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    bands = dataset.RasterCount
    print(f"图像宽度: {cols}, 高度: {rows}, 波段数: {bands}")

    # 随机选择像素位置
    x = random.randint(0, cols - 1)
    y = random.randint(0, rows - 1)
    print(f"随机选取的像素位置: (列={x}, 行={y})")

    # 仿射变换矩阵
    transform = dataset.GetGeoTransform()
    proj_x = transform[0] + x * transform[1] + y * transform[2]
    proj_y = transform[3] + x * transform[4] + y * transform[5]

    # 坐标系转换
    proj = osr.SpatialReference()
    proj.ImportFromWkt(dataset.GetProjection())
    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(4326)
    transform_coord = osr.CoordinateTransformation(proj, wgs84)
    lon, lat, _ = transform_coord.TransformPoint(proj_x, proj_y)
    print(f"对应经纬度 (WGS84): ({lon:.6f}, {lat:.6f})")

    # 读取波段反射率
    reflectances = []
    for b in range(1, bands + 1):
        band = dataset.GetRasterBand(b)
        value = band.ReadAsArray(x, y, 1, 1)[0, 0] / scale_factor
        reflectances.append(value)

    print("每个波段的物理反射率值:")
    for idx, val in enumerate(reflectances, start=1):
        print(f"波段 {idx}: {val:.6f}")

    # 返回结果
    return {
        'col': x,
        'row': y,
        'lon': lon,
        'lat': lat,
        'reflectances': reflectances
    }


# ================= 示例调用 =================
if __name__ == "__main__":
    srf_path = r".\SpecResponse\GF2\GF-2 PMS.xlsx"
    excel_folder = r".\excel_folder"
    output_folder = r".\output_folder"

    measured_reflectance(srf_path, excel_folder, output_folder)
    #
    # tif_path = r"F:\BaiduNetdiskDownload\GF2\tif\GF2_PMS1_E93.5_N42.6_20250624_L1A14721219001_fuse.tif"
    # result = get_random_reflectance(tif_path)
    # print("\n--- 返回结果 ---")
    # print(result)
