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
import random
import re
from osgeo import gdal, osr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from datetime import datetime
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

        # 删除“序号”字段
        basic_info.pop("序号", None)

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
        result_entry.update(basic_info)  # 添加基本信息字段
        result_entry.update({band_names[i]: MSR[i] for i in range(n_srf_bands)})
        results.append(result_entry)

        print(f"{os.path.basename(xlsx)} -> {MSR}")

    # ========== 8. 导出所有结果 ==========
    if results:
        out_df = pd.DataFrame(results)
        out_path = os.path.join(
            output_folder,
            os.path.basename(srf_file).split(".")[0] + "_实测反射率结果.xlsx"
        )
        out_df.to_excel(out_path, index=False)
        print(f"\n所有文件计算完成，结果已保存至：\n{out_path}")
        return out_df
    else:
        print("未生成任何有效结果，请检查数据格式。")
        return None

def get_reflectance_auto(tif_path, excel_path, output_path, scale_factor=10000, time_tolerance_days=10):
    """
    根据Excel样点信息与tif影像，提取匹配点的反射率并计算精度指标。
    自动识别影像波段数量，并只计算Excel中存在的波段。
    """
    gdal.UseExceptions()
    dataset = gdal.Open(tif_path)
    if dataset is None:
        raise FileNotFoundError(f"无法打开影像 {tif_path}")

    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    bands = dataset.RasterCount
    transform = dataset.GetGeoTransform()
    proj = osr.SpatialReference()
    proj.ImportFromWkt(dataset.GetProjection())
    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(4326)
    coordTrans = osr.CoordinateTransformation(wgs84, proj)

    # 自动波段命名
    if bands == 5:
        band_names = ['PAN', 'B1', 'B2', 'B3', 'B4']
    elif bands == 4:
        band_names = ['B1', 'B2', 'B3', 'B4']
    else:
        band_names = [f'B{i}' for i in range(1, bands + 1)]

    print(f"打开影像成功，尺寸: {cols}×{rows}, 波段数: {bands}")
    print("波段命名:", band_names)

    # 影像范围
    minx = transform[0]
    maxy = transform[3]
    maxx = minx + transform[1] * cols
    miny = maxy + transform[5] * rows
    print(f"影像范围: 经度({minx:.6f} ~ {maxx:.6f}), 纬度({miny:.6f} ~ {maxy:.6f})")

    # 读取 Excel 样点
    df = pd.read_excel(excel_path)
    if not {'经度(°)', '纬度(°)', '测量时间'}.issubset(df.columns):
        raise ValueError("Excel 中必须包含列：经度(°)、纬度(°)、测量时间")

    # 经纬度格式转换
    def dms2deg(dms):
        if isinstance(dms, (int, float)):
            return float(dms)
        s = str(dms).replace('°', ' ').replace('′', ' ').replace('’', ' ').replace('″', ' ').replace("''", ' ')
        parts = s.split()
        try:
            d, m, s = map(float, parts)
            return d + m/60 + s/3600
        except:
            return float(s) if len(parts)==1 else None

    df['经度'] = df['经度(°)'].apply(dms2deg)
    df['纬度'] = df['纬度(°)'].apply(dms2deg)
    df['测量时间'] = pd.to_datetime(df['测量时间'])

    # 解析影像日期
    tif_name = os.path.basename(tif_path)
    date_str = ''.join(filter(str.isdigit, tif_name))
    try:
        tif_date = datetime.strptime(date_str[:8], "%Y%m%d")
    except:
        tif_date = datetime.now()
    print(f"影像日期: {tif_date.strftime('%Y-%m-%d')}")

    # 筛选匹配点（影像范围 + 时间容差）
    df_filtered = df[
        (df['经度'] >= minx) & (df['经度'] <= maxx) &
        (df['纬度'] >= miny) & (df['纬度'] <= maxy) &
        (abs((df['测量时间'] - tif_date).dt.days) <= time_tolerance_days)
    ].copy()

    if df_filtered.empty:
        print("没有匹配的样点。")
        return

    print(f"匹配样点数量: {len(df_filtered)}")

    # 提取影像反射率
    results = []
    measured_all = []
    predicted_all = []

    available_bands = [b for b in band_names if b in df.columns]

    for _, row in df_filtered.iterrows():
        lon, lat = row['经度'], row['纬度']
        x_geo, y_geo, _ = coordTrans.TransformPoint(lon, lat)
        px = int((x_geo - transform[0]) / transform[1])
        py = int((y_geo - transform[3]) / transform[5])

        if 0 <= px < cols and 0 <= py < rows:
            refl_values = {}
            for i, bname in enumerate(band_names, start=1):
                val = dataset.GetRasterBand(i).ReadAsArray(px, py, 1, 1)[0, 0] / scale_factor
                refl_values[bname] = val

            entry = {
                '文件名': row['文件名'],
                '名称': row['名称'],
                '经度': lon,
                '纬度': lat,
                '高程(m)': row.get('高程(m)', None),
                '天气信息': row.get('天气信息', None),
                '地物类型': row.get('地物类型', None),
                '测量时间': row['测量时间'],
            }
            entry.update(refl_values)
            results.append(entry)

            # 保留双方都存在的波段用于计算
            y_true = [row[b] for b in available_bands]
            y_pred = [refl_values[b] for b in available_bands]
            measured_all.append(y_true)
            predicted_all.append(y_pred)

    # 保存匹配结果
    df_out = pd.DataFrame(results)

    # === 计算精度指标 ===
    measured_all = np.array(measured_all, dtype=float)
    predicted_all = np.array(predicted_all, dtype=float)

    metrics = []

    print("\n各波段精度指标：")
    for i, band in enumerate(available_bands):
        y_true = measured_all[:, i]
        y_pred = predicted_all[:, i]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r, _ = pearsonr(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"{band}: RMSE={rmse:.6f}, MAE={mae:.6f}, R={r:.6f}, R²={r2:.6f}")
        metrics.append({
            '波段': band,
            'RMSE': rmse,
            'MAE': mae,
            'R': r,
            'R2': r2
        })

    # 总体指标
    rmse_total = np.sqrt(mean_squared_error(measured_all.flatten(), predicted_all.flatten()))
    mae_total = mean_absolute_error(measured_all.flatten(), predicted_all.flatten())
    r_total, _ = pearsonr(measured_all.flatten(), predicted_all.flatten())
    r2_total = r2_score(measured_all.flatten(), predicted_all.flatten())

    print("\n总体指标：")
    print(f"RMSE={rmse_total:.6f}, MAE={mae_total:.6f}, R={r_total:.6f}, R²={r2_total:.6f}")

    # 将指标写入 Excel
    metrics_df = pd.DataFrame(metrics)
    total_metrics_df = pd.DataFrame([{
        '波段': '总体',
        'RMSE': rmse_total,
        'MAE': mae_total,
        'R': r_total,
        'R2': r2_total
    }])
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_out.to_excel(writer, index=False, sheet_name='匹配反射率')
        metrics_df.to_excel(writer, index=False, sheet_name='各波段指标')
        total_metrics_df.to_excel(writer, index=False, sheet_name='总体指标')

    print(f"匹配结果及指标已保存至：{output_path}")

    return df_out
# def get_reflectance(tif_path, scale_factor=10000):
#     """
#     参数：
#         tif_path (str): GeoTIFF 影像文件路径
#         scale_factor (float): 反射率缩放因子（默认 10000，适用于 GF 系列）
#     """
#     # 打开影像
#     dataset = gdal.Open(tif_path)
#     if dataset is None:
#         raise FileNotFoundError(f"无法打开影像 {tif_path}")
#
#     cols = dataset.RasterXSize
#     rows = dataset.RasterYSize
#     bands = dataset.RasterCount
#     print(f"图像宽度: {cols}, 高度: {rows}, 波段数: {bands}")
#
#     # 随机选择像素位置
#     x = random.randint(0, cols - 1)
#     y = random.randint(0, rows - 1)
#     print(f"随机选取的像素位置: (列={x}, 行={y})")
#
#     # 仿射变换矩阵
#     transform = dataset.GetGeoTransform()
#     proj_x = transform[0] + x * transform[1] + y * transform[2]
#     proj_y = transform[3] + x * transform[4] + y * transform[5]
#
#     # 坐标系转换
#     proj = osr.SpatialReference()
#     proj.ImportFromWkt(dataset.GetProjection())
#     wgs84 = osr.SpatialReference()
#     wgs84.ImportFromEPSG(4326)
#     transform_coord = osr.CoordinateTransformation(proj, wgs84)
#     lon, lat, _ = transform_coord.TransformPoint(proj_x, proj_y)
#     print(f"对应经纬度 (WGS84): ({lon:.6f}, {lat:.6f})")
#
#     # 读取波段反射率
#     reflectances = []
#     for b in range(1, bands + 1):
#         band = dataset.GetRasterBand(b)
#         value = band.ReadAsArray(x, y, 1, 1)[0, 0] / scale_factor
#         reflectances.append(value)
#
#     print("每个波段的物理反射率值:")
#     for idx, val in enumerate(reflectances, start=1):
#         print(f"波段 {idx}: {val:.6f}")
#
#     # 返回结果
#     return {
#         'col': x,
#         'row': y,
#         'lon': lon,
#         'lat': lat,
#         'reflectances': reflectances
#     }


# ================= 示例调用 =================
if __name__ == "__main__":
    # srf_path = r".\SpecResponse\GF2\GF-2 PMS.xlsx"
    # excel_folder = r".\excel_folder"
    # output_folder = r".\output_folder"
    #
    # measured_reflectance(srf_path, excel_folder, output_folder)


    tif_path = r"E:\GF2\GF2_PMS1_E93.5_N42.6_20250624_L1A14721219001_fuse.tif"
    excel_path = r".\output_folder\GF-2 PMS_实测反射率结果.xlsx"
    output_folder = r".\output_folder"

    df = get_reflectance_auto(tif_path, excel_path, output_folder)
