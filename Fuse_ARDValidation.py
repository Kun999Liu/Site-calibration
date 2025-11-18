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
import sys
import warnings
import pandas as pd
import numpy as np
from osgeo import gdal, osr
from datetime import datetime, timedelta
import re
import os
from pathlib import Path
from sklearn.metrics import r2_score
warnings.filterwarnings("ignore")
import xml.etree.ElementTree as ET


def get_base_dir():
    """获取程序运行的根目录（兼容PyInstaller打包）"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

def read_config(xml_path):
    """读取XML配置文件并返回字典"""
    config = {}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for child in root:
            config[child.tag] = child.text.strip()
        return config
    except Exception as e:
        print(f"读取配置文件失败: {e}")
        return None

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

class ReflectanceExtractor_Val:
    """高分2号影像反射率提取工具 - 根据影像范围查找点"""

    def __init__(self, image_folder, scale_factor=10000, time_threshold=3):
        """
        初始化
        :param image_folder: 包含多个影像的文件夹路径
        :param scale_factor: 反射率比例因子，默认10000
        """
        self.image_folder = image_folder
        self.images = []  # 存储影像信息列表
        self.time_threshold = time_threshold  # 时间匹配阈值（天）
        self.scale_factor = scale_factor  # 反射率比例因子

    def scan_images(self):
        """扫描文件夹中的所有影像文件并获取其时空范围"""
        image_extensions = ['.tif', '.tiff', '.img']

        print(f"正在扫描影像文件夹: {self.image_folder}")

        for root, dirs, files in os.walk(self.image_folder):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_path = os.path.join(root, file)

                    # 获取影像信息
                    image_info = self.get_image_info(image_path, file)
                    if image_info:
                        self.images.append(image_info)

        print(f"\n找到 {len(self.images)} 个影像文件:")
        for img in self.images:
            date_str = img['date'].strftime('%Y-%m-%d') if img['date'] else "未知日期"
            print(f"\n影像: {img['filename']}")
            print(f"  日期: {date_str}")
            print(f"  经度范围: {img['lon_min']:.6f}° ~ {img['lon_max']:.6f}°")
            print(f"  纬度范围: {img['lat_min']:.6f}° ~ {img['lat_max']:.6f}°")
            print(f"  波段数: {img['bands']}")

        return len(self.images)

    def get_image_info(self, image_path, filename):
        """
        获取影像的时间和经纬度范围信息
        """
        try:
            dataset = gdal.Open(image_path)
            if dataset is None:
                return None

            # 获取影像基本信息
            cols = dataset.RasterXSize
            rows = dataset.RasterYSize
            bands = dataset.RasterCount
            geotransform = dataset.GetGeoTransform()

            # 提取日期
            date = self.extract_date_from_filename(filename)

            # 获取影像四角坐标（投影坐标系）
            x_min = geotransform[0]
            y_max = geotransform[3]
            x_max = x_min + cols * geotransform[1]
            y_min = y_max + rows * geotransform[5]

            # 转换为经纬度
            try:
                import pyproj

                # 假设影像是UTM 49N投影（根据实际情况调整）
                utm49n = pyproj.CRS('EPSG:32649')
                wgs84 = pyproj.CRS('EPSG:4326')

                transformer = pyproj.Transformer.from_crs(utm49n, wgs84, always_xy=True)

                # 转换四角坐标
                lon_min, lat_min = transformer.transform(x_min, y_min)
                lon_max, lat_max = transformer.transform(x_max, y_max)

            except Exception as e:
                print(f"  警告: 坐标转换失败 ({filename}): {e}")
                dataset = None
                return None

            dataset = None

            return {
                'path': image_path,
                'filename': filename,
                'date': date,
                'lon_min': lon_min,
                'lon_max': lon_max,
                'lat_min': lat_min,
                'lat_max': lat_max,
                'bands': bands,
                'cols': cols,
                'rows': rows,
                'geotransform': geotransform
            }

        except Exception as e:
            print(f"  错误: 无法读取影像信息 ({filename}): {e}")
            return None

    def extract_date_from_filename(self, filename):
        """从文件名中提取日期"""
        patterns = [
            r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})',
            r'(\d{8})',
        ]

        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                try:
                    if len(match.groups()) == 3:
                        year, month, day = match.groups()
                    else:
                        date_str = match.group(1)
                        year = date_str[0:4]
                        month = date_str[4:6]
                        day = date_str[6:8]

                    return datetime(int(year), int(month), int(day))
                except:
                    continue

        return None

    def parse_dms_to_decimal(self, dms_str):
        """将度分秒格式转换为十进制度"""
        pattern = r'(\d+)[°]\s*(\d+)[\'′]\s*(\d+(?:\.\d+)?)'
        match = re.search(pattern, str(dms_str))

        if match:
            degrees = float(match.group(1))
            minutes = float(match.group(2))
            seconds = float(match.group(3))
            decimal = degrees + minutes / 60 + seconds / 3600
            return decimal
        else:
            try:
                return float(dms_str)
            except:
                raise ValueError(f"无法解析坐标: {dms_str}")

    def find_points_in_image(self, image_info, points_df, lon_col, lat_col, time_col):
        """
        查找在影像时空范围内的点
        """
        matched_points = []

        for idx, row in points_df.iterrows():
            try:
                # 解析经纬度
                lon = self.parse_dms_to_decimal(row[lon_col])
                lat = self.parse_dms_to_decimal(row[lat_col])

                # 检查空间范围
                if not (image_info['lon_min'] <= lon <= image_info['lon_max'] and
                        image_info['lat_min'] <= lat <= image_info['lat_max']):
                    continue

                # 检查时间范围（如果有测量时间）
                if time_col and pd.notna(row[time_col]):
                    measure_time = self.parse_time(row[time_col])
                    if measure_time and image_info['date']:
                        time_diff = abs(image_info['date'] - measure_time)
                        if time_diff > timedelta(days=self.time_threshold):
                            continue

                matched_points.append((idx, row, lon, lat))

            except Exception as e:
                print(f"  警告: 处理点 {idx + 1} 时出错: {e}")
                continue

        return matched_points

    def parse_time(self, time_str):
        """解析时间字符串"""
        if isinstance(time_str, datetime):
            return time_str

        try:
            for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%Y%m%d', '%Y-%m-%d %H:%M:%S']:
                try:
                    return datetime.strptime(str(time_str), fmt)
                except:
                    continue
        except:
            pass
        return None

    def lonlat_to_pixel(self, image_info, lon, lat):
        """将经纬度转换为像素坐标"""
        try:
            import pyproj

            wgs84 = pyproj.CRS('EPSG:4326')
            utm49n = pyproj.CRS('EPSG:32649')

            transformer = pyproj.Transformer.from_crs(wgs84, utm49n, always_xy=True)
            x, y = transformer.transform(lon, lat)

            geotransform = image_info['geotransform']
            x_origin = geotransform[0]
            y_origin = geotransform[3]
            pixel_width = geotransform[1]
            pixel_height = geotransform[5]

            col = int((x - x_origin) / pixel_width)
            row = int((y - y_origin) / pixel_height)

            # 检查范围
            if 0 <= col < image_info['cols'] and 0 <= row < image_info['rows']:
                return col, row
            else:
                return None, None

        except Exception as e:
            print(f"    坐标转换失败: {e}")
            return None, None

    def extract_reflectance(self, image_path, col, row):
        """提取指定像素位置的各波段反射率（除以10000）"""
        dataset = gdal.Open(image_path)
        if dataset is None:
            return None

        bands = dataset.RasterCount
        reflectances = []

        for band_idx in range(1, bands + 1):
            band = dataset.GetRasterBand(band_idx)
            data = band.ReadAsArray(col, row, 1, 1)
            # 除以比例因子
            reflectances.append(float(data[0, 0]) / self.scale_factor)

        dataset = None
        return reflectances

    def process_excel(self, excel_path, output_path=None):
        """
        处理Excel文件：根据影像范围查找点并提取反射率
        """
        # 读取Excel
        df = pd.read_excel(excel_path)

        print(f"\n读取到 {len(df)} 个采样点")
        print(f"Excel列名: {df.columns.tolist()}")

        # 识别列名
        lon_col = lat_col = name_col = time_col = None
        weather_col = landtype_col = elevation_col = None
        measured_cols = {}

        for col in df.columns:
            col_str = str(col).lower()
            if '经度' in col_str:
                lon_col = col
            elif '纬度' in col_str:
                lat_col = col
            elif '名称' in col_str:
                name_col = col
            elif '时间' in col_str:
                time_col = col
            elif '天气' in col_str:
                weather_col = col
            elif '地物' in col_str or '类型' in col_str:
                landtype_col = col
            elif '高程' in col_str:
                elevation_col = col
            elif col in ['B1', 'B2', 'B3', 'B4', 'PAN']:
                measured_cols[col] = col

        print(f"\n使用列: 经度={lon_col}, 纬度={lat_col}, 测量时间={time_col}")
        print(f"实测反射率列: {list(measured_cols.keys())}")

        results = []
        all_measured = []
        all_extracted = []

        # 遍历每个影像
        for img_idx, image_info in enumerate(self.images):
            print(f"\n{'=' * 60}")
            print(f"处理影像 {img_idx + 1}/{len(self.images)}: {image_info['filename']}")
            print(f"{'=' * 60}")

            # 查找在该影像范围内的点
            matched_points = self.find_points_in_image(
                image_info, df, lon_col, lat_col, time_col
            )

            print(f"找到 {len(matched_points)} 个匹配点")

            # 处理每个匹配点
            for point_idx, (idx, row, lon, lat) in enumerate(matched_points):
                try:
                    point_name = row[name_col] if name_col and pd.notna(row[name_col]) else f"点{idx + 1}"
                    measure_time = row[time_col] if time_col else None
                    weather = row[weather_col] if weather_col and pd.notna(row[weather_col]) else ""
                    landtype = row[landtype_col] if landtype_col and pd.notna(row[landtype_col]) else ""
                    elevation = row[elevation_col] if elevation_col and pd.notna(row[elevation_col]) else ""

                    print(f"\n  处理点 {point_idx + 1}: {point_name}")
                    print(f"    坐标: ({lon:.6f}°, {lat:.6f}°)")

                    # 转换为像素坐标
                    col, row_px = self.lonlat_to_pixel(image_info, lon, lat)

                    if col is None or row_px is None:
                        print(f"    ✗ 坐标转换失败")
                        continue

                    # 提取反射率
                    reflectances = self.extract_reflectance(image_info['path'], col, row_px)

                    if not reflectances:
                        print(f"    ✗ 反射率提取失败")
                        continue

                    print(f"    ✓ 成功提取反射率")

                    # 构建结果
                    result = {
                        '点位': idx + 1,
                        '名称': point_name,
                        '经度': lon,
                        '纬度': lat,
                        '高程(m)': elevation,
                        '测量时间': measure_time,
                        '天气信息': weather,
                        '地物类型': landtype,
                        '匹配影像': image_info['filename']
                    }

                    # 添加波段反射率
                    band_names = ['B1', 'B2', 'B3', 'B4'] if len(reflectances) >= 4 else ['PAN']

                    point_measured = []
                    point_extracted = []

                    for i, band_name in enumerate(band_names):
                        if i < len(reflectances) and band_name in measured_cols:
                            # 实测值
                            measured_val = float(row[measured_cols[band_name]]) if pd.notna(
                                row[measured_cols[band_name]]) else np.nan

                            # 提取值（已除以10000）
                            extracted_val = reflectances[i]

                            result[f'{band_name}_实测'] = measured_val
                            result[f'{band_name}_提取'] = extracted_val

                            # 存储该点的四个波段数据用于计算该点的R²
                            if not np.isnan(measured_val):
                                point_measured.append(measured_val)
                                point_extracted.append(extracted_val)

                                # 同时存储用于总体R²计算
                                all_measured.append(measured_val)
                                all_extracted.append(extracted_val)

                            print(f"      {band_name}: 实测={measured_val:.6f}, 提取={extracted_val:.6f}")

                    # 计算该点的R²（基于该点的四个波段）
                    if len(point_measured) >= 2:  # 至少需要2个数据点才能计算R²
                        point_r2 = r2_score(point_measured, point_extracted)
                        result['R²'] = point_r2
                        print(f"      该点R² = {point_r2:.6f}")
                    else:
                        result['R²'] = np.nan
                        print(f"      该点数据不足，无法计算R²")

                    results.append(result)

                except Exception as e:
                    print(f"    ✗ 处理失败: {e}")
                    import traceback
                    traceback.print_exc()

        # 创建结果DataFrame
        result_df = pd.DataFrame(results)

        # 计算所有点的总体R²（用于参考）
        print(f"\n{'=' * 60}")
        print("精度评估结果:")
        print(f"{'=' * 60}")

        if len(all_measured) > 0:
            measured_array = np.array(all_measured)
            extracted_array = np.array(all_extracted)

            # 移除NaN
            mask = ~(np.isnan(measured_array) | np.isnan(extracted_array))
            measured_array = measured_array[mask]
            extracted_array = extracted_array[mask]

            if len(measured_array) > 0:
                overall_r2 = r2_score(measured_array, extracted_array)
                print(f"\n所有点所有波段的总体 R² = {overall_r2:.6f}")
                print(f"总样本数: {len(measured_array)}")
            else:
                print("\n没有有效的配对数据用于计算总体R²")
        else:
            print("\n未找到任何匹配点")

        # 统计每个点的R²
        if len(results) > 0:
            valid_r2 = [r['R²'] for r in results if 'R²' in r and not np.isnan(r['R²'])]
            if len(valid_r2) > 0:
                print(f"\n各点R²统计:")
                print(f"  平均R²: {np.mean(valid_r2):.6f}")
                print(f"  最大R²: {np.max(valid_r2):.6f}")
                print(f"  最小R²: {np.min(valid_r2):.6f}")
                print(f"  有效点数: {len(valid_r2)}")
            else:
                print("\n没有点能够计算R²")

        # 保存结果
        if output_path and len(results) > 0:
            result_df.to_excel(output_path, index=False)
            print(f"\n结果已保存到: {output_path}")

        print(f"\n处理统计:")
        print(f"  总匹配点数: {len(results)}")

        return result_df


# 使用示例
if __name__ == "__main__":

    # base_dir = get_base_dir()
    #
    # xml_path = os.path.join(base_dir, "config.xml")
    #
    # if not os.path.exists(xml_path):
    #     print(f"未找到配置文件: {xml_path}")
    #     input("按任意键退出...")
    #     exit(1)
    #
    # # 读取配置
    # config = read_config(xml_path)
    # if not config:
    #     print("配置文件读取失败，请检查 config.xml")
    #     input("按任意键退出...")
    #     exit(1)
    #
    #
    # # 获取路径（自动兼容绝对/相对路径）
    # def get_abs(path_str):
    #     path_str = path_str.strip()
    #     if os.path.isabs(path_str):
    #         return path_str
    #     else:
    #         return os.path.abspath(os.path.join(base_dir, path_str))
    #
    #
    # image_folder = get_abs(config.get("image_folder", "./images"))
    # excel_path = get_abs(config.get("excel_path", "./input.xlsx"))
    # output_path = get_abs(config.get("output_path", "./reflectance_results.xlsx"))
    # scale_factor = int(config.get("scale_factor", "10000"))
    # time_threshold = int(config.get("time_threshold", "3"))
    #
    # print("====== 配置参数 ======")
    # print(f"影像文件夹: {image_folder}")
    # print(f"Excel路径: {excel_path}")
    # print(f"输出路径: {output_path}")
    # print(f"比例因子: {scale_factor}")
    # print(f"时间阈值: {time_threshold} 天")
    # print("=====================")
    #
    # # 创建提取器
    # extractor = ReflectanceExtractor_Val(
    #     image_folder=image_folder,
    #     scale_factor=scale_factor,
    #     time_threshold=time_threshold
    # )
    #
    # # 扫描影像并执行
    # if extractor.scan_images() > 0:
    #     results = extractor.process_excel(excel_path, output_path)
    #     if len(results) > 0:
    #         print("\n提取结果预览:")
    #         print(results.head())
    #     else:
    #         print("\n未找到任何匹配点")
    # else:
    #     print("未找到影像文件，请检查路径")
    #
    # input("\n任务完成，按任意键退出...")

    # srf_path = r".\SpecRsp\GF2\GF-2 PMS1.xlsx"
    #     excel_folder = r".\excel_folder"
    #     output_folder = r".\output_folder"
    #
    #     measured_reflectance(srf_path, excel_folder, output_folder)

    # 配置文件路径
    image_folder = r"E:\testimage"
    excel_path = r".\output_folder\test.xlsx"
    output_path = r".\reflectance_results.xlsx"

    # 创建提取器
    extractor = ReflectanceExtractor_Val(image_folder, scale_factor=10000)

    # 扫描影像文件
    if extractor.scan_images() > 0:
        # 处理Excel
        results = extractor.process_excel(excel_path, output_path)

        # 显示结果预览
        if len(results) > 0:
            print("\n提取结果预览:")
            print(results.head())
        else:
            print("\n未找到任何匹配点")
    else:
        print("未找到影像文件，请检查文件夹路径")



# # ================= 示例调用 =================
# if __name__ == "__main__":
#     srf_path = r".\SpecRsp\GF2\GF-2 PMS1.xlsx"
#     excel_folder = r".\excel_folder"
#     output_folder = r".\output_folder"
#
#     measured_reflectance(srf_path, excel_folder, output_folder)


    # tif_path = r"E:\GF2\GF2_PMS1_E93.5_N42.6_20250624_L1A14721219001_fuse.tif"
    # excel_path = r".\output_folder\GF-2 PMS_实测反射率结果.xlsx"
    # output_folder = r".\output_folder"

    # df = get_reflectance_auto(tif_path, excel_path, output_folder)
