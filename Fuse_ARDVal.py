# -*- coding: utf-8 -*-
# @Time    : 2025/11/2 19:22
# @Author  : Liu Kun
# @Email   : liukunjsj@163.com
# @File    : Fuse_ARDValidation.py
# @Software: PyCharm

"""
Describe: 读取大气校正后的影像数据，分别将每个波段的数据跟真实值做验证
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
    """计算实测反射率的光谱响应函数卷积"""
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

        # ========== 3.2 提取"基本信息"sheet ==========
        basic_info = {}
        basic_sheet = next((sn for sn in sheet_names if "基本信息" in sn), None)
        if basic_sheet:
            try:
                df_basic = pd.read_excel(xlsx, sheet_name=basic_sheet, header=0)
                if df_basic.shape[0] > 0:
                    basic_info = df_basic.iloc[0].to_dict()
                    basic_info = {str(k): str(v) for k, v in basic_info.items() if pd.notna(v)}
                else:
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
            print("未找到'基本信息'sheet。")

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
        result_entry.update(basic_info)
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
        self.image_folder = image_folder
        self.images = []
        self.time_threshold = time_threshold
        self.scale_factor = scale_factor

    def scan_images(self):
        """扫描文件夹中的所有影像文件并获取其时空范围"""
        image_extensions = ['.tif', '.tiff', '.img']

        print(f"正在扫描影像文件夹: {self.image_folder}")

        for root, dirs, files in os.walk(self.image_folder):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_path = os.path.join(root, file)
                    image_info = self.get_image_info(image_path, file)
                    if image_info:
                        self.images.append(image_info)

        print(f"\n找到 {len(self.images)} 个影像文件:")
        for img in self.images:
            date_str = img['date'].strftime('%Y-%m-%d') if img['date'] else "未知日期"
            print(f"\n影像: {img['filename']}")
            print(f"  日期: {date_str}")
            print(f"  投影: {img['projection_name']}")
            print(f"  经度范围: {img['lon_min']:.6f}° ~ {img['lon_max']:.6f}°")
            print(f"  纬度范围: {img['lat_min']:.6f}° ~ {img['lat_max']:.6f}°")
            print(f"  波段数: {img['bands']}")

        return len(self.images)

    def get_image_info(self, image_path, filename):
        """获取影像的时间和经纬度范围信息（通用版本）"""
        try:
            dataset = gdal.Open(image_path)
            if dataset is None:
                return None

            cols = dataset.RasterXSize
            rows = dataset.RasterYSize
            bands = dataset.RasterCount
            geotransform = dataset.GetGeoTransform()
            projection_wkt = dataset.GetProjection()

            date = self.extract_date_from_filename(filename)

            # 获取影像四角坐标
            x_min = geotransform[0]
            y_max = geotransform[3]
            x_max = x_min + cols * geotransform[1]
            y_min = y_max + rows * geotransform[5]

            try:
                import pyproj

                # 从影像投影信息创建CRS
                try:
                    image_crs = pyproj.CRS.from_wkt(projection_wkt)
                    projection_name = image_crs.name
                    print(f"  读取投影: {projection_name}")
                except Exception as e:
                    print(f"  警告: 无法解析投影信息，使用默认UTM Zone 49N")
                    image_crs = pyproj.CRS('EPSG:32649')
                    projection_name = "UTM Zone 49N (默认)"

                # 创建WGS84坐标系
                wgs84 = pyproj.CRS('EPSG:4326')

                # 创建坐标转换器（从影像投影转到WGS84）
                transformer = pyproj.Transformer.from_crs(image_crs, wgs84, always_xy=True)

                # 转换四个角点
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
                'projection_wkt': projection_wkt,
                'projection_name': projection_name,
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
        """查找在影像时空范围内的点"""
        matched_points = []

        for idx, row in points_df.iterrows():
            try:
                lon = self.parse_dms_to_decimal(row[lon_col])
                lat = self.parse_dms_to_decimal(row[lat_col])

                if not (image_info['lon_min'] <= lon <= image_info['lon_max'] and
                        image_info['lat_min'] <= lat <= image_info['lat_max']):
                    continue

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
        try:
            import pyproj

            # 源坐标系：WGS84
            wgs84 = pyproj.CRS('EPSG:4326')

            # 目标坐标系：从影像信息获取
            try:
                image_crs = pyproj.CRS.from_wkt(image_info['projection_wkt'])
            except:
                # 如果投影信息无效，根据经度自动判断UTM分区
                utm_zone = int((lon + 180) / 6) + 1
                hemisphere = 'north' if lat >= 0 else 'south'
                epsg_code = 32600 + utm_zone if hemisphere == 'north' else 32700 + utm_zone
                image_crs = pyproj.CRS(f'EPSG:{epsg_code}')
                print(f"    使用自动判断投影: UTM Zone {utm_zone}{hemisphere[0].upper()}")

            # 创建坐标转换器
            transformer = pyproj.Transformer.from_crs(wgs84, image_crs, always_xy=True)
            x, y = transformer.transform(lon, lat)

            # 获取仿射变换参数
            geotransform = image_info['geotransform']
            x_origin = geotransform[0]
            y_origin = geotransform[3]
            pixel_width = geotransform[1]
            pixel_height = geotransform[5]

            # 计算像素坐标
            col = int((x - x_origin) / pixel_width)
            row = int((y - y_origin) / pixel_height)

            # 检查是否在影像范围内
            if 0 <= col < image_info['cols'] and 0 <= row < image_info['rows']:
                return col, row
            else:
                print(f"    警告: 像素坐标({col}, {row})超出影像范围")
                return None, None

        except Exception as e:
            print(f"    坐标转换失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def extract_reflectance(self, image_path, col, row):
        """提取指定像素位置的各波段反射率，并应用比例因子"""
        dataset = gdal.Open(image_path)
        if dataset is None:
            return None

        bands = dataset.RasterCount
        reflectances = []

        for band_idx in range(1, bands + 1):
            band = dataset.GetRasterBand(band_idx)
            data = band.ReadAsArray(col, row, 1, 1)
            # 应用比例因子
            reflectances.append(float(data[0, 0]) / self.scale_factor)

        dataset = None
        return reflectances

    def process_excel(self, excel_path, output_path=None):
        """处理Excel文件：根据影像范围查找点并提取反射率"""
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
            print(f"投影系统: {image_info['projection_name']}")
            print(f"{'=' * 60}")

            matched_points = self.find_points_in_image(
                image_info, df, lon_col, lat_col, time_col
            )

            print(f"找到 {len(matched_points)} 个匹配点")

            for point_idx, (idx, row, lon, lat) in enumerate(matched_points):
                try:
                    point_name = row[name_col] if name_col and pd.notna(row[name_col]) else f"点{idx + 1}"
                    measure_time = row[time_col] if time_col else None
                    weather = row[weather_col] if weather_col and pd.notna(row[weather_col]) else ""
                    landtype = row[landtype_col] if landtype_col and pd.notna(row[landtype_col]) else ""
                    elevation = row[elevation_col] if elevation_col and pd.notna(row[elevation_col]) else ""

                    print(f"\n  处理点 {point_idx + 1}: {point_name}")
                    print(f"    坐标: ({lon:.6f}°, {lat:.6f}°)")

                    col, row_px = self.lonlat_to_pixel(image_info, lon, lat)

                    if col is None or row_px is None:
                        print(f"坐标转换失败")
                        continue

                    print(f"像素坐标: ({col}, {row_px})")

                    reflectances = self.extract_reflectance(image_info['path'], col, row_px)

                    if not reflectances:
                        print(f"反射率提取失败")
                        continue

                    print(f"成功提取反射率 (已除以比例因子{self.scale_factor})")

                    result = {
                        '点位': idx + 1,
                        '名称': point_name,
                        '经度': lon,
                        '纬度': lat,
                        '高程(m)': elevation,
                        '测量时间': measure_time,
                        '天气信息': weather,
                        '地物类型': landtype,
                        '匹配影像': image_info['filename'],
                        '影像投影': image_info['projection_name']
                    }

                    # 根据波段数匹配
                    band_names = ['B1', 'B2', 'B3', 'B4'] if len(reflectances) >= 4 else ['PAN']

                    point_measured = []
                    point_extracted = []

                    for i, band_name in enumerate(band_names):
                        if i < len(reflectances) and band_name in measured_cols:
                            measured_val = float(row[measured_cols[band_name]]) if pd.notna(
                                row[measured_cols[band_name]]) else np.nan
                            extracted_val = reflectances[i]

                            result[f'{band_name}_实测'] = measured_val
                            result[f'{band_name}_提取'] = extracted_val

                            if not np.isnan(measured_val):
                                diff = extracted_val - measured_val
                                # rel_error = (diff / measured_val * 100) if measured_val != 0 else np.nan

                                result[f'{band_name}_差值'] = diff
                                # result[f'{band_name}_相对误差(%)'] = rel_error

                                point_measured.append(measured_val)
                                point_extracted.append(extracted_val)

                                all_measured.append(measured_val)
                                all_extracted.append(extracted_val)

                                print(f"      {band_name}: 实测={measured_val:.6f}, "
                                      f"提取={extracted_val:.6f}, 差值={diff:.6f}, ")
                                      # f"相对误差={rel_error:.2f}%")

                    # 计算统计指标
                    if len(point_measured) >= 2:
                        try:
                            # RMSE
                            point_rmse = np.sqrt(np.mean(
                                (np.array(point_measured) - np.array(point_extracted)) ** 2
                            ))
                            result['RMSE'] = point_rmse
                            print(f"RMSE = {point_rmse:.6f}")

                            # MAPE
                            point_mape = np.mean([
                                abs((m - e) / m) for m, e in zip(point_measured, point_extracted) if m != 0
                            ]) * 100
                            result['MAPE(%)'] = point_mape
                            print(f"MAPE = {point_mape:.2f}%")

                            # 相关系数
                            point_corr = np.corrcoef(point_measured, point_extracted)[0, 1]
                            #计算决定系数
                            point_r2 = r2_score(point_measured, point_extracted)
                            result['相关系数'] = point_corr
                            print(f"相关系数 = {point_corr:.6f}")

                            # R²
                            point_r2 = point_corr ** 2
                            result['R²'] = point_r2
                            print(f"R² = {point_r2:.6f}")

                        except Exception as e:
                            print(f"      统计指标计算失败: {e}")
                            result['RMSE'] = np.nan
                            result['MAPE(%)'] = np.nan
                            result['相关系数'] = np.nan
                            result['R²'] = np.nan
                    else:
                        result['RMSE'] = np.nan
                        result['MAPE(%)'] = np.nan
                        result['相关系数'] = np.nan
                        result['R²'] = np.nan
                        print(f"      该点数据不足，无法计算统计指标")

                    results.append(result)

                except Exception as e:
                    print(f"    ✗ 处理失败: {e}")
                    import traceback
                    traceback.print_exc()

        # 创建结果DataFrame
        result_df = pd.DataFrame(results)

        # 保存结果
        if output_path and len(results) > 0:
            result_df.to_excel(output_path, index=False)
            print(f"\n结果已保存到: {output_path}")

        print(f"\n处理统计:")
        print(f"  总匹配点数: {len(results)}")

        return result_df


# 使用示例
if __name__ == "__main__":
    # 配置路径
    image_folder = r"C:\Users\7X\Desktop\images"  # 影像文件夹
    excel_path = r".\output_folder\GF-2 PMS2_实测反射率结果.xlsx"  # 实测反射率Excel
    output_path = r".\reflectance_results.xlsx"  # 输出结果

    # 创建提取器
    extractor = ReflectanceExtractor_Val(
        image_folder=image_folder,
        scale_factor=10000,  # 反射率比例因子
        time_threshold=3  # 时间匹配阈值（天）
    )

    # 扫描影像文件
    if extractor.scan_images() > 0:
        # 处理Excel并提取反射率
        results = extractor.process_excel(excel_path, output_path)

        # 显示结果预览
        if len(results) > 0:
            print("\n提取结果预览:")
            print(results.head(10))
        else:
            print("\n未找到任何匹配点")
    else:
        print("未找到影像文件，请检查文件夹路径")
