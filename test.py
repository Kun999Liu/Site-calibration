# -*- coding: utf-8 -*-
# @Time    : 2025/11/9 15:24
# @Author  : Liu Kun
# @Email   : liukunjsj@163.com
# @File    : test_improved.py
# @Software: PyCharm

"""
改进版：根据影像时间和经纬度范围查找点并提取反射率
"""
import pandas as pd
import numpy as np
from osgeo import gdal, osr
from datetime import datetime, timedelta
import re
import os
from pathlib import Path
from sklearn.metrics import r2_score


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
    # 配置文件路径
    image_folder = r"C:\Users\liuku\Desktop\testimages\rpc_image"
    excel_path = r"C:\Users\liuku\Desktop\Afield\output_folder\GF-2 PMS2_实测反射率结果.xlsx"
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
