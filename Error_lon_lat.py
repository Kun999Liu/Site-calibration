from osgeo import gdal
import shutil
import os


def shift_image_georeference(input_file, output_file, shift_x, shift_y):
    """
    平移影像的地理参考坐标

    参数:
        input_file: 输入影像文件路径
        output_file: 输出影像文件路径
        shift_x: X方向平移量（米），正值向东，负值向西
        shift_y: Y方向平移量（米），正值向北，负值向南
    """

    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：找不到文件 {input_file}")
        return False

    print("=" * 60)
    print("影像地理参考平移工具")
    print("=" * 60)

    # 打开原始影像
    print(f"\n正在打开文件: {input_file}")
    src_ds = gdal.Open(input_file)
    if src_ds is None:
        print("错误：无法打开文件！")
        return False

    # 获取原始地理变换参数
    geo_transform = src_ds.GetGeoTransform()
    projection = src_ds.GetProjection()

    print("\n原始地理参考信息:")
    print(f"  投影: {projection.split(',')[0] if projection else '未定义'}...")
    print(f"  左上角X (东坐标): {geo_transform[0]:.3f} 米")
    print(f"  左上角Y (北坐标): {geo_transform[3]:.3f} 米")
    print(f"  像元大小: {geo_transform[1]} x {abs(geo_transform[5])} 米")
    print(f"  影像尺寸: {src_ds.RasterXSize} x {src_ds.RasterYSize} 像素")

    # 显示平移信息
    print(f"\n平移设置:")
    print(f"  X方向: {shift_x:+.2f} 米 {'(向东)' if shift_x > 0 else '(向西)'}")
    print(f"  Y方向: {shift_y:+.2f} 米 {'(向北)' if shift_y > 0 else '(向南)'}")
    print(f"  总距离: {((shift_x ** 2 + shift_y ** 2) ** 0.5):.2f} 米")

    # 创建新的地理变换参数
    new_geo_transform = (
        geo_transform[0] + shift_x,  # 新的左上角X
        geo_transform[1],  # 像元宽度不变
        geo_transform[2],  # 旋转参数不变
        geo_transform[3] + shift_y,  # 新的左上角Y
        geo_transform[4],  # 旋转参数不变
        geo_transform[5]  # 像元高度不变
    )

    print("\n新的地理参考信息:")
    print(f"  左上角X (东坐标): {new_geo_transform[0]:.3f} 米")
    print(f"  左上角Y (北坐标): {new_geo_transform[3]:.3f} 米")

    # 复制文件
    print(f"\n正在复制文件到: {output_file}")
    try:
        shutil.copy(input_file, output_file)
    except Exception as e:
        print(f"错误：复制文件失败 - {e}")
        src_ds = None
        return False

    # 打开输出文件并修改地理参考
    print("正在修改地理参考...")
    out_ds = gdal.Open(output_file, gdal.GA_Update)
    if out_ds is None:
        print("错误：无法打开输出文件！")
        src_ds = None
        return False

    # 设置新的地理变换和投影
    out_ds.SetGeoTransform(new_geo_transform)
    out_ds.SetProjection(projection)

    # 刷新并关闭数据集
    out_ds.FlushCache()
    out_ds = None
    src_ds = None

    print("\n" + "=" * 60)
    print("✓ 完成！")
    print("=" * 60)
    print(f"\n输出文件: {output_file}")
    print("\n请在ENVI中加载输出文件检查位置是否正确")

    return True


def batch_shift_test(input_file, test_shifts):
    """
    批量生成多个测试平移版本

    参数:
        input_file: 输入影像文件路径
        test_shifts: 测试平移量列表 [(shift_x, shift_y, 描述), ...]
    """
    print("\n批量生成测试文件...")
    print("=" * 60)

    base_name = os.path.splitext(input_file)[0]
    results = []

    for i, (shift_x, shift_y, description) in enumerate(test_shifts, 1):
        output_file = f"{base_name}_test{i}_{description}.tif"
        print(f"\n测试 {i}/{len(test_shifts)}: {description}")
        success = shift_image_georeference(input_file, output_file, shift_x, shift_y)
        if success:
            results.append(output_file)

    print("\n" + "=" * 60)
    print("批量处理完成！")
    print("=" * 60)
    print("\n生成的文件:")
    for i, file in enumerate(results, 1):
        print(f"  {i}. {file}")

    return results


# ============ 主程序 ============

if __name__ == "__main__":
    # ===== 配置参数 =====

    # 输入文件路径（请修改为您的实际文件路径）
    input_file = r"E:\GF2_PMS2_E110.3_N37.5_20231122_L1A13489459001\GF2_PMS2_E110.3_N37.5_20231122_L1A13489459001\GF2_PMS2_E110.3_N37.5_20231122_L1A13489459001-MSS2.tif"

    # ===== 模式1：单次平移 =====
    # 取消下面的注释来使用

    # 输出文件路径
    output_file = "output_shifted.tif"

    # 设置平移量（单位：米）
    # 正值：向东/向北，负值：向西/向南
    shift_x = 65  # X方向平移量（东西方向）
    shift_y = -75  # Y方向平移量（南北方向）

    # 执行平移
    shift_image_georeference(input_file, output_file, shift_x, shift_y)

    # ===== 模式2：批量测试多个平移量 =====
    # 如果您不确定平移量，可以使用这个模式生成多个测试文件
    # 取消下面的注释来使用

    """
    # 定义多个测试平移量
    test_shifts = [
        (50, -50, "右下50m"),      # 向右下移动50米
        (100, -100, "右下100m"),   # 向右下移动100米
        (150, -150, "右下150m"),   # 向右下移动150米
        (-50, 50, "左上50m"),      # 向左上移动50米
        (-100, 100, "左上100m"),   # 向左上移动100米
        (0, -100, "正下100m"),     # 向正下移动100米
        (100, 0, "正右100m"),      # 向正右移动100米
    ]

    # 批量生成测试文件
    batch_shift_test(input_file, test_shifts)
    """

    # ===== 模式3：根据控制点计算平移量 =====
    # 如果您已经知道对应点的坐标差异
    # 取消下面的注释来使用

    """
    # 控制点坐标（从ENVI中获取）
    # 在错位影像中的坐标
    point_in_wrong_image = {
        'x': 436861.0451,  # 东坐标（米）
        'y': 4152060.2619  # 北坐标（米）
    }

    # 正确位置的坐标
    point_in_correct_position = {
        'x': 436799.4237,  # 东坐标（米）
        'y': 4152132.9394  # 北坐标（米）
    }

    # 计算平移量
    shift_x = point_in_correct_position['x'] - point_in_wrong_image['x']
    shift_y = point_in_correct_position['y'] - point_in_wrong_image['y']

    print(f"\n根据控制点计算的平移量:")
    print(f"  shift_x = {shift_x:.2f} 米")
    print(f"  shift_y = {shift_y:.2f} 米")

    output_file = "output_corrected.tif"
    shift_image_georeference(input_file, output_file, shift_x, shift_y)
    """

    print("\n提示：")
    print("1. 如果位置还不对，请修改 shift_x 和 shift_y 的值")
    print("2. 可以使用批量测试模式生成多个版本对比")
    print("3. 最精确的方法是使用模式3，根据控制点坐标计算平移量")