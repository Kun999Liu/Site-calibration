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
warnings.filterwarnings("ignore")

# ========== 1. 读取光谱响应函数（SRF） ==========
srf_df = pd.read_excel(r"D:\Git\Site-calibration\SpecResponse\GF2\GF-2 PMS.xlsx")
srf_wl = srf_df.iloc[:, 0].values  # wavelength (nm)
srf_matrix = srf_df.iloc[:, 1:].values
band_names = list(srf_df.columns[1:])
n_srf_bands = len(band_names)

print(f"光谱响应函数波段: {band_names}")
print(f"波长范围: {srf_wl.min()} - {srf_wl.max()} nm")

# ========== 2. 地面实测光谱文件夹 ==========
excel_folder = r"D:\Git\Site-calibration\excel_folder"

# ========== 3. 输出文件夹 ==========
output_folder = r"D:\Git\Site-calibration\output_folder"

# ========== 3. 结果保存容器 ==========
results = []

# ========== 4. 批量处理 Excel 文件 ==========
for xlsx in glob.glob(os.path.join(excel_folder, "*.xlsx")):
    try:
        xl = pd.ExcelFile(xlsx)
        # 自动选择含“波长”、“反射”、“光谱”的 sheet，否则取第二个
        candidate = None
        for sn in xl.sheet_names:
            if any(k in sn for k in ["波长", "反射", "光谱"]):
                candidate = sn
                break
        if candidate:
            df_spec = pd.read_excel(xlsx, sheet_name=candidate)
        else:
            df_spec = pd.read_excel(xlsx, sheet_name=1 if len(xl.sheet_names) > 1 else 0)

    except Exception as e:
        print(f" 读取 Excel 失败: {xlsx}, 错误: {e}")
        continue

    # ========== 5. 提取波长与反射率 ==========
    try:
        wl = pd.to_numeric(df_spec.iloc[:, 0], errors='coerce').dropna().values
        refl = pd.to_numeric(df_spec.iloc[:, 1], errors='coerce').dropna().values

        # 保证两列长度一致
        min_len = min(len(wl), len(refl))
        wl = wl[:min_len]
        refl = refl[:min_len]

        # 限制波长范围在 SRF 的波长覆盖区间
        valid = (wl >= srf_wl.min()) & (wl <= srf_wl.max())
        wl = wl[valid]
        refl = refl[valid]

        if len(wl) < 5:
            print(f"{os.path.basename(xlsx)} 有效波长点太少，跳过")
            continue

    except Exception as e:
        print(f"解析波长/反射率错误: {xlsx}, {e}")
        continue

    # ========== 6. 插值到 SRF 波长上 ==========
    refl_interp = np.interp(srf_wl, wl, refl, left=0, right=0)

    # ========== 7. 对每个波段进行加权积分 ==========
    simulated = []
    for b in range(n_srf_bands):
        resp = srf_matrix[:, b]
        num = np.trapz(refl_interp * resp, srf_wl)
        den = np.trapz(resp, srf_wl)
        simul_val = np.nan if den == 0 else num / den
        simulated.append(simul_val)
    simulated = np.array(simulated)
    # # ========== 7. 对每个波段进行卷积积分（MSR） ==========
    #
    # simulated = []
    # dl = np.mean(np.diff(srf_wl))  # 离散波长步长
    #
    # for b in range(n_srf_bands):
    #     resp = srf_matrix[:, b]
    #
    #     # 卷积积分：f(τ) * g(x-τ)
    #     conv_result = np.convolve(refl_interp, resp[::-1], mode='same') * dl
    #
    #     # 使用卷积结果的平均值作为模拟反射率
    #     simul_val = np.nanmean(conv_result)
    #     simulated.append(simul_val)
    #
    # simulated = np.array(simulated)

    # ========== 8. 保存结果 ==========
    results.append({
        "文件名": os.path.basename(xlsx),
        **{band_names[i]: simulated[i] for i in range(n_srf_bands)}
    })

    print(f"{os.path.basename(xlsx)} -> {simulated}")

# ========== 9. 导出所有结果 ==========
if results:
    out_df = pd.DataFrame(results)
    out_path = os.path.join(output_folder, os.path.basename(xlsx).split(".")[0]+"反射率结果.xlsx")
    out_df.to_excel(out_path, index=False)
    print(f"\n所有文件计算完成，结果已保存至：\n{out_path}")
else:
    print("未生成任何有效结果，请检查数据格式。")
