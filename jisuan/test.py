import numpy as np
import pandas as pd

# >>>>>>>>>>>>>>> 需手动输入下三项 <<<<<<<<<<<<<<
input_rrs = r"D:\Git\Site-calibration\excel_folder\测试数据.xlsx"  # 高光谱反射率excel文件名
input_srf = r"D:\Git\Site-calibration\SpecResponse\GF2\GF-2 PMS.xlsx"                  # 卫星光谱响应函数excel文件名
output = r"test.xlsx"           # 输出等效遥感反射率文件名

# ==================== 读取数据 ====================
try:
    Rrs_df = pd.read_excel(input_rrs, header=None)
    SRF_df = pd.read_excel(input_srf, header=None)
except Exception as e:
    raise Exception("读取失败，请检查Rrs与SRF的excel文件名是否有误") from e

# 站点名与波段名
sn = Rrs_df.iloc[0, 1:].tolist()
bn = SRF_df.iloc[0, 1:].tolist()

# 转为 numpy 矩阵
# Rrs = Rrs_df.iloc[1:, :].to_numpy(dtype=float)
# SRF = SRF_df.iloc[1:, :].to_numpy(dtype=float)
# 把非数字值（如'-'）替换为 NaN，再转换为 float
Rrs = Rrs_df.iloc[1:, :].replace(['-', ''], np.nan).astype(float).to_numpy()
SRF = SRF_df.iloc[1:, :].replace(['-', ''], np.nan).astype(float).to_numpy()

# 将 SRF 中的 NaN 或 <0 的值置为0
SRF[np.isnan(SRF)] = 0
SRF[SRF < 0] = 0

c1 = Rrs.shape[1]  # 高光谱反射率列数
c2 = SRF.shape[1]  # SRF 列数

# ==================== 计算等效 Rrs ====================
er = np.zeros((c2-1, c1-1))  # 存储等效Rrs

for i in range(1, c2):
    srf_col = SRF[:, i]
    l1 = np.argmax(srf_col != 0)  # 积分下限索引
    l2 = len(srf_col) - 1 - np.argmax(srf_col[::-1] != 0)  # 积分上限索引

    # 找出 Rrs 中对应的波长范围
    rrs_wavelength = Rrs[:, 0]
    srf_wavelength = SRF[l1:l2+1, 0]

    # 找到 Rrs 中波长对应索引
    idx_start = np.where(rrs_wavelength == srf_wavelength[0])[0][0]
    idx_end = np.where(rrs_wavelength == srf_wavelength[-1])[0][0]

    for j in range(1, c1):
        er[i-1, j-1] = np.trapz(Rrs[idx_start:idx_end+1, j] * srf_col[l1:l2+1]) / np.trapz(srf_col[l1:l2+1])

# ==================== 保存数据 ====================
# 创建标好站点名和波段名的 DataFrame
temp = pd.DataFrame(index=range(c2), columns=range(c1))

# 填充站点名与波段名
temp.iloc[0, 1:] = sn
temp.iloc[1:, 0] = bn
temp.iloc[1:, 1:] = er

# 输出 Excel
temp.to_excel(output, index=False, header=False)
print(f"等效遥感反射率已保存到 {output}")
