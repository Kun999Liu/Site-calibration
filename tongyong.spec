# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_dynamic_libs
import os
import sys
import osgeo

# =================== 只需修改这里 ===================
APP_NAME = "RadiometricCorrection"
ENTRY_SCRIPT = "radcorr_win.py"
PATHEX = ["C:/Users/liuku/Desktop/Afield"]
# ===================================================

block_cipher = None

# =================== Conda 根路径 ===================
conda_prefix = sys.prefix

gdal_data = os.path.join(conda_prefix, "Library", "share", "gdal")
proj_data = os.path.join(conda_prefix, "Library", "share", "proj")

# =================== osgeo DLL ======================
osgeo_bins = collect_dynamic_libs("osgeo")

datas = [
    (gdal_data, "gdal"),
    (proj_data, "proj"),
]

hiddenimports = [
    "osgeo",
    "osgeo.gdal",
    "osgeo.ogr",
    "osgeo.osr",
    "osgeo.gdalnumeric",
]

a = Analysis(
    [ENTRY_SCRIPT],
    pathex=PATHEX,
    binaries=osgeo_bins,
    datas=datas,
    hiddenimports=hiddenimports,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name=APP_NAME,
    debug=False,
    strip=False,
    upx=False,     # ❗GDAL 禁止 UPX
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name=APP_NAME,
)
