import os, sys
import xml.etree.ElementTree as ET
import clr

clr.AddReference("System")
from System import Array, Double
from System.Collections.Generic import List
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QComboBox,
    QDateEdit, QTextEdit, QMessageBox, QSplitter, QRubberBand
)
from PyQt5.QtCore import QDate, Qt, QRect, QSize
from PyQt5.QtGui import QPixmap, QImage
from osgeo import gdal
import numpy as np


class CalibrationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("高分五号DPC绝对辐射定标-场地定标")
        self.setGeometry(200, 100, 1000, 600)
        self.pick_mode = False  # 是否处于画框模式
        self.image = None  # 完整图像数据用于计算
        self.display_scale = 1.0  # 显示缩放比例
        self.valid_sizes = ["3", "5", "9", "13", "19", "25", "33", "43", "51"]
        self.rubber_rect = QRect()
        self.Ltoa = 0  # 表观辐亮度

        # --- 主分割布局（左右） ---
        splitter = QSplitter(Qt.Horizontal)

        # ---------------- 左侧布局 ----------------
        left_widget = QWidget()
        left_layout = QVBoxLayout()

        # 1. 图像信息
        image_group = QGroupBox("1.选择图像信息")
        img_layout = QVBoxLayout()
        self.image_path = QLineEdit()
        btn_img = QPushButton("选择影像")
        btn_img.clicked.connect(self.open_image)
        h1 = QHBoxLayout()
        h1.addWidget(self.image_path)
        h1.addWidget(btn_img)

        self.center_row = QLineEdit()
        self.center_col = QLineEdit()
        self.sub_size = QLineEdit()
        self.range_km = QLineEdit("1.15")
        self.date_edit = QDateEdit()
        self.date_edit.setDate(QDate.currentDate())

        h2 = QHBoxLayout()
        h2.addWidget(QLabel("中心行:"))
        h2.addWidget(self.center_row)
        h2.addWidget(QLabel("中心列:"))
        h2.addWidget(self.center_col)

        # 行方向：可编辑的下拉框
        self.sub_size_row = QComboBox()
        self.sub_size_row.setEditable(True)
        self.sub_size_row.addItems(self.valid_sizes)
        self.sub_size_row.setCurrentText("5")  # 默认值

        # 列方向：跟随，不可编辑
        self.sub_size_col = QComboBox()
        self.sub_size_col.setEditable(False)
        self.sub_size_col.addItem("5")  # 初始值，保证 count() > 0

        # 当行修改时触发校正并同步列
        def sync_sizes(text):
            try:
                val = int(text)
            except ValueError:
                return  # 非法输入直接忽略

            # 限制在图像大小范围内
            if hasattr(self, "image") and self.image is not None:
                h, w = self.image.shape[:2]
                max_size = min(h, w)
                val = max(1, min(val, max_size))  # 限制 1 ~ max_size

            # 更新行控件显示
            self.sub_size_row.blockSignals(True)
            self.sub_size_row.setCurrentText(str(val))
            self.sub_size_row.blockSignals(False)

            # 更新列控件显示
            self.sub_size_col.setItemText(0, str(val))  # 直接修改第0项显示

        # 连接信号
        self.sub_size_row.currentTextChanged.connect(sync_sizes)

        # 初始化同步一次
        sync_sizes(self.sub_size_row.currentText())

        h3 = QHBoxLayout()
        h3.addWidget(QLabel("子图像大小:"))
        h3.addWidget(self.sub_size_row)
        h3.addWidget(QLabel("×"))
        h3.addWidget(self.sub_size_col)
        h3_1 = QHBoxLayout()
        h3_1.addWidget(QLabel("图像区域选择定标中心点:"))
        btn_pick = QPushButton("选择中心点")
        btn_pick.clicked.connect(self.activate_pick_mode)
        h3_1.addWidget(btn_pick)
        h4 = QHBoxLayout()
        h4.addWidget(QLabel("定标区域海拔:"))
        h4.addWidget(self.range_km)
        h4.addWidget(QLabel("km"))
        h5 = QHBoxLayout()
        h5.addWidget(QLabel("成像日期:"))
        h5.addWidget(self.date_edit)

        for h in [h1, h2, h3, h3_1, h4, h5]:
            img_layout.addLayout(h)
        image_group.setLayout(img_layout)

        # 2. 传感器信息
        sensor_group = QGroupBox("2.输入传感器信息")
        sensor_layout = QHBoxLayout()
        self.band_combo = QComboBox()
        self.band_combo.addItems([
            "443nm", "490nm", "565nm", "670nm",
            "763nm", "765nm", "865nm", "910nm"
        ])
        sensor_layout.addWidget(QLabel("波段:"))
        sensor_layout.addWidget(self.band_combo)
        sensor_group.setLayout(sensor_layout)

        # 3. 几何条件信息
        geom_group = QGroupBox("3.输入几何条件信息")
        geom_layout = QVBoxLayout()
        self.sun_az = QLineEdit()
        self.sun_el = QLineEdit()
        self.sat_az = QLineEdit()
        self.sat_zn = QLineEdit()
        btn_xml = QPushButton("读取 XML")
        btn_xml.clicked.connect(self.load_from_xml)

        h6 = QHBoxLayout()
        h6.addWidget(QLabel("太阳方位角:"))
        h6.addWidget(self.sun_az)
        h7 = QHBoxLayout()
        h7.addWidget(QLabel("太阳天顶角:"))
        h7.addWidget(self.sun_el)
        h8 = QHBoxLayout()
        h8.addWidget(QLabel("观测方位角:"))
        h8.addWidget(self.sat_az)
        h9 = QHBoxLayout()
        h9.addWidget(QLabel("观测天顶角:"))
        h9.addWidget(self.sat_zn)

        for h in [h6, h7, h8, h9]:
            geom_layout.addLayout(h)
        geom_layout.addWidget(btn_xml)
        geom_group.setLayout(geom_layout)

        # 4. 地气参数
        atm_group = QGroupBox("4.输入实测地气参数")
        atm_layout = QVBoxLayout()
        self.reflect_file = QLineEdit()
        btn_ref = QPushButton("选择反射率文件")
        btn_ref.clicked.connect(self.open_reflect)
        h10 = QHBoxLayout()
        h10.addWidget(self.reflect_file)
        h10.addWidget(btn_ref)

        self.aod_550 = QLineEdit()
        h11 = QHBoxLayout()
        h11.addWidget(QLabel("气溶胶光学厚度550nm:"))
        h11.addWidget(self.aod_550)

        atm_layout.addLayout(h10)
        atm_layout.addLayout(h11)
        atm_group.setLayout(atm_layout)

        # 结果输出框
        self.result_box = QTextEdit()
        self.result_box.setPlaceholderText("计算结果将在这里显示...")

        # 底部按钮
        btn_calc = QPushButton("计算定标系数")
        btn_calc.clicked.connect(self.calculate)

        btn_help = QPushButton("帮助")
        btn_help.clicked.connect(self.show_help)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(btn_calc)
        bottom_layout.addWidget(btn_help)

        # 左侧整体布局
        for w in [image_group, sensor_group, geom_group, atm_group, self.result_box]:
            left_layout.addWidget(w)
        left_layout.addLayout(bottom_layout)
        left_widget.setLayout(left_layout)

        # ---------------- 右侧图像显示 ----------------
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        self.image_display = CalibrationApp.ImageLabel()
        self.image_display.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.image_display.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        right_layout.addWidget(self.image_display)
        right_widget.setLayout(right_layout)

        # 分割布局
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([100, 900])  # 左右比例

        # 设置主布局
        layout = QHBoxLayout()
        layout.addWidget(splitter)
        self.setLayout(layout)

        # ---------- 下拉框联动矩形 ----------
        self.sub_size_row.currentTextChanged.connect(self.update_rubber_from_combo)

    # --- 功能函数 ---
    def activate_pick_mode(self):
        """进入选点模式"""
        if self.image_display.pixmap() is None:
            QMessageBox.warning(self, "错误", "请先加载一幅图像。")
            return
        self.pick_mode = not self.pick_mode
        self.setCursor(Qt.CrossCursor if self.pick_mode else Qt.ArrowCursor)
        msg = "画框模式已开启，请拖动鼠标选择区域。" if self.pick_mode else "画框模式已关闭。"
        QMessageBox.information(self, "提示", msg)

    def update_rubber_from_combo(self, text):
        """下拉框修改矩形大小"""
        if self.image is None:
            return
        try:
            size = int(text)
        except ValueError:
            return

        img_h, img_w = self.image.shape[:2]

        # 以当前矩形中心为中心，计算新矩形（显示坐标系）
        if self.rubber_rect.isNull():
            # 如果没有已有矩形，用图像中心
            cx, cy = int(img_w * self.display_scale / 2), int(img_h * self.display_scale / 2)
        else:
            cx, cy = self.rubber_rect.center().x(), self.rubber_rect.center().y()

        # 在显示坐标系中的尺寸
        display_size = int(size * self.display_scale)
        half = display_size // 2

        # 获取显示图像尺寸
        if self.image_display.pixmap():
            display_w = self.image_display.pixmap().width()
            display_h = self.image_display.pixmap().height()

            # 限制矩形不越界
            cx = max(half, min(cx, display_w - half - 1))
            cy = max(half, min(cy, display_h - half - 1))

            new_rect = QRect(cx - half, cy - half, display_size, display_size)
            self.rubber_rect = new_rect
            self.image_display.update_rubber(self.rubber_rect)

            # 转换回原始坐标并更新显示
            orig_cx = int(cx / self.display_scale)
            orig_cy = int(cy / self.display_scale)
            self.center_col.setText(str(orig_cx))
            self.center_row.setText(str(orig_cy))
            self.sub_size_col.setItemText(0, str(size))

    # ---------- ImageLabel 支持画框 ----------
    class ImageLabel(QLabel):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.start_pos = None
            self.rubber_band = QRubberBand(QRubberBand.Rectangle, self)

        def mousePressEvent(self, event):
            main_win = self.window()
            if main_win.pick_mode and event.button() == Qt.LeftButton:
                self.start_pos = event.pos()
                self.rubber_band.setGeometry(QRect(self.start_pos, QSize()))
                self.rubber_band.show()
            else:
                super().mousePressEvent(event)

        def mouseMoveEvent(self, event):
            main_win = self.window()
            if main_win.pick_mode and self.start_pos:
                rect = self.calc_rect(main_win, event.pos())
                self.rubber_band.setGeometry(rect)
            else:
                super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event):
            main_win = self.window()
            if main_win.pick_mode and self.start_pos and event.button() == Qt.LeftButton:
                rect = self.calc_rect(main_win, event.pos())
                self.rubber_band.setGeometry(rect)
                self.rubber_band.show()

                # 更新主窗口矩形信息（显示坐标系）
                main_win.rubber_rect = rect

                # 转换到原始图像坐标系
                scale = main_win.display_scale
                center_x = int(rect.center().x() / scale)
                center_y = int(rect.center().y() / scale)
                side = int(rect.width() / scale)

                main_win.center_col.setText(str(center_x))
                main_win.center_row.setText(str(center_y))
                main_win.sub_size_row.setCurrentText(str(side))
                main_win.sub_size_col.setItemText(0, str(side))

                self.start_pos = None
            else:
                super().mouseReleaseEvent(event)

        def calc_rect(self, main_win, current_pos):
            """根据起点和当前鼠标位置计算正方形矩形，保证不越界"""
            dx = current_pos.x() - self.start_pos.x()
            dy = current_pos.y() - self.start_pos.y()
            side = max(abs(dx), abs(dy))

            # 获取显示图像尺寸
            if self.pixmap():
                display_w = self.pixmap().width()
                display_h = self.pixmap().height()
            else:
                return QRect(self.start_pos, QSize())

            x = self.start_pos.x()
            y = self.start_pos.y()
            if dx < 0: x -= side
            if dy < 0: y -= side

            # 限制矩形在显示图像内
            if x < 0: x = 0
            if y < 0: y = 0
            if x + side > display_w: side = display_w - x
            if y + side > display_h: side = display_h - y

            return QRect(x, y, side, side)

        def update_rubber(self, rect):
            self.rubber_band.setGeometry(rect)
            self.rubber_band.show()

    def open_image(self):
        """打开遥感影像图片"""
        file, _ = QFileDialog.getOpenFileName(
            self, "选择影像", "", "影像文件 (*.tif *.jpg *.png *.tiff)"
        )
        if not file:
            return

        self.image_path.setText(file)

        if file.lower().endswith(".tif") or file.lower().endswith(".tiff"):
            try:
                dataset = gdal.Open(file)
                if dataset is None:
                    QMessageBox.critical(self, "错误", f"无法打开影像: {file}")
                    return

                # 读取完整影像数据用于计算
                bands = min(3, dataset.RasterCount)
                img_data = []
                for i in range(1, bands + 1):
                    band = dataset.GetRasterBand(i).ReadAsArray()
                    img_data.append(band)

                img_data = np.dstack(img_data).astype(np.float32)
                self.image = img_data  # 保存完整数据用于计算

                # ========== 生成缩略图用于显示 ==========
                max_display_size = 800  # 最大显示尺寸
                h, w = img_data.shape[:2]

                # 计算缩放比例
                scale = min(max_display_size / w, max_display_size / h, 1.0)
                new_w, new_h = int(w * scale), int(h * scale)

                # 保存缩放比例，用于坐标转换
                self.display_scale = scale

                # 使用 gdal.Translate 生成缩略图
                thumbnail = gdal.Translate(
                    '', dataset,
                    format='MEM',
                    width=new_w,
                    height=new_h
                )

                # 读取缩略图数据
                thumb_data = []
                for i in range(1, bands + 1):
                    band = thumbnail.GetRasterBand(i).ReadAsArray()
                    thumb_data.append(band)

                thumb_data = np.dstack(thumb_data).astype(np.float32)

                # 归一化显示
                thumb_data -= thumb_data.min()
                thumb_data /= (thumb_data.max() + 1e-6)
                thumb_data *= 255
                thumb_data = thumb_data.astype(np.uint8)

                h, w, c = thumb_data.shape
                qimg = QImage(thumb_data.data, w, h, 3 * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)

            except Exception as e:
                QMessageBox.critical(self, "错误", f"读取tif失败: {e}")
                return
        else:
            # 普通格式直接显示
            pixmap = QPixmap(file)
            if not pixmap.isNull():
                # 缩放到合适大小
                max_display_size = 800
                orig_w, orig_h = pixmap.width(), pixmap.height()

                if orig_w > max_display_size or orig_h > max_display_size:
                    self.display_scale = min(max_display_size / orig_w, max_display_size / orig_h)
                    pixmap = pixmap.scaled(max_display_size, max_display_size,
                                           Qt.KeepAspectRatio, Qt.SmoothTransformation)
                else:
                    self.display_scale = 1.0

                # 将普通图片转换成 np.array 存储到 self.image
                qimg = pixmap.toImage()
                ptr = qimg.bits()
                ptr.setsize(qimg.byteCount())
                self.image = np.array(ptr).reshape(qimg.height(), qimg.width(), 4)

        if not pixmap.isNull():
            self.image_display.setPixmap(pixmap)
            self.image_display.adjustSize()
            # 限制 ImageLabel 的最大尺寸
            self.image_display.setMaximumSize(pixmap.size())

    def open_reflect(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择反射率文件", "", "文本文件 (*.txt *.csv)")
        if file:
            self.reflect_file.setText(file)

    def load_from_xml(self):
        xml_file, _ = QFileDialog.getOpenFileName(self, "选择 XML 文件", "", "XML Files (*.xml)")
        if not xml_file:
            return
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # 示例字段（需根据实际 XML 调整）
            sun_az = root.findtext(".//SolarAzimuth") or root.findtext(".//SunAzimuth")
            sun_zn = root.findtext(".//SolarZenith") or root.findtext(".//SunZenith")
            sat_az = root.findtext(".//SatelliteAzimuth")
            sat_zn = root.findtext(".//SatelliteZenith")

            if sun_az: self.sun_az.setText(sun_az)
            if sun_zn: self.sun_el.setText(sun_zn)
            if sat_az: self.sat_az.setText(sat_az)
            if sat_zn: self.sat_zn.setText(sat_zn)

            QMessageBox.information(self, "提示", "XML 解析完成，几何参数已填充。")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"XML 解析失败: {e}")

    def calculate(self):
        """计算定标系数"""
        try:
            # ------------------ 1. 读取基本输入 ------------------
            img_file = self.image_path.text().strip()
            if not img_file:
                QMessageBox.warning(self, "提示", "请先选择影像文件！")
                return

            # 子图大小（行列相同）
            try:
                sub_size = int(self.sub_size_row.currentText())
            except Exception:
                QMessageBox.warning(self, "提示", "子图像大小不正确！")
                return

            # 中心像素
            try:
                row_c = int(self.center_row.text())
                col_c = int(self.center_col.text())
            except Exception:
                QMessageBox.warning(self, "提示", "请输入中心像素行列号！")
                return

            # 气溶胶光学厚度
            try:
                aod = float(self.aod_550.text())
            except:
                QMessageBox.warning(self, "提示", "请输入气溶胶光学厚度 (550nm) 和 几何条件信息")
                return

            # ------------------ 2. 读取子区域 DN ------------------
            dataset = gdal.Open(img_file, gdal.GA_ReadOnly)
            if dataset is None:
                QMessageBox.critical(self, "错误", f"无法打开文件: {img_file}")
                return

            # 波段
            band_text = self.band_combo.currentText()
            band_index = self.band_combo.currentIndex() + 1  # GDAL band 从1开始
            band_count = dataset.RasterCount
            if band_count == 0:
                QMessageBox.critical(self, "错误", f"影像文件 {img_file} 无波段！")
                return

            # 获取当前选择波段的波段对象
            band = dataset.GetRasterBand(band_index)
            if band is None:
                QMessageBox.critical(self, "错误", f"无法读取第 {band_index} 波段！")
                return

            half = sub_size // 2
            # 限制中心点
            img_h, img_w = self.image.shape[:2]

            # 如果子图大于图像尺寸，自动调整子图大小
            if sub_size > min(img_h, img_w):
                sub_size = min(img_h, img_w)
                half = sub_size // 2
                self.sub_size_row.setCurrentText(str(sub_size))
                self.sub_size_col.setItemText(0, str(sub_size))

            # 限制中心点在图像内部
            row_c = max(half, min(row_c, img_h - half - 1))
            col_c = max(half, min(col_c, img_w - half - 1))

            xoff, yoff = col_c - half, row_c - half
            data = band.ReadAsArray(xoff, yoff, sub_size, sub_size).astype(float)

            # 中心像元大小：DN_mean
            center_DN = data[half, half]
            DN_mean = float(np.mean(data))

            # ------------------ 3. 地表反射率文件 ------------------
            ref_file = self.reflect_file.text().strip()
            if not ref_file:
                QMessageBox.warning(self, "提示", "请选择地表反射率文件！")
                return

            # 简单读取：假设 CSV 两列 (WaveLengthField, RefField)
            WaveLengthField, RefField = [], []
            with open(ref_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    parts = line.replace(",", " ").split()
                    if len(parts) >= 2:
                        WaveLengthField.append(float(parts[0]))
                        RefField.append(float(parts[1]))

            if len(WaveLengthField) == 0:
                QMessageBox.warning(self, "提示", "反射率文件为空或格式错误！")
                return

            # ------------------ 4. 光谱响应函数 ------------------
            SpcRspfile = os.path.join(os.getcwd(), "./Release/SpecResponseGF5DPC.csv")
            if not os.path.exists(SpcRspfile):
                QMessageBox.warning(self, "提示", f"找不到光谱响应函数文件: {SpcRspfile}")
                return

            WavelgtSpcRsps, SpcRsps = [], []
            with open(SpcRspfile, "r", encoding="utf-8") as f:
                next(f)  # 跳过表头
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) > band_index:
                        WavelgtSpcRsps.append(float(parts[0]))
                        SpcRsps.append(float(parts[band_index]))

            # 导入AFD文件
            dll_path = os.path.join(os.getcwd(), "Release/AFieldDll.dll")
            clr.AddReference(dll_path)
            import AFieldDll
            AFdll = AFieldDll.AFieldDLL()

            StepedRef = AFdll.LinearInterPolation(
                List[Double](Array[Double](WaveLengthField)),
                List[Double](Array[Double](RefField)),
                WavelgtSpcRsps[0],
                WavelgtSpcRsps[len(WavelgtSpcRsps) - 1],
                1.0
            )

            # 地表反射率和光谱响应函数的卷积和
            Ref_Conv = AFdll.Convolution(
                StepedRef,
                List[Double](Array[Double](SpcRsps)),
                List[Double](Array[Double](WavelgtSpcRsps))
            )

            # ------------------ 5. 调用 6S ------------------
            SpcRspfile_txt = os.path.join(os.getcwd(), "Release/" + self.band_combo.currentText()[:3] + ".txt")
            Ltoa = AFdll.call6SDesert(
                SpcRspfile_txt,
                float(self.sun_el.text() or 0),  # SolarZenith
                float(self.sun_az.text() or 0),  # SolarAzimuth
                float(self.sat_zn.text() or 0),  # ViewZenith
                float(self.sat_az.text() or 0),  # ViewAzimuth
                self.date_edit.date().month(),
                self.date_edit.date().day(),
                aod,
                Ref_Conv,
                float(self.range_km.text() or 0)  # Altitude
            )

            # ------------------ 6. 计算定标系数 ------------------
            coef = Ltoa / DN_mean

            # ------------------ 7. 输出结果 ------------------
            result = (
                f"波段数量为: {band_count}\n"
                f"波段为: {band_text} (index={band_index})\n"
                f"区域中心像素: 行={row_c}, 列={col_c}\n"
                f"区域中心像元为: {center_DN}\n"
                f"定标区域DN 平均值: {DN_mean:.6f}\n"
                f"地表反射率卷积: {Ref_Conv}\n"
                f"表观辐亮度 (Ltoa): {Ltoa}\n"
                f"定标系数: {coef:.6f}\n"
            )
            self.result_box.setText(result)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"计算失败: {e}")
            import traceback
            self.result_box.setText(traceback.format_exc())

    def show_help(self):
        help_text = """
高分五号 DPC 绝对辐射定标工具使用说明：

1. 选择图像信息
   - 点击"选择影像"加载遥感图像
   - 点击"选择中心点"按钮进入画框模式
   - 在图像上拖动鼠标选择定标区域
   - 或手动输入中心行列和子图像大小

2. 输入传感器信息
   - 选择需要定标的波段

3. 输入几何条件信息
   - 手动输入太阳方位角、天顶角
   - 输入观测方位角、天顶角
   - 或点击"读取XML"从元数据文件导入

4. 输入实测地气参数
   - 选择地表反射率文件（txt/csv格式）
   - 输入气溶胶光学厚度（550nm）

5. 点击"计算定标系数"开始计算

注意事项：
- 大尺寸图像会自动生成缩略图显示
- 坐标会自动在显示坐标和原始坐标间转换
- 确保Release文件夹包含所需的光谱响应函数和DLL文件
        """
        QMessageBox.information(self, "帮助", help_text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CalibrationApp()
    window.show()
    sys.exit(app.exec_())