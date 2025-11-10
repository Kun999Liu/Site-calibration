import os, sys
import xml.etree.ElementTree as ET
import clr

clr.AddReference("System")
from System import Array, Double
from System.Collections.Generic import List

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QComboBox,
    QDateEdit, QTextEdit, QMessageBox, QSplitter, QRubberBand, QDialog
)
from PyQt5.QtCore import QDate, Qt, QRect, QSize, QPointF
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont
from osgeo import gdal
import numpy as np


class SpectralCurveDialog(QDialog):
    """光谱响应曲线弹窗 - 使用Qt原生绘图"""

    def __init__(self, wavelengths, spectral_response, reflectance_curve, ref_conv, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ref_SpcRsp_Curves")
        self.setGeometry(100, 100, 900, 600)

        self.wavelengths = wavelengths
        self.spectral_response = spectral_response
        self.reflectance_curve = reflectance_curve
        self.ref_conv = ref_conv

        layout = QVBoxLayout()

        # 标题
        title_label = QLabel(f'地表等效反射率为: {ref_conv:.12f}')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold; padding: 10px;")
        layout.addWidget(title_label)

        # 绘图区域
        self.canvas = PlotCanvas(wavelengths, spectral_response, reflectance_curve)
        layout.addWidget(self.canvas)

        # 关闭按钮
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        self.setLayout(layout)


class PlotCanvas(QLabel):
    """使用Qt原生绘图的画布"""

    def __init__(self, wavelengths, spectral_response, reflectance_curve):
        super().__init__()
        self.wavelengths = wavelengths
        self.spectral_response = spectral_response
        self.reflectance_curve = reflectance_curve

        self.setMinimumSize(800, 500)
        self.setStyleSheet("background-color: white; border: 1px solid gray;")

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 设置绘图区域
        width = self.width()
        height = self.height()
        margin_left = 80
        margin_right = 40
        margin_top = 40
        margin_bottom = 60

        plot_width = width - margin_left - margin_right
        plot_height = height - margin_top - margin_bottom

        # 绘制坐标轴
        painter.setPen(QPen(QColor(0, 0, 0), 2))
        # Y轴
        painter.drawLine(margin_left, margin_top, margin_left, height - margin_bottom)
        # X轴
        painter.drawLine(margin_left, height - margin_bottom, width - margin_right, height - margin_bottom)

        # 绘制网格和刻度
        painter.setPen(QPen(QColor(200, 200, 200), 1))

        # X轴刻度 (400-1100 nm)
        x_min, x_max = 400, 1100
        x_step = 100
        for x_val in range(x_min, x_max + 1, x_step):
            x_pos = margin_left + (x_val - x_min) / (x_max - x_min) * plot_width
            painter.drawLine(int(x_pos), margin_top, int(x_pos), height - margin_bottom)
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            painter.drawText(int(x_pos - 15), height - margin_bottom + 20, str(x_val))
            painter.setPen(QPen(QColor(200, 200, 200), 1))

        # Y轴刻度 (0-1.0)
        y_min, y_max = 0, 1.0
        y_step = 0.2
        y_val = y_min
        while y_val <= y_max + 0.01:
            y_pos = height - margin_bottom - (y_val - y_min) / (y_max - y_min) * plot_height
            painter.drawLine(margin_left, int(y_pos), width - margin_right, int(y_pos))
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            painter.drawText(margin_left - 40, int(y_pos + 5), f'{y_val:.1f}')
            painter.setPen(QPen(QColor(200, 200, 200), 1))
            y_val += y_step

        # 绘制光谱响应曲线（红色）
        if len(self.wavelengths) > 0:
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            points = []
            for i in range(len(self.wavelengths)):
                x = self.wavelengths[i]
                y = self.spectral_response[i]
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    x_pos = margin_left + (x - x_min) / (x_max - x_min) * plot_width
                    y_pos = height - margin_bottom - (y - y_min) / (y_max - y_min) * plot_height
                    points.append(QPointF(x_pos, y_pos))

            for i in range(len(points) - 1):
                painter.drawLine(points[i], points[i + 1])

        # 绘制地物反射率曲线（蓝色）
        if len(self.reflectance_curve[0]) > 0:
            painter.setPen(QPen(QColor(0, 0, 255), 2))
            points = []
            for i in range(len(self.reflectance_curve[0])):
                x = self.reflectance_curve[0][i]
                y = self.reflectance_curve[1][i]
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    x_pos = margin_left + (x - x_min) / (x_max - x_min) * plot_width
                    y_pos = height - margin_bottom - (y - y_min) / (y_max - y_min) * plot_height
                    points.append(QPointF(x_pos, y_pos))

            for i in range(len(points) - 1):
                painter.drawLine(points[i], points[i + 1])

        # 绘制坐标轴标签
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        font = QFont()
        font.setPointSize(10)
        painter.setFont(font)

        # X轴标签
        painter.drawText(width // 2 - 30, height - 10, "波长 (nm)")

        # Y轴标签（旋转）
        painter.save()
        painter.translate(15, height // 2)
        painter.rotate(-90)
        painter.drawText(-50, 0, "反射率")
        painter.restore()

        # 图例
        legend_x = width - 200
        legend_y = 60

        painter.setPen(QPen(QColor(255, 0, 0), 2))
        painter.drawLine(legend_x, legend_y, legend_x + 40, legend_y)
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.drawText(legend_x + 50, legend_y + 5, "光谱响应曲线")

        painter.setPen(QPen(QColor(0, 0, 255), 2))
        painter.drawLine(legend_x, legend_y + 25, legend_x + 40, legend_y + 25)
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.drawText(legend_x + 50, legend_y + 30, "地物反射率谱曲线")


class CalibrationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("高分二号PMS1绝对辐射定标-场地定标")
        self.setGeometry(200, 100, 1000, 600)
        self.pick_mode = False
        self.image = None
        self.display_scale = 1.0
        self.valid_sizes = ["3", "5", "9", "13", "19", "25", "33", "43", "51"]
        self.rubber_rect = QRect()
        self.Ltoa = 0

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
        self.sub_size_row.setCurrentText("5")

        # 列方向：跟随，不可编辑
        self.sub_size_col = QComboBox()
        self.sub_size_col.setEditable(False)
        self.sub_size_col.addItem("5")

        def sync_sizes(text):
            try:
                val = int(text)
            except ValueError:
                return

            if hasattr(self, "image") and self.image is not None:
                h, w = self.image.shape[:2]
                max_size = min(h, w)
                val = max(1, min(val, max_size))

            self.sub_size_row.blockSignals(True)
            self.sub_size_row.setCurrentText(str(val))
            self.sub_size_row.blockSignals(False)
            self.sub_size_col.setItemText(0, str(val))

        self.sub_size_row.currentTextChanged.connect(sync_sizes)
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
        self.band_combo.addItems(["B1", "B2", "B3", "B4"])
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
        h11.addWidget(QLabel("气溶胶光学厚度:"))
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
        splitter.setSizes([100, 900])

        # 设置主布局
        layout = QHBoxLayout()
        layout.addWidget(splitter)
        self.setLayout(layout)

        self.sub_size_row.currentTextChanged.connect(self.update_rubber_from_combo)

    def activate_pick_mode(self):
        if self.image_display.pixmap() is None:
            QMessageBox.warning(self, "错误", "请先加载一幅图像。")
            return
        self.pick_mode = not self.pick_mode
        self.setCursor(Qt.CrossCursor if self.pick_mode else Qt.ArrowCursor)
        msg = "画框模式已开启，请拖动鼠标选择区域。" if self.pick_mode else "画框模式已关闭。"
        QMessageBox.information(self, "提示", msg)

    def update_rubber_from_combo(self, text):
        if self.image is None:
            return
        try:
            size = int(text)
        except ValueError:
            return

        img_h, img_w = self.image.shape[:2]

        if self.rubber_rect.isNull():
            cx, cy = int(img_w * self.display_scale / 2), int(img_h * self.display_scale / 2)
        else:
            cx, cy = self.rubber_rect.center().x(), self.rubber_rect.center().y()

        display_size = int(size * self.display_scale)
        half = display_size // 2

        if self.image_display.pixmap():
            display_w = self.image_display.pixmap().width()
            display_h = self.image_display.pixmap().height()

            cx = max(half, min(cx, display_w - half - 1))
            cy = max(half, min(cy, display_h - half - 1))

            new_rect = QRect(cx - half, cy - half, display_size, display_size)
            self.rubber_rect = new_rect
            self.image_display.update_rubber(self.rubber_rect)

            orig_cx = int(cx / self.display_scale)
            orig_cy = int(cy / self.display_scale)
            self.center_col.setText(str(orig_cx))
            self.center_row.setText(str(orig_cy))
            self.sub_size_col.setItemText(0, str(size))

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

                main_win.rubber_rect = rect

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
            dx = current_pos.x() - self.start_pos.x()
            dy = current_pos.y() - self.start_pos.y()
            side = max(abs(dx), abs(dy))

            if self.pixmap():
                display_w = self.pixmap().width()
                display_h = self.pixmap().height()
            else:
                return QRect(self.start_pos, QSize())

            x = self.start_pos.x()
            y = self.start_pos.y()
            if dx < 0: x -= side
            if dy < 0: y -= side

            if x < 0: x = 0
            if y < 0: y = 0
            if x + side > display_w: side = display_w - x
            if y + side > display_h: side = display_h - y

            return QRect(x, y, side, side)

        def update_rubber(self, rect):
            self.rubber_band.setGeometry(rect)
            self.rubber_band.show()

    def open_image(self):
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

                band_count = dataset.RasterCount

                # 读取完整影像数据用于计算（保持原样）
                bands = min(3, band_count)
                img_data = []
                for i in range(1, bands + 1):
                    band = dataset.GetRasterBand(i).ReadAsArray()
                    img_data.append(band)
                img_data = np.dstack(img_data).astype(np.float32)
                self.image = img_data

                # ========== 生成RGB显示缩略图 ==========
                max_display_size = 800
                h, w = img_data.shape[:2]

                scale = min(max_display_size / w, max_display_size / h, 1.0)
                new_w, new_h = int(w * scale), int(h * scale)
                self.display_scale = scale

                # 根据波段数量选择RGB合成方案
                if band_count >= 3:
                    # 有3个或更多波段：B1(蓝), B2(绿), B3(红)
                    band_indices = [3, 2, 1]  # R, G, B
                elif band_count == 2:
                    # 只有2个波段：B1(蓝), B2(绿), 0(红)
                    band_indices = [1, 2, None]
                else:
                    # 只有1个波段：灰度显示
                    band_indices = [1, 1, 1]

                # 读取并组合RGB通道
                rgb_channels = []
                for band_idx in band_indices:
                    if band_idx is None:
                        # 蓝色通道缺失，用0填充
                        zero_band = np.zeros((new_h, new_w), dtype=np.float32)
                        rgb_channels.append(zero_band)
                    else:
                        # 使用 gdal.Translate 缩放单个波段
                        thumbnail = gdal.Translate(
                            '', dataset,
                            format='MEM',
                            width=new_w,
                            height=new_h,
                            bandList=[band_idx]
                        )
                        band_data = thumbnail.GetRasterBand(1).ReadAsArray().astype(np.float32)
                        rgb_channels.append(band_data)

                # 合成RGB图像
                thumb_data = np.dstack(rgb_channels)

                def envi_optimized_linear(channel):
                    # 拉成一维
                    flat = channel.flatten()

                    # 计算直方图
                    hist, bins = np.histogram(flat, bins=2000)

                    # 防止某些值出现极端高频(噪点)，限制其影响
                    threshold = np.percentile(hist, 99)
                    hist = np.minimum(hist, threshold)

                    # 累计分布
                    cdf = np.cumsum(hist) / np.sum(hist)

                    # 取最适拉伸上下限(类似 ENVI optimized linear)
                    low_idx = np.searchsorted(cdf, 0.005)
                    high_idx = np.searchsorted(cdf, 0.995)

                    p_low = bins[low_idx]
                    p_high = bins[min(high_idx, len(bins) - 1)]

                    # 线性拉伸
                    out = (channel - p_low) / (p_high - p_low + 1e-6)
                    return np.clip(out, 0, 1)

                # 你的原循环替换成下面这段
                for i in range(3):
                    thumb_data[:, :, i] = envi_optimized_linear(thumb_data[:, :, i])

                thumb_data *= 255
                thumb_data = thumb_data.astype(np.uint8)

                h, w, c = thumb_data.shape
                qimg = QImage(thumb_data.data, w, h, 3 * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)

            except Exception as e:
                QMessageBox.critical(self, "错误", f"读取tif失败: {e}")
                return
        else:
            pixmap = QPixmap(file)
            if not pixmap.isNull():
                max_display_size = 800
                orig_w, orig_h = pixmap.width(), pixmap.height()

                if orig_w > max_display_size or orig_h > max_display_size:
                    self.display_scale = min(max_display_size / orig_w, max_display_size / orig_h)
                    pixmap = pixmap.scaled(max_display_size, max_display_size,
                                           Qt.KeepAspectRatio, Qt.SmoothTransformation)
                else:
                    self.display_scale = 1.0

                qimg = pixmap.toImage()
                ptr = qimg.bits()
                ptr.setsize(qimg.byteCount())
                self.image = np.array(ptr).reshape(qimg.height(), qimg.width(), 4)

        if not pixmap.isNull():
            self.image_display.setPixmap(pixmap)
            self.image_display.adjustSize()
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
        """计算定标系数并显示光谱曲线"""
        try:
            # 读取基本输入
            img_file = self.image_path.text().strip()
            if not img_file:
                QMessageBox.warning(self, "提示", "请先选择影像文件！")
                return

            try:
                sub_size = int(self.sub_size_row.currentText())
            except Exception:
                QMessageBox.warning(self, "提示", "子图像大小不正确！")
                return

            try:
                row_c = int(self.center_row.text())
                col_c = int(self.center_col.text())
            except Exception:
                QMessageBox.warning(self, "提示", "请输入中心像素行列号！")
                return

            try:
                aod = float(self.aod_550.text())
            except:
                QMessageBox.warning(self, "提示", "请输入气溶胶光学厚度和几何条件信息")
                return

            # 读取子区域 DN
            dataset = gdal.Open(img_file, gdal.GA_ReadOnly)
            if dataset is None:
                QMessageBox.critical(self, "错误", f"无法打开文件: {img_file}")
                return

            band_text = self.band_combo.currentText()
            band_index = self.band_combo.currentIndex() + 2
            band_count = dataset.RasterCount
            if band_count == 0:
                QMessageBox.critical(self, "错误", f"影像文件 {img_file} 无波段！")
                return

            band = dataset.GetRasterBand(band_index)
            if band is None:
                QMessageBox.critical(self, "错误", f"无法读取第 {band_index} 波段！")
                return

            half = sub_size // 2
            img_h, img_w = self.image.shape[:2]

            if sub_size > min(img_h, img_w):
                sub_size = min(img_h, img_w)
                half = sub_size // 2
                self.sub_size_row.setCurrentText(str(sub_size))
                self.sub_size_col.setItemText(0, str(sub_size))

            row_c = max(half, min(row_c, img_h - half - 1))
            col_c = max(half, min(col_c, img_w - half - 1))

            xoff, yoff = col_c - half, row_c - half
            data = band.ReadAsArray(xoff, yoff, sub_size, sub_size).astype(float)

            center_DN = data[half, half]
            DN_mean = float(np.mean(data))

            # 地表反射率文件
            ref_file = self.reflect_file.text().strip()
            if not ref_file:
                QMessageBox.warning(self, "提示", "请选择地表反射率文件！")
                return

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

            # 光谱响应函数
            SpcRspfile = os.path.join(os.getcwd(), "./SpecRsp/GF2/GF-2 PMS1.csv")
            if not os.path.exists(SpcRspfile):
                QMessageBox.warning(self, "提示", f"找不到光谱响应函数文件: {SpcRspfile}")
                return

            WavelgtSpcRsps, SpcRsps = [], []
            with open(SpcRspfile, "r", encoding="utf-8") as f:
                next(f)
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) > band_index + 1:
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

            # 调用 6S
            SpcRspfile_txt = os.path.join(os.getcwd(), "Release/" + self.band_combo.currentText()[:3] + ".txt")
            Ltoa = AFdll.call6SDesert(
                SpcRspfile_txt,
                float(self.sun_el.text() or 0),
                float(self.sun_az.text() or 0),
                float(self.sat_zn.text() or 0),
                float(self.sat_az.text() or 0),
                self.date_edit.date().month(),
                self.date_edit.date().day(),
                aod,
                Ref_Conv,
                float(self.range_km.text() or 0)
            )

            # 计算定标系数
            coef = Ltoa / DN_mean

            # 输出结果
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

            # 显示光谱曲线弹窗
            dialog = SpectralCurveDialog(
                WavelgtSpcRsps,
                SpcRsps,
                (WaveLengthField, RefField),
                Ref_Conv,
                self
            )
            dialog.exec_()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"计算失败: {e}")
            import traceback
            self.result_box.setText(traceback.format_exc())

    def show_help(self):
        help_text = """
高分二号PMS1绝对辐射定标工具使用说明：

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
   - 输入气溶胶光学厚度

5. 点击"计算定标系数"开始计算
   - 计算完成后会显示光谱响应曲线弹窗

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
