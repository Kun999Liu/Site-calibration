# -*- coding: utf-8 -*-
"""
反射率提取验证工具 - PyQt5图形界面（美化版）
"""
import sys
import os
import xml.etree.ElementTree as ET
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QLineEdit,
                             QTextEdit, QFileDialog, QSpinBox, QGroupBox,
                             QProgressBar, QSplitter, QScrollArea, QSlider,
                             QListWidget, QListWidgetItem, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint, QRectF, QSize
from PyQt5.QtGui import (QPixmap, QImage, QFont, QPainter, QPen, QColor,
                         QBrush, QCursor, QTransform)
import numpy as np
from osgeo import gdal
import warnings

warnings.filterwarnings("ignore")

# 导入原有的处理类
from Fuse_ARDVal import ReflectanceExtractor_Val


class ConfigLoader:
    """XML配置文件加载器"""

    @staticmethod
    def load_config(xml_path):
        """加载XML配置文件"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            config = {
                'image_folder': root.find('image_folder').text.strip() if root.find('image_folder') is not None else '',
                'excel_path': root.find('excel_path').text.strip() if root.find('excel_path') is not None else '',
                'output_path': root.find('output_path').text.strip() if root.find('output_path') is not None else '',
                'scale_factor': int(root.find('scale_factor').text) if root.find('scale_factor') is not None else 10000,
                'time_threshold': int(root.find('time_threshold').text) if root.find(
                    'time_threshold') is not None else 3,
            }

            # 处理相对路径
            base_dir = os.path.dirname(os.path.abspath(xml_path))
            for key in ['image_folder', 'excel_path', 'output_path']:
                if config[key] and not os.path.isabs(config[key]):
                    config[key] = os.path.normpath(os.path.join(base_dir, config[key]))

            return config
        except Exception as e:
            raise Exception(f"加载配置文件失败: {str(e)}")

    @staticmethod
    def save_config(xml_path, config):
        """保存配置到XML文件"""
        try:
            root = ET.Element('config')

            comment = ET.Comment(' 路径可以是相对的，也可以是绝对的 ')
            root.append(comment)

            for key, value in config.items():
                element = ET.SubElement(root, key)
                element.text = str(value)

            tree = ET.ElementTree(root)
            ET.indent(tree, space="    ")
            tree.write(xml_path, encoding='utf-8', xml_declaration=True)
            return True
        except Exception as e:
            print(f"保存配置文件失败: {str(e)}")
            return False


class ProcessThread(QThread):
    """处理线程"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, extractor, excel_path, output_path):
        super().__init__()
        self.extractor = extractor
        self.excel_path = excel_path
        self.output_path = output_path

    def run(self):
        try:
            self.progress.emit("开始扫描影像文件...")
            num_images = self.extractor.scan_images()

            if num_images > 0:
                self.progress.emit(f"找到 {num_images} 个影像文件")
                self.progress.emit("开始处理Excel数据...")

                results = self.extractor.process_excel(self.excel_path, self.output_path)

                if len(results) > 0:
                    self.finished.emit(True, f"处理完成！共匹配 {len(results)} 个点")
                else:
                    self.finished.emit(False, "未找到任何匹配点")
            else:
                self.finished.emit(False, "未找到影像文件，请检查路径")
        except Exception as e:
            self.finished.emit(False, f"处理出错: {str(e)}")


class ImageViewer(QLabel):
    """影像显示组件（支持缩放和点击）"""
    point_clicked = pyqtSignal(dict)  # 点击点时发出信号

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5;
                border: 2px solid #4A90E2;
                border-radius: 10px;
            }
        """)
        self.setMinimumSize(600, 600)
        self.setScaledContents(False)

        # 缩放相关
        self.scale = 1.0
        self.original_pixmap = None
        self.points = []  # 存储点位信息 [{'x': x, 'y': y, 'info': {...}}]
        self.image_size = None  # 原始图像尺寸

        # 鼠标交互
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)

        self.show_placeholder()

    def show_placeholder(self):
        """显示占位符"""
        placeholder = QPixmap(600, 600)
        placeholder.fill(QColor("#f5f5f5"))
        painter = QPainter(placeholder)
        painter.setPen(QColor("#999999"))
        font = QFont("Microsoft YaHei", 14)
        painter.setFont(font)
        painter.drawText(placeholder.rect(), Qt.AlignCenter,
                         "影像预览区\n\n点击'扫描影像'加载图像")
        painter.end()
        self.setPixmap(placeholder)

    def load_tif(self, tif_path, points_data=None):
        """加载并显示TIF影像"""
        try:
            dataset = gdal.Open(tif_path)
            if dataset is None:
                return False

            # 获取影像信息
            cols = dataset.RasterXSize
            rows = dataset.RasterYSize
            bands = dataset.RasterCount

            self.image_size = (cols, rows)

            # 读取影像数据
            if bands >= 3:
                # 尝试读取RGB波段
                try:
                    r_band = dataset.GetRasterBand(min(3, bands)).ReadAsArray()
                    g_band = dataset.GetRasterBand(min(2, bands)).ReadAsArray()
                    b_band = dataset.GetRasterBand(1).ReadAsArray()
                except:
                    # 如果失败，使用第一个波段
                    gray = dataset.GetRasterBand(1).ReadAsArray()
                    r_band = g_band = b_band = gray
            else:
                gray = dataset.GetRasterBand(1).ReadAsArray()
                r_band = g_band = b_band = gray

            # 归一化到0-255
            def normalize(band):
                if band is None:
                    return np.zeros((rows, cols), dtype=np.uint8)
                # 过滤无效值
                valid_data = band[np.isfinite(band)]
                if len(valid_data) == 0:
                    return np.zeros_like(band, dtype=np.uint8)
                vmin, vmax = np.percentile(valid_data, (2, 98))
                if vmax == vmin:
                    return np.zeros_like(band, dtype=np.uint8)
                band = np.clip((band - vmin) / (vmax - vmin) * 255, 0, 255)
                return band.astype(np.uint8)

            r_band = normalize(r_band)
            g_band = normalize(g_band)
            b_band = normalize(b_band)

            # 创建RGB图像
            height, width = r_band.shape
            rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
            rgb_image[:, :, 0] = r_band
            rgb_image[:, :, 1] = g_band
            rgb_image[:, :, 2] = b_band

            # 转换为QImage和QPixmap
            bytes_per_line = 3 * width
            qimage = QImage(rgb_image.data, width, height, bytes_per_line,
                            QImage.Format_RGB888)

            # 确保数据不会被释放
            self.rgb_data = rgb_image
            self.original_pixmap = QPixmap.fromImage(qimage.copy())

            # 保存点位信息
            self.points = points_data if points_data else []

            # 重置缩放
            self.scale = 1.0
            self.update_display()

            dataset = None
            return True

        except Exception as e:
            print(f"影像显示错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def update_display(self):
        """更新显示（应用缩放和绘制点）"""
        if self.original_pixmap is None:
            return

        # 先缩放到合适大小以适应窗口
        available_size = self.size()
        scaled_to_fit = self.original_pixmap.scaled(
            available_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        # 再应用用户缩放
        final_size = QSize(int(scaled_to_fit.width() * self.scale),
                           int(scaled_to_fit.height() * self.scale))
        scaled_pixmap = self.original_pixmap.scaled(
            final_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        # 在图像上绘制点
        if self.points:
            scaled_pixmap = self.draw_points(scaled_pixmap)

        self.setPixmap(scaled_pixmap)

    def draw_points(self, pixmap):
        """在图像上绘制点"""
        if not self.points or not self.image_size:
            return pixmap

        result = QPixmap(pixmap)
        painter = QPainter(result)
        painter.setRenderHint(QPainter.Antialiasing)

        # 计算缩放比例
        scale_x = pixmap.width() / self.image_size[0]
        scale_y = pixmap.height() / self.image_size[1]

        for point in self.points:
            # 计算在缩放后图像上的坐标
            x = point['x'] * scale_x
            y = point['y'] * scale_y

            # 绘制外圈（白色边框）
            painter.setPen(QPen(QColor(255, 255, 255), 4))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPoint(int(x), int(y)), 10, 10)

            # 绘制内圈
            painter.setPen(QPen(QColor(74, 144, 226), 3))
            painter.setBrush(QBrush(QColor(74, 144, 226, 180)))
            painter.drawEllipse(QPoint(int(x), int(y)), 7, 7)

            # 绘制中心点
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.drawEllipse(QPoint(int(x), int(y)), 2, 2)

        painter.end()
        return result

    def wheelEvent(self, event):
        """鼠标滚轮缩放"""
        if self.original_pixmap is None:
            return

        # 计算缩放因子
        delta = event.angleDelta().y()
        if delta > 0:
            self.scale *= 1.1
        else:
            self.scale /= 1.1

        # 限制缩放范围
        self.scale = max(0.5, min(self.scale, 5.0))

        self.update_display()

    def mousePressEvent(self, event):
        """鼠标点击事件"""
        if event.button() == Qt.LeftButton and self.points and self.pixmap():
            # 获取点击位置
            click_pos = event.pos()
            pixmap_rect = self.pixmap().rect()

            # 计算图像在Label中的位置（居中显示）
            label_rect = self.rect()
            x_offset = (label_rect.width() - pixmap_rect.width()) // 2
            y_offset = (label_rect.height() - pixmap_rect.height()) // 2

            # 转换为图像坐标
            img_x = click_pos.x() - x_offset
            img_y = click_pos.y() - y_offset

            # 检查是否在图像范围内
            if img_x < 0 or img_y < 0 or img_x >= pixmap_rect.width() or img_y >= pixmap_rect.height():
                return

            # 计算原始图像坐标
            if self.image_size:
                scale_x = pixmap_rect.width() / self.image_size[0]
                scale_y = pixmap_rect.height() / self.image_size[1]

                orig_x = img_x / scale_x
                orig_y = img_y / scale_y

                # 查找最近的点
                min_dist = float('inf')
                nearest_point = None

                for point in self.points:
                    dist = np.sqrt((point['x'] - orig_x) ** 2 + (point['y'] - orig_y) ** 2)
                    if dist < min_dist and dist < 30:  # 30像素内
                        min_dist = dist
                        nearest_point = point

                if nearest_point:
                    self.point_clicked.emit(nearest_point)

    def set_scale(self, scale):
        """设置缩放比例"""
        self.scale = scale
        self.update_display()

    def resizeEvent(self, event):
        """窗口大小改变时重新显示图像"""
        super().resizeEvent(event)
        if self.original_pixmap:
            self.update_display()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("反射率提取验证工具 v2.0")
        self.setGeometry(100, 100, 1600, 900)

        # 应用全局样式
        self.setStyleSheet(self.get_stylesheet())

        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # 左侧控制面板
        left_widget = self.create_control_panel()
        main_layout.addWidget(left_widget, stretch=2)

        # 右侧显示区域
        right_widget = self.create_display_panel()
        main_layout.addWidget(right_widget, stretch=3)

        # 初始化变量
        self.extractor = None
        self.current_image_index = 0
        self.matched_points = {}  # {image_index: [points]}

    def create_control_panel(self):
        """创建左侧控制面板"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)

        # 标题
        title = QLabel("反射率提取验证配置")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # 配置文件加载
        config_group = QGroupBox("📄 配置文件")
        config_layout = QVBoxLayout()

        config_btn_layout = QHBoxLayout()
        load_config_btn = QPushButton("📂 加载配置")
        load_config_btn.clicked.connect(self.load_config_file)
        config_btn_layout.addWidget(load_config_btn)

        save_config_btn = QPushButton("💾 保存配置")
        save_config_btn.clicked.connect(self.save_config_file)
        config_btn_layout.addWidget(save_config_btn)
        config_layout.addLayout(config_btn_layout)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # 路径配置组
        path_group = QGroupBox("📁 路径配置")
        path_layout = QVBoxLayout()

        self.image_folder_edit = self.create_path_input(
            "影像文件夹:", "选择", path_layout
        )

        self.excel_path_edit = self.create_path_input(
            "实测Excel:", "选择", path_layout, is_file=True
        )

        self.output_path_edit = self.create_path_input(
            "输出路径:", "保存", path_layout, is_file=True, save=True
        )

        path_group.setLayout(path_layout)
        layout.addWidget(path_group)

        # 参数配置组
        param_group = QGroupBox("⚙️ 处理参数")
        param_layout = QVBoxLayout()

        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("反射率比例因子:"))
        self.scale_factor_spin = QSpinBox()
        self.scale_factor_spin.setRange(1, 100000)
        self.scale_factor_spin.setValue(10000)
        self.scale_factor_spin.setSingleStep(1000)
        scale_layout.addWidget(self.scale_factor_spin)
        param_layout.addLayout(scale_layout)

        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("时间阈值(天):"))
        self.time_threshold_spin = QSpinBox()
        self.time_threshold_spin.setRange(1, 30)
        self.time_threshold_spin.setValue(3)
        time_layout.addWidget(self.time_threshold_spin)
        param_layout.addLayout(time_layout)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # 操作按钮
        btn_layout = QHBoxLayout()
        self.scan_btn = QPushButton("🔍 扫描影像")
        self.scan_btn.setObjectName("primaryButton")
        self.scan_btn.clicked.connect(self.scan_images)
        btn_layout.addWidget(self.scan_btn)

        self.process_btn = QPushButton("▶️ 开始处理")
        self.process_btn.setObjectName("successButton")
        self.process_btn.clicked.connect(self.start_processing)
        btn_layout.addWidget(self.process_btn)
        layout.addLayout(btn_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # 日志区域
        log_group = QGroupBox("📋 处理日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        layout.addStretch()
        return widget

    def create_display_panel(self):
        """创建右侧显示面板"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        # 影像切换控制
        control_group = QGroupBox("🖼️ 影像控制")
        control_layout = QVBoxLayout()

        # 切换按钮
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("⬅️ 上一幅")
        self.prev_btn.clicked.connect(self.show_previous_image)
        self.prev_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_btn)

        self.image_label = QLabel("暂无影像")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("font-weight: bold; color: #4A90E2;")
        nav_layout.addWidget(self.image_label)

        self.next_btn = QPushButton("下一幅 ➡️")
        self.next_btn.clicked.connect(self.show_next_image)
        self.next_btn.setEnabled(False)
        nav_layout.addWidget(self.next_btn)
        control_layout.addLayout(nav_layout)

        # 缩放控制
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("缩放:"))

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 500)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setTickPosition(QSlider.TicksBelow)
        self.zoom_slider.setTickInterval(50)
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        zoom_layout.addWidget(self.zoom_slider)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(50)
        zoom_layout.addWidget(self.zoom_label)

        reset_btn = QPushButton("重置")
        reset_btn.clicked.connect(self.reset_zoom)
        zoom_layout.addWidget(reset_btn)
        control_layout.addLayout(zoom_layout)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # 影像显示器
        self.image_viewer = ImageViewer()
        self.image_viewer.point_clicked.connect(self.on_point_clicked)
        layout.addWidget(self.image_viewer, stretch=1)

        # 点位信息显示
        info_group = QGroupBox("📍 点位信息")
        info_layout = QVBoxLayout()
        self.point_info_text = QTextEdit()
        self.point_info_text.setReadOnly(True)
        self.point_info_text.setMaximumHeight(150)
        self.point_info_text.setText("点击影像上的点查看详细信息")
        info_layout.addWidget(self.point_info_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        return widget

    def create_path_input(self, label_text, btn_text, parent_layout,
                          is_file=False, save=False):
        """创建路径输入行"""
        h_layout = QHBoxLayout()
        label = QLabel(label_text)
        label.setMinimumWidth(100)
        h_layout.addWidget(label)

        line_edit = QLineEdit()
        line_edit.setPlaceholderText("请选择...")
        h_layout.addWidget(line_edit)

        btn = QPushButton(btn_text)
        btn.setMaximumWidth(80)

        if is_file:
            if save:
                btn.clicked.connect(lambda: self.select_save_file(line_edit))
            else:
                btn.clicked.connect(lambda: self.select_file(line_edit))
        else:
            btn.clicked.connect(lambda: self.select_folder(line_edit))

        h_layout.addWidget(btn)
        parent_layout.addLayout(h_layout)
        return line_edit

    def select_folder(self, line_edit):
        """选择文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder:
            line_edit.setText(folder)

    def select_file(self, line_edit):
        """选择文件"""
        file, _ = QFileDialog.getOpenFileName(
            self, "选择文件", "", "Excel Files (*.xlsx *.xls)"
        )
        if file:
            line_edit.setText(file)

    def select_save_file(self, line_edit):
        """选择保存文件"""
        file, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "", "Excel Files (*.xlsx)"
        )
        if file:
            if not file.endswith('.xlsx'):
                file += '.xlsx'
            line_edit.setText(file)

    def log(self, message):
        """添加日志"""
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def load_config_file(self):
        """加载XML配置文件"""
        file, _ = QFileDialog.getOpenFileName(
            self, "选择配置文件", "", "XML Files (*.xml)"
        )
        if file:
            try:
                config = ConfigLoader.load_config(file)

                # 更新界面
                self.image_folder_edit.setText(config['image_folder'])
                self.excel_path_edit.setText(config['excel_path'])
                self.output_path_edit.setText(config['output_path'])
                self.scale_factor_spin.setValue(config['scale_factor'])
                self.time_threshold_spin.setValue(config['time_threshold'])

                self.log("=" * 50)
                self.log(f"✅ 成功加载配置文件: {os.path.basename(file)}")
                self.log(f"   影像文件夹: {config['image_folder']}")
                self.log(f"   Excel路径: {config['excel_path']}")
                self.log(f"   输出路径: {config['output_path']}")

            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载配置文件失败:\n{str(e)}")
                self.log(f"❌ 加载配置文件失败: {str(e)}")

    def save_config_file(self):
        """保存配置到XML文件"""
        file, _ = QFileDialog.getSaveFileName(
            self, "保存配置文件", "config.xml", "XML Files (*.xml)"
        )
        if file:
            try:
                config = {
                    'image_folder': self.image_folder_edit.text(),
                    'excel_path': self.excel_path_edit.text(),
                    'output_path': self.output_path_edit.text(),
                    'scale_factor': self.scale_factor_spin.value(),
                    'time_threshold': self.time_threshold_spin.value(),
                }

                if ConfigLoader.save_config(file, config):
                    self.log("=" * 50)
                    self.log(f"✅ 配置已保存到: {file}")
                    QMessageBox.information(self, "成功", "配置文件保存成功！")
                else:
                    raise Exception("保存失败")

            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存配置文件失败:\n{str(e)}")
                self.log(f"❌ 保存配置文件失败: {str(e)}")

    def scan_images(self):
        """扫描影像文件"""
        image_folder = self.image_folder_edit.text()

        if not image_folder or not os.path.exists(image_folder):
            self.log("❌ 请先选择有效的影像文件夹")
            return

        self.log("=" * 50)
        self.log("🔍 开始扫描影像文件...")

        try:
            self.extractor = ReflectanceExtractor_Val(
                image_folder=image_folder,
                scale_factor=self.scale_factor_spin.value(),
                time_threshold=self.time_threshold_spin.value()
            )

            num_images = self.extractor.scan_images()

            if num_images > 0:
                self.log(f"✅ 找到 {num_images} 个影像文件")
                self.current_image_index = 0
                self.update_image_display()
                self.prev_btn.setEnabled(True)
                self.next_btn.setEnabled(True)
            else:
                self.log("❌ 未找到影像文件")

        except Exception as e:
            self.log(f"❌ 扫描出错: {str(e)}")

    def update_image_display(self):
        """更新影像显示"""
        if not self.extractor or not self.extractor.images:
            return

        image_info = self.extractor.images[self.current_image_index]
        image_path = image_info['path']

        # 获取该影像的匹配点（如果有）
        points_data = self.matched_points.get(self.current_image_index, [])

        # 加载影像
        if self.image_viewer.load_tif(image_path, points_data):
            filename = os.path.basename(image_path)
            self.image_label.setText(
                f"影像 {self.current_image_index + 1}/{len(self.extractor.images)}: {filename}"
            )
            self.log(f"📷 显示: {filename} (包含 {len(points_data)} 个点)")

    def show_previous_image(self):
        """显示上一幅影像"""
        if self.extractor and self.current_image_index > 0:
            self.current_image_index -= 1
            self.update_image_display()

    def show_next_image(self):
        """显示下一幅影像"""
        if self.extractor and self.current_image_index < len(self.extractor.images) - 1:
            self.current_image_index += 1
            self.update_image_display()

    def on_zoom_changed(self, value):
        """缩放滑块变化"""
        scale = value / 100.0
        self.zoom_label.setText(f"{value}%")
        self.image_viewer.set_scale(scale)

    def reset_zoom(self):
        """重置缩放"""
        self.zoom_slider.setValue(100)

    def on_point_clicked(self, point):
        """点击点时显示信息"""
        info_text = f"""
<b>🎯 点位详细信息</b><br>
<hr>
<b>影像坐标:</b> ({point['x']:.2f}, {point['y']:.2f})<br>
"""
        if 'info' in point:
            info = point['info']
            info_text += f"<b>点位ID:</b> {info.get('id', 'N/A')}<br>"
            info_text += f"<b>测量时间:</b> {info.get('time', 'N/A')}<br>"
            info_text += f"<b>反射率:</b> {info.get('reflectance', 'N/A')}<br>"

        self.point_info_text.setHtml(info_text)

    def start_processing(self):
        """开始处理"""
        image_folder = self.image_folder_edit.text()
        excel_path = self.excel_path_edit.text()
        output_path = self.output_path_edit.text()

        if not image_folder or not os.path.exists(image_folder):
            self.log("❌ 请选择有效的影像文件夹")
            return

        if not excel_path or not os.path.exists(excel_path):
            self.log("❌ 请选择有效的Excel文件")
            return

        if not output_path:
            self.log("❌ 请指定输出路径")
            return

        self.log("=" * 50)
        self.log("▶️ 开始处理任务...")

        self.scan_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        if not self.extractor:
            self.extractor = ReflectanceExtractor_Val(
                image_folder=image_folder,
                scale_factor=self.scale_factor_spin.value(),
                time_threshold=self.time_threshold_spin.value()
            )

        self.thread = ProcessThread(self.extractor, excel_path, output_path)
        self.thread.progress.connect(self.log)
        self.thread.finished.connect(self.on_processing_finished)
        self.thread.start()

    def on_processing_finished(self, success, message):
        """处理完成回调"""
        self.log("=" * 50)
        if success:
            self.log(f"✅ {message}")
        else:
            self.log(f"❌ {message}")

        self.scan_btn.setEnabled(True)
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def get_stylesheet(self):
        """返回QSS样式表"""
        return """
            QMainWindow {
                background-color: #ffffff;
            }

            QWidget {
                background-color: #ffffff;
                color: #333333;
                font-family: "Microsoft YaHei", "Segoe UI", Arial;
                font-size: 13px;
            }

            QLabel#title {
                font-size: 26px;
                font-weight: bold;
                color: #4A90E2;
                padding: 20px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4A90E2, stop:1 #5BA3F5);
                color: white;
                border-radius: 10px;
                margin-bottom: 10px;
            }

            QGroupBox {
                border: 2px solid #E0E0E0;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 10px;
                font-weight: bold;
                color: #4A90E2;
                background-color: #FAFAFA;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
                background-color: #FAFAFA;
            }

            QLineEdit {
                background-color: #ffffff;
                border: 2px solid #E0E0E0;
                border-radius: 6px;
                padding: 8px;
                color: #333333;
            }

            QLineEdit:focus {
                border: 2px solid #4A90E2;
            }

            QPushButton {
                background-color: #F5F5F5;
                border: 2px solid #E0E0E0;
                border-radius: 6px;
                padding: 10px 20px;
                color: #333333;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #E8F4FD;
                border: 2px solid #4A90E2;
            }

            QPushButton:pressed {
                background-color: #D0E8FA;
            }

            QPushButton#primaryButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5BA3F5, stop:1 #4A90E2);
                color: white;
                border: none;
            }

            QPushButton#primaryButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #6BB0FF, stop:1 #5BA3F5);
            }

            QPushButton#successButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5CB85C, stop:1 #4CAF50);
                color: white;
                border: none;
            }

            QPushButton#successButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #6CC86C, stop:1 #5CB85C);
            }

            QSpinBox {
                background-color: #ffffff;
                border: 2px solid #E0E0E0;
                border-radius: 6px;
                padding: 6px;
                color: #333333;
            }

            QSpinBox:focus {
                border: 2px solid #4A90E2;
            }

            QTextEdit {
                background-color: #ffffff;
                border: 2px solid #E0E0E0;
                border-radius: 6px;
                padding: 10px;
                color: #333333;
            }

            QProgressBar {
                border: 2px solid #4A90E2;
                border-radius: 6px;
                text-align: center;
                background-color: #F0F0F0;
                color: #333333;
                font-weight: bold;
            }

            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4A90E2, stop:1 #5BA3F5);
                border-radius: 4px;
            }

            QSlider::groove:horizontal {
                border: 1px solid #E0E0E0;
                height: 8px;
                background: #F0F0F0;
                border-radius: 4px;
            }

            QSlider::handle:horizontal {
                background: #4A90E2;
                border: 2px solid #ffffff;
                width: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }

            QSlider::handle:horizontal:hover {
                background: #5BA3F5;
            }

            QScrollBar:vertical {
                background-color: #F5F5F5;
                width: 12px;
                margin: 0px;
                border-radius: 6px;
            }

            QScrollBar::handle:vertical {
                background-color: #4A90E2;
                min-height: 20px;
                border-radius: 6px;
            }

            QScrollBar::handle:vertical:hover {
                background-color: #5BA3F5;
            }

            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }

            QScrollBar:horizontal {
                background-color: #F5F5F5;
                height: 12px;
                margin: 0px;
                border-radius: 6px;
            }

            QScrollBar::handle:horizontal {
                background-color: #4A90E2;
                min-width: 20px;
                border-radius: 6px;
            }
        """


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 使用Fusion风格获得更好的跨平台体验
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()