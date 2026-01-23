import sys
import os
import re  # 用于正则处理乱码
import numpy as np
from osgeo import gdal  # pip install gdal

# 消除 GDAL 的 FutureWarning 警告
gdal.UseExceptions()

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QTextEdit, QLabel,
    QSplitter, QGroupBox, QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)
from PyQt5.QtCore import QProcess, Qt, QTimer, QRunnable, QThreadPool, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QImage, QPainter

# ================= 全局配置 =================
# 读取分辨率：为了支持缩放查看细节，这里设大一点
IMAGE_DISPLAY_SIZE = 1500


# ===========================================

class WorkerSignals(QObject):
    finished = pyqtSignal(str, object)
    error = pyqtSignal(str, str)


class ImageLoader(QRunnable):
    """ 后台读取线程 (包含波段修正和拉伸) """

    def __init__(self, file_path, target_size=(IMAGE_DISPLAY_SIZE, IMAGE_DISPLAY_SIZE)):
        super(ImageLoader, self).__init__()
        self.file_path = file_path
        self.target_size = target_size
        self.signals = WorkerSignals()

    def run(self):
        try:
            ds = gdal.Open(self.file_path, gdal.GA_ReadOnly)
            if not ds: raise Exception("无法打开文件")

            width = ds.RasterXSize
            height = ds.RasterYSize
            bands_count = ds.RasterCount

            # ---------------------------------------------------
            # 1. 波段修正: 尝试读取 R(3), G(2), B(1)
            # ---------------------------------------------------
            if bands_count >= 3:
                read_bands = [3, 2, 1]
            else:
                read_bands = [1]

            # ---------------------------------------------------
            # 2. 计算合适的缩放比例 (ReadAsArray 加速)
            # ---------------------------------------------------
            scale = min(self.target_size[0] / width, self.target_size[1] / height)
            buf_w = max(1, int(width * scale))
            buf_h = max(1, int(height * scale))

            # 容错读取
            try:
                data = ds.ReadAsArray(0, 0, width, height, buf_xsize=buf_w, buf_ysize=buf_h, band_list=read_bands)
            except:
                actual_bands = [i + 1 for i in range(min(3, bands_count))]
                data = ds.ReadAsArray(0, 0, width, height, buf_xsize=buf_w, buf_ysize=buf_h, band_list=actual_bands)

            if len(data.shape) == 2: data = data[np.newaxis, ...]

            # ---------------------------------------------------
            # 3. 仿 ENVI 2% 线性拉伸
            # ---------------------------------------------------
            stretched_channels = []
            for band in data:
                band = band.astype(np.float32)
                valid_pixels = band[band > 0]  # 排除背景0值

                if valid_pixels.size > 0:
                    p_min, p_max = np.percentile(valid_pixels, (2, 98))

                    # 防止除以0
                    if p_max - p_min < 1e-5:
                        p_max = p_min + 1

                    # 拉伸归一化
                    band = (band - p_min) / (p_max - p_min)
                    band = np.clip(band, 0, 1) * 255
                else:
                    band[:] = 0
                stretched_channels.append(band.astype(np.uint8))

            img_data = np.stack(stretched_channels, axis=-1)
            if img_data.shape[2] == 1: img_data = np.repeat(img_data, 3, axis=2)
            if img_data.shape[2] > 3: img_data = img_data[:, :, :3]

            h, w, ch = img_data.shape
            # .copy() 至关重要，防止内存回收导致花屏
            q_img = QImage(img_data.data, w, h, ch * w, QImage.Format_RGB888).copy()
            self.signals.finished.emit(self.file_path, q_img)

        except Exception as e:
            self.signals.error.emit(self.file_path, str(e))


# ================= 支持缩放和平移的视图类 =================
class ImageViewer(QGraphicsView):
    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = None

        # 渲染优化
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)

        # 交互设置
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setStyleSheet("background-color: #2b2b2b; border: 1px solid #444;")

    def set_image(self, q_pixmap):
        self.scene.clear()
        self.pixmap_item = QGraphicsPixmapItem(q_pixmap)
        self.scene.addItem(self.pixmap_item)
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor
        self.scale(zoom_factor, zoom_factor)


# ================= 主窗口 =================
class ImageProcessorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.process = None
        self.thread_pool = QThreadPool()

        # --- 路径定义 ---
        # 1. 结果输出文件夹
        self.output_folder = os.path.join(os.getcwd(), "Autocorr", "AutoAcImage")
        # 2. 原始图像文件夹
        self.input_folder = os.path.join(os.getcwd(), "Autocorr", "GFimage_hirac")

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("图像处理工具 - 完整版")
        self.setGeometry(100, 100, 1400, 950)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- 1. 控制面板 ---
        control_group = QGroupBox("控制面板")
        control_layout = QHBoxLayout()

        # 按钮1: 开始处理
        self.start_button = QPushButton("开始处理图像")
        self.start_button.setMinimumHeight(45)
        self.start_button.setStyleSheet("""
            QPushButton { background-color: #4CAF50; color: white; font-weight: bold; border-radius: 4px; }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        self.start_button.clicked.connect(self.start_processing)

        # 按钮2: 显示原始图像
        self.btn_show_original = QPushButton("显示原始图像")
        self.btn_show_original.setMinimumHeight(45)
        self.btn_show_original.setStyleSheet("""
            QPushButton { background-color: #FF9800; color: white; font-weight: bold; border-radius: 4px; }
            QPushButton:hover { background-color: #e68900; }
        """)
        self.btn_show_original.clicked.connect(self.show_original_image)

        # 按钮3: 显示处理结果
        self.btn_show_result = QPushButton("显示处理结果")
        self.btn_show_result.setMinimumHeight(45)
        self.btn_show_result.setStyleSheet("""
            QPushButton { background-color: #2196F3; color: white; font-weight: bold; border-radius: 4px; }
            QPushButton:hover { background-color: #0b7dda; }
        """)
        self.btn_show_result.clicked.connect(self.refresh_result_images)

        # 按钮4: 复位视图
        reset_btn = QPushButton("复位视图")
        reset_btn.setMinimumHeight(45)
        reset_btn.setStyleSheet("QPushButton { background-color: #607D8B; color: white; border-radius: 4px; }")
        reset_btn.clicked.connect(self.reset_view)

        control_layout.addWidget(self.start_button, 2)
        control_layout.addWidget(self.btn_show_original, 1)  # 橙色
        control_layout.addWidget(self.btn_show_result, 1)  # 蓝色
        control_layout.addWidget(reset_btn, 1)
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)

        self.status_label = QLabel(f"就绪")
        main_layout.addWidget(self.status_label)

        # --- 2. 图像显示区 ---
        splitter = QSplitter(Qt.Vertical)
        image_group = QGroupBox("图像预览 (滚轮缩放，左键拖拽)")
        image_layout = QVBoxLayout()
        self.viewer = ImageViewer()
        image_layout.addWidget(self.viewer)
        image_group.setLayout(image_layout)
        splitter.addWidget(image_group)

        # --- 3. 日志区 ---
        log_group = QGroupBox("处理日志")
        log_layout = QVBoxLayout()
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        # 黑色背景，浅灰文字
        self.log_output.setStyleSheet("QTextEdit { font-family: Consolas; background-color: #1e1e1e; color: #d4d4d4; }")
        log_layout.addWidget(self.log_output)
        log_group.setLayout(log_layout)
        splitter.addWidget(log_group)

        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

        # 启动时若有结果则自动加载
        if os.path.exists(self.output_folder):
            QTimer.singleShot(500, self.refresh_result_images)

    # ================= 业务逻辑：外部 EXE 调用 =================
    def start_processing(self):
        self.log_output.clear()
        self.start_button.setEnabled(False)

        exe_path = self.find_exe_file()
        if not exe_path:
            QMessageBox.critical(self, "错误", "找不到 Auto_getAD_AtmCor.exe")
            self.start_button.setEnabled(True)
            return

        self.process = QProcess(self)
        self.process.setWorkingDirectory(os.path.dirname(exe_path))
        self.process.readyReadStandardOutput.connect(self.on_stdout)
        self.process.readyReadStandardError.connect(self.on_stderr)
        self.process.finished.connect(self.on_process_finished)
        self.process.start(exe_path, [])

    def find_exe_file(self):
        exe = "Auto_getAD_AtmCor.exe"
        for p in [os.getcwd(), os.path.dirname(os.path.abspath(__file__))]:
            path = os.path.join(p, exe)
            if os.path.exists(path): return path
        return None

    # --- 乱码清洗辅助函数 ---
    def clean_log_text(self, text):
        # 正则表达式：移除 ANSI 转义序列 (如颜色代码 [0m)
        ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
        cleaned_text = ansi_escape.sub('', text)
        return cleaned_text

    def on_stdout(self):
        data = self.process.readAllStandardOutput()
        text = bytes(data).decode("gbk", errors="ignore")

        # 清洗
        clean_text = self.clean_log_text(text)
        if clean_text.strip():
            self.log_output.append(clean_text)

    def on_stderr(self):
        data = self.process.readAllStandardError()
        text = bytes(data).decode("gbk", errors="ignore")

        clean_text = self.clean_log_text(text)
        if clean_text.strip():
            self.log_output.append(f"<span style='color:orange'>{clean_text}</span>")

    def on_process_finished(self):
        self.start_button.setEnabled(True)
        self.log_output.append("<b>处理完成，加载最新结果...</b>")
        self.refresh_result_images()

    def reset_view(self):
        if self.viewer.pixmap_item:
            self.viewer.fitInView(self.viewer.pixmap_item, Qt.KeepAspectRatio)

    # ================= 图像加载逻辑 =================
    # 功能 1：加载原始图像
    def show_original_image(self):
        self.load_image_from_folder(self.input_folder, "原始图像")

    # 功能 2：加载处理结果
    def refresh_result_images(self):
        self.load_image_from_folder(self.output_folder, "处理结果")

    # 核心加载函数
    def load_image_from_folder(self, folder_path, label_prefix):
        self.thread_pool.clear()

        if not os.path.exists(folder_path):
            self.log_output.append(f"<span style='color:red'>文件夹不存在: {folder_path}</span>")
            return

        # 查找最新的 TIFF
        files = [f for f in os.listdir(folder_path) if f.lower().endswith((".tif", ".tiff"))]
        full_paths = [os.path.join(folder_path, f) for f in files]
        full_paths.sort(key=os.path.getmtime, reverse=True)

        if not full_paths:
            self.log_output.append(f"在 {folder_path} 中未找到 TIFF 影像")
            return

        target_image = full_paths[0]

        # 更新界面提示
        msg = f"当前显示 [{label_prefix}]: {os.path.basename(target_image)}"
        self.status_label.setText(msg)
        self.log_output.append(msg)

        # 启动后台加载
        loader = ImageLoader(target_image)
        loader.signals.finished.connect(self.on_image_loaded)
        loader.signals.error.connect(self.on_image_error)
        self.thread_pool.start(loader)

    def on_image_loaded(self, path, q_image):
        pixmap = QPixmap.fromImage(q_image)
        self.viewer.set_image(pixmap)

    def on_image_error(self, path, error_msg):
        self.log_output.append(f"<span style='color:red'>加载失败: {error_msg}</span>")


def main():
    app = QApplication(sys.argv)
    window = ImageProcessorGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
