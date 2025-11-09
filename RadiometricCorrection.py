# -*- coding: utf-8 -*-
# @Time    : 2025/11/9 20:20
# @Author  : Liu Kun
# @Email   : liukunjsj@163.com
# @File    : RadiometricCorrection.py
# @Software: PyCharm

"""
Describe:
"""
# -*- coding: utf-8 -*-
import sys, os
from PyQt_Form import CalibrationApp
from PyQt5.QtWidgets import QApplication
def get_base_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

base_dir = get_base_dir()
os.chdir(base_dir)  # 切换到程序目录，保证资源相对路径正确


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CalibrationApp()
    window.show()
    sys.exit(app.exec_())
