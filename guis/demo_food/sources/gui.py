import os

import pandas as pd
from pathlib import Path
from PyQt5 import QtWidgets
from natsort import natsorted
from PyQt5.QtCore import pyqtProperty, QSize
from PyQt5.QtGui import QIcon, QColor
from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QListWidget, QPushButton, QVBoxLayout, QLabel, \
    QFileDialog

from guis.demo_food.sources.others import QtImageViewer, QRoundProgressBar
from toolbox.classification.common import IO


class QDemo(QMainWindow):

    def __init__(self):
        super(QDemo, self).__init__()
        self.__init_gui()

    def __init_gui(self):
        # Parent part of windows
        process_widget = QWidget()
        process_layout = QHBoxLayout(process_widget)

        # Build patients list
        patients_widget = QWidget()
        patients_layout = QVBoxLayout(patients_widget)
        label = QLabel('Patients')
        patients_layout.addWidget(label)
        self.patients = QListWidget()
        self.patients.currentTextChanged.connect(self.set_current_parent)
        patients_layout.addWidget(self.patients)
        process_layout.addWidget(patients_widget)

        # Build images list
        images_widget = QWidget()
        images_layout = QVBoxLayout(images_widget)
        label = QLabel('Images')
        images_layout.addWidget(label)
        self.images = QListWidget()
        self.images.currentTextChanged.connect(self.set_current_image)
        images_layout.addWidget(self.images)
        process_layout.addWidget(images_widget)

        # Build compute
        icon_folder = Path(__file__).parent.parent / 'images'
        self.compute = QPushButton()
        self.compute.setFlat(True)
        self.compute.setIcon(QIcon(str(icon_folder / 'process.svg')))
        self.compute.setMinimumSize(QSize(60, 60))
        self.compute.pressed.connect(self.process)
        process_layout.addWidget(self.compute)

        # Build scorer
        self.score = QRoundProgressBar()
        # from PyQt5 import Qt, QtCore, QtGui, QtWidgets
        # self.dark_palette = QtGui.QPalette()
        # self.dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
        # self.dark_palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
        # self.dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
        # self.dark_palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
        # self.dark_palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
        # self.dark_palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
        # self.dark_palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
        # # self.dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
        # # self.dark_palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
        # # self.dark_palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
        # # self.dark_palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
        # # self.dark_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
        # # self.dark_palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)

        # self.score.setPalette(DarkPalette.color_palette())
        self.score.setFixedSize(300, 300)
        self.score.setDataPenWidth(3)
        self.score.setOutlinePenWidth(3)
        self.score.setDonutThicknessRatio(0.85)
        self.score.setDecimals(1)
        self.score.setFormat(' %p %')
        self.score.setNullPosition(90)
        self.score.setBarStyle(QRoundProgressBar.StyleDonut)
        self.score.setDataColors([(0., QColor.fromRgb(0, 0, 0)), (1, QColor.fromRgb(40, 255, 220))])
        self.score.setRange(0, 1)
        self.score.setValue(0.85)

        process_layout.addWidget(self.score)

        # Build viewer
        self.viewer = QtImageViewer()


        # Build parent
        parent_widget = QWidget()
        parent_layout = QVBoxLayout(parent_widget)
        parent_layout.addWidget(process_widget)
        parent_layout.addWidget(self.viewer)

        self.setCentralWidget(parent_widget)

    def load_data(self):
        data_file, filter = QFileDialog.getOpenFileName(self, 'Choose file', str(Path().home()), 'Pickle files (*.pickle)')
        if data_file:
            self.data = IO.load(data_file)
        self.patients.addItems(self.data['ID'].unique())

    def process(self):
        print('lol')

    def set_current_parent(self, parent):
        self.images.addItems(self.data[self.data['ID'] == parent]['Datum'].unique())

    def set_current_image(self, path):
        self.viewer.loadImageFromFile(str(Path().home()/path.replace('/home/rcendre/', '')))


#
# class CircularProgressBar(QWidget):
#
#     CircularProgressBar(QWidget * p = 0) : QWidget(p), p(0) {
#       setMinimumSize(208, 208);
#     }
#
#     void upd(qreal pp) {
#       if (p == pp) return;
#       p = pp;
#       update();
#     }
#   void paintEvent(QPaintEvent *) {
#     qreal pd = p * 360;
#     qreal rd = 360 - pd;
#     QPainter p(this);
#     p.fillRect(rect(), Qt::white);
#     p.translate(4, 4);
#     p.setRenderHint(QPainter::Antialiasing);
#     QPainterPath path, path2;
#     path.moveTo(100, 0);
#     path.arcTo(QRectF(0, 0, 200, 200), 90, -pd);
#     QPen pen, pen2;
#     pen.setCapStyle(Qt::FlatCap);
#     pen.setColor(QColor("#30b7e0"));
#     pen.setWidth(8);
#     p.strokePath(path, pen);
#     path2.moveTo(100, 0);
#     pen2.setWidth(8);
#     pen2.setColor(QColor("#d7d7d7"));
#     pen2.setCapStyle(Qt::FlatCap);
#     pen2.setDashPattern(QVector<qreal>{0.5, 1.105});
#     path2.arcTo(QRectF(0, 0, 200, 200), 90, rd);
#     pen2.setDashOffset(2.2);
#     p.strokePath(path2, pen2);
#   }