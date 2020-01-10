import os

import numpy
import pandas as pd
from pathlib import Path
from PyQt5 import Qt, QtCore, QtGui, QtWidgets
from natsort import natsorted
from PyQt5.QtCore import pyqtProperty, QSize, QCoreApplication
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

        # Build scorer
        self.score = QRoundProgressBar()
        dark_palette = QtGui.QPalette()
        dark_palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(25, 35, 45))
        dark_palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
        self.score.setPalette(dark_palette)
        self.score.setFixedSize(300, 300)
        self.score.setDataPenWidth(0)
        self.score.setOutlinePenWidth(0)
        self.score.setDonutThicknessRatio(0.85)
        self.score.setDecimals(1)
        self.score.setFormat(' %p %')
        self.score.setNullPosition(90)
        self.score.setBarStyle(QRoundProgressBar.StyleDonut)
        self.score.setDataColors([(0., QColor.fromRgb(0, 0, 0)), (1, QColor.fromRgb(40, 255, 220))])
        self.score.setRange(0, 1)
        self.score.setValue(0)
        # self.score.mouseReleaseEvent = self.process
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
            self.patients.setCurrentRow(0)

    def set_current_parent(self, parent):
        # Set images
        self.images.clear()
        self.images.addItems(self.data[self.data['ID'] == parent]['Datum'].unique())

        # Set values
        current = self.data[self.data['ID'] == self.patients.currentItem().text()]
        single_record = current[current['ImageID'] == '0M']
        fold = single_record['Fold'].values[0]
        self.score.setValue(single_record[f'D_ALO_Probability_{fold}'].values[0][1])
        for index, row in enumerate(current.iterrows()):
            # Prediction
            image_prediction = row[1]['Supervised_Prediction']
            if isinstance(image_prediction, numpy.ndarray):
                image_prediction = image_prediction[0]
            # Truth
            image_truth = row[1]['MalignantEncode']
            # Color!
            if image_prediction == image_truth == 1:
                self.images.item(index).setBackground(QColor.fromRgb(10, 60, 50))
            if image_prediction != image_truth == 0:
                self.images.item(index).setBackground(QColor.fromRgb(200, 100, 0))
            if image_prediction != image_truth == 1:
                self.images.item(index).setBackground(QColor.fromRgb(200, 100, 0))

    def set_current_image(self, path):
        self.viewer.loadImageFromFile(str(Path().home()/path.replace('/home/rcendre/', '')))
