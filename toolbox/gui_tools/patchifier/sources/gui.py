import glob
import os
import pandas as pd
from pathlib import Path

from PyQt5 import QtWidgets
from natsort import natsorted
from os.path import join, isfile, abspath, exists
from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QRect, QPropertyAnimation, pyqtProperty
from PyQt5.QtGui import QImage, QPixmap, QPainterPath, QColor, QFont, QPen
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGridLayout, QMainWindow, QHBoxLayout, QWidget, QLabel, \
    QGraphicsTextItem, QPushButton, QSpinBox, QGraphicsRectItem, QProgressBar, QVBoxLayout, \
    QTableWidget, QTabWidget, QTableWidgetItem, QAbstractItemView, QGraphicsItemGroup
from toolbox.IO import dermatology


class QPatchExtractor(QMainWindow):
    # Modes
    LABEL = 0
    PATCH = 1

    def __init__(self, input_folder, pathologies, settings, output=''):
        super(QPatchExtractor, self).__init__()
        self.patient_index = 0
        self.image_index = 0
        self.patch_index = 0
        self.dataframe = None
        self.patients_directories = [str(path) for path in input_folder.iterdir()]
        self.patients_directories = natsorted(self.patients_directories)
        self.patients_directories = [Path(path) for path in self.patients_directories]
        self.pathologies = pathologies
        self.settings = settings
        self.__init_gui()
        # Init the state
        self.change_patient(0)

    def __init_gui(self):
        # Parent part of windows
        parent_widget = QWidget()
        parent_layout = QVBoxLayout(parent_widget)
        # Build patient progress bar
        self.patient_bar = QProgressBar()
        self.patient_bar.setAlignment(Qt.AlignCenter)
        self.patient_bar.setStyleSheet('QProgressBar {border: 2px solid grey;border-radius: 5px;}'
                                       'QProgressBar::chunk {background-color: #bff88b;width: 20px;}')
        self.patient_bar.setRange(0, len(self.patients_directories))
        self.patient_bar.setTextVisible(True)
        parent_layout.addWidget(self.patient_bar)
        # Build image progress bar
        self.image_bar = QProgressBar()
        self.image_bar.setAlignment(Qt.AlignCenter)
        self.image_bar.setStyleSheet('QProgressBar {border: 2px solid grey;border-radius: 5px;}'
                                     'QProgressBar::chunk {background-color: #05B8CC;width: 20px;}')
        self.image_bar.setTextVisible(True)
        parent_layout.addWidget(self.image_bar)
        # Build image viewer
        self.viewer = QtImageViewer()
        self.viewer.grabKeyboard()
        self.viewer.keyPressed.connect(self.key_pressed)
        self.viewer.leftMouseButtonPressed.connect(self.click_event)
        self.viewer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # Build annotate component
        self.annotate_widget = QTabWidget()
        self.label_widget = QLabelWidget(self.annotate_widget, self.pathologies, self.settings)
        self.label_widget.change_label.connect(self.change_image_label)
        self.patch_widget = QPatchWidget(self.annotate_widget, self.pathologies, self.settings)
        self.patch_widget.changed_patch_selection.connect(self.change_patch)
        self.patch_widget.changed_patch_size.connect(self.viewer.setRectangleSize)
        self.patch_widget.set_value(250)
        self.annotate_widget.currentChanged.connect(self.change_mode)
        self.annotate_widget.addTab(self.label_widget, 'Labels')
        self.annotate_widget.addTab(self.patch_widget, 'Patchs')
        self.annotate_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        # Build final layout
        global_widget = QWidget()
        global_layout = QGridLayout(global_widget)
        global_layout.addWidget(parent_widget, 0, 0)
        global_layout.addWidget(self.viewer, 1, 0)
        global_layout.addWidget(self.annotate_widget, 2, 0)
        self.setCentralWidget(global_widget)

    def change_image(self, move):
        dataframe = self.get_dataframe('Full')
        length = len(dataframe)
        if length == 0:
            self.image_index = 0
        else:
            self.image_index = (self.image_index + move) % length
        # Send data to components
        self.label_widget.send_image(self.get_image_data())
        self.update_image()

    def change_image_label(self, label):
        current = self.get_image_data()
        if not current['Label'] == label:
            self.dataframe.loc[current.name, 'Label'] = label
            dataframe = self.get_dataframe('Full')
            # Send data to components
            self.label_widget.send_patient(dataframe)

    def change_mode(self):
        mode = self.get_mode()
        # Actions depend on mode
        if mode == QPatchExtractor.LABEL:
            self.viewer.change_selection_state(False)
        else:
            self.viewer.change_selection_state(True)

    def change_patch(self, index):
        self.patch_index = index
        self.viewer.set_current_patch(index)

    def change_patient(self, move):
        # Start by closing previous df
        self.close_dataframe()
        # Change patient
        length = len(self.patients_directories)
        self.patient_index = (self.patient_index + move) % length
        # Open new df
        self.open_dataframe()
        self.update_patient()
        # Update image
        dataframe = self.get_dataframe('Full')
        if dataframe is None:
            return
        self.image_bar.setRange(0, len(dataframe)-1)
        # Send data to components
        self.label_widget.send_patient(dataframe)
        self.patch_widget.send_patches(self.get_dataframe('Patch'))
        self.viewer.set_patches(self.get_patches())
        self.reset_image()

    def click_event(self, x, y):
        mode = self.get_mode()
        # Actions depend on mode
        if mode == QPatchExtractor.LABEL:
            return
        # Acquire image and it to dataframe
        # self.viewer.mouseRectColorTransition(QColor(Qt.red))
        # self.write_patch(x, y)

    def close_dataframe(self):
        # TODO
        file = None  # self.get_dataframe()
        if not file:
            return

        if self.dataframe is None:
            return

        folder = self.get_current_folder()
        if not exists(folder):
            os.makedirs(folder)

        self.dataframe.to_csv(file, index=False)

    def closeEvent(self, event):
        self.close_dataframe()
        super().closeEvent(event)

    def extract_patch(self, x, y):
        # Load image
        raw_image = self.viewer.image()
        image_rect = raw_image.rect()

        # Compute patch position
        size = self.out_size.value()
        patch_rect = QRect(x - size / 2, y - size / 2, size, size)

        # Test if patch rectangle is full
        if not image_rect.intersected(patch_rect) == patch_rect:
            return None

        # Extract patch
        return raw_image.copy(patch_rect)

    def get_dataframe(self, type_filter=None):
        if type_filter is None:
            return self.dataframe
        return self.dataframe[self.dataframe['Type'] == type_filter]

    def get_image(self, absolute=True):
        dataframe = self.get_dataframe('Full')
        if dataframe is None or len(dataframe) == 0:
            return None

        if self.image_index >= len(dataframe):
            self.image_index = 0

        if absolute:
            return dataframe['Full_Path'].iloc[self.image_index]
        else:
            return dataframe['Path'].iloc[self.image_index]

    def get_image_data(self):
        dataframe = self.get_dataframe('Full')
        if self.image_index < len(dataframe):
            return dataframe.iloc[self.image_index]
        return None

    def get_mode(self):
        return self.annotate_widget.currentIndex()

    def get_patches(self):
        dataframe = self.get_dataframe('Patch')
        patches = []
        for index, row in dataframe.iterrows():
            patches.append((int(row['Center_X']), int(row['Center_Y']), int(row['Width'])))
        return patches

    def get_patient(self):
        return self.get_patient_folder().name

    def get_patient_binary(self):
        if self.dataframe is None or len(self.dataframe) == 0:
            return None

        return self.dataframe.iloc[0].loc['Binary_Diagnosis']

    def get_patient_folder(self):
        return self.patients_directories[self.patient_index]

    def get_patient_pathology(self):
        if self.dataframe is None or len(self.dataframe) == 0:
            return None

        return self.dataframe.iloc[0].loc['Diagnosis']

    def key_pressed(self, key):
        # Action that move image of a patient
        if key == Qt.Key_Left:
            self.change_image(-1)
        elif key == Qt.Key_Right:
            self.change_image(1)
        # Action that move patient
        elif key == Qt.Key_Up:
            self.change_patient(1)
        elif key == Qt.Key_Down:
            self.change_patient(-1)
        elif Qt.Key_0 <= key <= Qt.Key_9 or Qt.Key_Delete:
            self.tool_controls(key)

    def open_dataframe(self):
        self.dataframe = dermatology.Reader.scan_subfolder(self.get_patient_folder(), parameters={'patches': True,
                                                                                                  'modality': 'Microscopy'})

    def reset_image(self):
        self.image_index = 0
        # Send data to components
        self.label_widget.send_image(self.get_image_data())
        self.update_image()

    def show_patches(self):
        print('todo')

    def tool_controls(self, key):
        if self.get_mode() == QPatchExtractor.LABEL:
            self.label_widget.send_key(key - Qt.Key_0)
        else:
            if key == Qt.Key_Delete:
                self.patch_widget.send_remove()
            else:
                self.patch_widget.send_key(key - Qt.Key_0)

    def update_image(self):
        self.image_bar.setValue(self.image_index)
        self.image_bar.setFormat(self.get_image(absolute=False))
        self.viewer.loadImage(self.get_image())

    def update_patient(self):
        self.patient_bar.setValue(self.patient_index)
        self.patient_bar.setFormat(
            'Patient: {patient} - Pathology: {pathology}({binary})'.format(patient=self.get_patient(),
                                                                          pathology=self.get_patient_pathology(),
                                                                          binary=self.get_patient_binary()))

    def write_patch(self, x, y):
        patch = self.extract_patch(x, y)
        if not patch:
            self.viewer.mouseRectColorTransition(QColor(Qt.red))
            return

        filename = '{count}.bmp'.format(count=len(glob.glob(join(self.get_current_folder(), '*.bmp'))))
        patch_path = join(self.get_current_folder(), 'patches', filename)
        # Write image in dataframe
        self.dataframe = self.dataframe.append({'Modality': 'Microscopy',
                                                'Path': os.path.relpath(patch_path, self.get_current_folder()),
                                                'Height': patch.height(),
                                                'Width': patch.width(),
                                                'Center_X': int(x),
                                                'Center_Y': int(y),
                                                'Label': self.pathologies[self.pathology_index],
                                                'Source': self.get_current_image(full=False)}, ignore_index=True)

        patch_dir = abspath(join(patch_path, os.pardir))
        if not exists(patch_dir):
            os.makedirs(patch_dir)
        patch.save(patch_path, format='bmp')
        self.viewer.mouseRectColorTransition(QColor(Qt.green))


class QLabelWidget(QWidget):
    # Signals
    change_label = pyqtSignal(str)

    def __init__(self, parent, pathologies, settings):
        super(QLabelWidget, self).__init__(parent)
        self.pathologies = pathologies
        self.settings = settings
        self.default_value = len(pathologies)-1
        self.init_gui()

    def change_mode(self, current, previous):
        current_label = self.pathologies[current.row()]
        self.update_color(current_label)
        self.change_label.emit(current_label)

    def init_gui(self):
        # Then build annotation tool
        self.table = QTableWidget()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setRowCount(len(self.pathologies))
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(('Keyboard Shortcut', 'Name', 'Total'))
        # Keyboard + Label
        for index, label in enumerate(self.pathologies):
            self.table.setItem(index, 0, QTableWidgetItem('Numpad {index}'.format(index=index)))
            self.table.setItem(index, 1, QTableWidgetItem('{label}'.format(label=label)))
        self.table.resizeColumnsToContents()
        hheader = self.table.horizontalHeader()
        hheader.setStretchLastSection(True)
        vheader = self.table.verticalHeader()
        vheader.hide()
        vheader.setStretchLastSection(True)
        # Connect to changes
        self.table.selectionModel().currentRowChanged.connect(self.change_mode)

        patch_layout = QVBoxLayout(self)
        patch_layout.addWidget(self.table)

    def send_image(self, data):
        if data is None:
            return
        current = data['Label']
        try:
            index = self.pathologies.index(current)
        except:
            index = self.default_value
        self.table.selectRow(index)

    def send_patient(self, data):
        values = data['Label'].value_counts()
        for index, label in enumerate(self.pathologies):
            self.table.setItem(index, 2, QTableWidgetItem('{count}'.format(count=values.get(label, 0))))

    def send_key(self, key):
        self.table.selectRow(key)

    def update_color(self, current_label):
        color_tuple = self.settings.get_color(current_label)
        qcolor = QColor.fromRgbF(color_tuple[0], color_tuple[1], color_tuple[2], 0.75)
        self.table.setStyleSheet('QTableView{selection-background-color: '+qcolor.name()+'}')


class QPatchWidget(QWidget):
    # Signals
    changed_patch_size = pyqtSignal(int)
    changed_patch_selection = pyqtSignal(int)

    def __init__(self, parent, pathologies, settings):
        super(QPatchWidget, self).__init__(parent)
        self.pathologies = pathologies
        self.settings = settings
        self.mode = 0
        self.init_gui()

    def init_gui(self):
        # Create table that list current patches
        self.table = QTableWidget()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(('Center X', 'Center Y', 'Label'))
        self.table.resizeColumnsToContents()
        hheader = self.table.horizontalHeader()
        hheader.setStretchLastSection(True)
        self.table.selectionModel().currentRowChanged.connect(self.row_changed)
        # Manage patch size
        self.size = QSpinBox()
        self.size.valueChanged.connect(self.changed_patch_size.emit)
        self.size.setRange(0, 1000)
        self.size.setSingleStep(10)
        self.size.setSuffix("px")
        # Then build patch tool
        patch_layout = QGridLayout(self)
        patch_layout.addWidget(self.table, 0, 0, 1, 2)
        patch_layout.addWidget(QLabel('Keyboard Shortcuts: 0: Healthy / 1: Benign / 1:Malignant'), 1, 0, 1, 2)
        patch_layout.addWidget(QLabel('Patch Width/Height'), 2, 0)
        patch_layout.addWidget(self.size, 2, 1)

    def row_changed(self, current, previous):
        self.changed_patch_selection.emit(current.row())

    def send_key(self, key):
        if key < len(self.pathologies):
            self.mode = key

    def send_patches(self, data):
        self.table.setRowCount(0)
        if len(data) == 0:
            return
        self.table.setRowCount(len(data))
        for index, row in data.iterrows():
            self.table.setItem(index, 0, QTableWidgetItem('{center}'.format(center=row['Center_X'])))
            self.table.setItem(index, 1, QTableWidgetItem('{center}'.format(center=row['Center_Y'])))
            self.table.setItem(index, 2, QTableWidgetItem('{label}'.format(label=row['Label'])))

    def set_value(self, size):
        self.size.setValue(size)


class QtImageViewer(QGraphicsView):
    # Viewer signals
    leftMouseButtonPressed = pyqtSignal(float, float)
    rightMouseButtonPressed = pyqtSignal(float, float)
    leftMouseButtonReleased = pyqtSignal(float, float)
    rightMouseButtonReleased = pyqtSignal(float, float)
    leftMouseButtonDoubleClicked = pyqtSignal(float, float)
    rightMouseButtonDoubleClicked = pyqtSignal(float, float)
    keyPressed = pyqtSignal(int)

    def __init__(self):
        QGraphicsView.__init__(self)
        # Image is displayed as a QPixmap in a QGraphicsScene attached to this QGraphicsView.
        self.scene = QGraphicsScene()
        self.patch_color = QColor(Qt.white)
        self.mouse_color = QColor(Qt.blue)
        self.mouse_rect = QGraphicsRectItem(-25, -25, 50, 50)
        self.mouse_rect.setPen(QPen(self.mouse_color, 6, Qt.DotLine))
        self.patches = QGraphicsItemGroup()
        self.scene.addItem(self.patches)
        self.scene.addItem(self.mouse_rect)
        self.setScene(self.scene)
        # Store a local handle to the scene's current image pixmap.
        self._pixmapHandle = None
        # Image aspect ratio mode.
        self.aspectRatioMode = Qt.KeepAspectRatio
        # Scroll bar behaviour.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # Stack of QRectF zoom boxes in scene coordinates.
        self.zoomStack = []
        # Flags for enabling/disabling mouse interaction.
        self.selection_state = True

    def change_mouse_default_color(self, color):
        self.mouse_color = QColor.fromRgbF(color[0], color[1], color[2])
        self.mouse_rect.setPen(QPen(self.mouse_color, 6, Qt.DotLine))

    def change_selection_state(self, enable=True):
        self.selection_state = enable
        if self.selection_state:
            self.mouse_rect.show()
            self.patches.show()
        else:
            self.mouse_rect.hide()
            self.patches.hide()

    def clearImage(self):
        """ Removes the current image pixmap from the scene if it exists.
        """
        if self.hasImage():
            self.scene.removeItem(self._pixmapHandle)
            self._pixmapHandle = None

    def keyPressEvent(self, event):
        self.keyPressed.emit(event.key())

    def hasImage(self):
        """ Returns whether or not the scene contains an image pixmap.
        """
        return self._pixmapHandle is not None

    def image(self):
        """ Returns the scene's current image pixmap as a QImage, or else None if no image exists.
        :rtype: QImage | None
        """
        if self.hasImage():
            return self._pixmapHandle.pixmap().toImage()
        return None

    def mouseRectColorTransition(self, color):
        if self.selection_state:
            self.animation = QPropertyAnimation(self, b'pcolor')
            self.animation.setDuration(1000)
            self.animation.setStartValue(color)
            self.animation.setEndValue(self.mouse_color)
            self.animation.start()

    def loadImage(self, path):
        """ Load an image from file.
        Without any arguments, loadImageFromFile() will popup a file dialog to choose the image file.
        With a fileName argument, loadImageFromFile(fileName) will attempt to load the specified image file directly.
        """
        if path is None or len(path) == 0 or not isfile(path):
            self.setImage(QImage())
        else:
            image = QImage(path)
            self.setImage(image)

    def pixmap(self):
        """ Returns the scene's current image pixmap as a QPixmap, or else None if no image exists.
        :rtype: QPixmap | None
        """
        if self.hasImage():
            return self._pixmapHandle.pixmap()
        return None

    def setImage(self, image):
        """ Set the scene's current image pixmap to the input QImage or QPixmap.
        Raises a RuntimeError if the input image has type other than QImage or QPixmap.
        :type image: QImage | QPixmap
        """
        if type(image) is QPixmap:
            pixmap = image
        elif type(image) is QImage:
            pixmap = QPixmap.fromImage(image)
        else:
            raise RuntimeError("ImageViewer.setImage: Argument must be a QImage or QPixmap.")
        if self.hasImage():
            self._pixmapHandle.setPixmap(pixmap)
        else:
            self._pixmapHandle = self.scene.addPixmap(pixmap)
        self._pixmapHandle.setZValue(-1)
        self.setSceneRect(QRectF(pixmap.rect()))  # Set scene size to image size.
        self.updateViewer()

    def set_current_patch(self, index):
        # Default color items
        for patch in self.patches:
            patch.setPen(QPen(self.patch_color, 4, Qt.SolidLine))
        # Set selection
        self.patches[index].setPen(QPen(self.patch_color, 4, Qt.SolidLine))

    def set_patches(self, patches):
        # Delete patches items
        for patch in self.patches.childItems():
            self.patches.removeFromGroup(patch)
        # Create new items
        for patch in patches:
            patch_item = QGraphicsRectItem(patch[0]-patch[2], patch[1]-patch[2], patch[2], patch[2])
            patch_item.setPen(QPen(self.patch_color, 4, Qt.SolidLine))
            self.patches.addToGroup(patch_item)

    def updateViewer(self):
        """ Show current zoom (if showing entire image, apply current aspect ratio mode).
        """
        if not self.hasImage():
            return
        if len(self.zoomStack) and self.sceneRect().contains(self.zoomStack[-1]):
            self.fitInView(self.zoomStack[-1], Qt.IgnoreAspectRatio)  # Show zoomed rect (ignore aspect ratio).
        else:
            self.zoomStack = []  # Clear the zoom stack (in case we got here because of an invalid zoom).
            self.fitInView(self.sceneRect(), self.aspectRatioMode)  # Show entire image (use current aspect ratio mode).

    def resizeEvent(self, event):
        """ Maintain current zoom on resize.
        """
        self.updateViewer()

    def setRectangleSize(self, size):
        self.mouse_rect.setRect(-(size / 2), -(size / 2), size, size)

    def mouseMoveEvent(self, event):
        self.mouse_rect.setPos(self.mapToScene(event.pos()))

    def mousePressEvent(self, event):
        """ Start mouse pan or zoom mode.
        """
        scenePos = self.mapToScene(event.pos())
        if event.button() == Qt.LeftButton:
            self.leftMouseButtonPressed.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.RightButton:
            self.rightMouseButtonPressed.emit(scenePos.x(), scenePos.y())
        QGraphicsView.mousePressEvent(self, event)

    def mouseReleaseEvent(self, event):
        """ Stop mouse pan or zoom mode (apply zoom if valid).
        """
        QGraphicsView.mouseReleaseEvent(self, event)
        scenePos = self.mapToScene(event.pos())
        if event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.NoDrag)
            self.leftMouseButtonReleased.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.RightButton:
            self.setDragMode(QGraphicsView.NoDrag)
            self.rightMouseButtonReleased.emit(scenePos.x(), scenePos.y())

    def mouseDoubleClickEvent(self, event):
        """ Show entire image.
        """
        scenePos = self.mapToScene(event.pos())
        if event.button() == Qt.LeftButton:
            self.leftMouseButtonDoubleClicked.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.RightButton:
            self.rightMouseButtonDoubleClicked.emit(scenePos.x(), scenePos.y())
        QGraphicsView.mouseDoubleClickEvent(self, event)

    def _set_pcolor(self, color):
        pen = self.mouse_rect.pen()
        pen.setColor(color)
        self.mouse_rect.setPen(pen)

    pcolor = pyqtProperty(QColor, fset=_set_pcolor)
