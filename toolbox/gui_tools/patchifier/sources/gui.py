import glob
import os
import pandas as pd
from os.path import join, isfile, abspath, exists
from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QRect, QPropertyAnimation, pyqtProperty
from PyQt5.QtGui import QImage, QPixmap, QPainterPath, QColor, QFont, QPen
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGridLayout, QMainWindow, QHBoxLayout, QWidget, QLabel, \
    QGraphicsTextItem, QFileDialog, QPushButton, QDialog, QSpinBox, QGraphicsRectItem, QProgressBar, QVBoxLayout


class QPatchExtractor(QMainWindow):

    def __init__(self, inputs, pathologies, settings, output='', editable=True):
        super(QPatchExtractor, self).__init__()
        self.editable = editable
        self.patient_index = 0
        self.image_index = 0
        self.pathology_index = 0
        self.data = inputs.data.groupby('ID')
        self.patients = list(inputs.data['ID'].unique())
        self.pathologies = pathologies
        self.settings = settings
        self.output = output
        self.dataframe = None
        # Design GUI
        self.define_layer()
        self.update_directory()
        self.update_image()
        self.update_output()
        self.change_label(0)
        self.image_bar.setRange(0, len(self.data.get_group(self.get_current_patient())))
        # Open first df
        self.open_dataframe()

    def change_output(self):
        dialog = QFileDialog(self, 'Output folder')
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        dialog.setOption(QFileDialog.ShowDirsOnly)
        if dialog.exec_() == QDialog.Accepted:
            self.output = dialog.selectedFiles()[0]
            self.update_output()

    def change_label(self, index):
        self.pathology_index = index%len(self.pathologies)
        pathology = self.pathologies[self.pathology_index]
        self.setWindowTitle(pathology)

    def change_image(self, move):
        length = len(self.data.get_group(self.get_current_patient()))
        self.image_index = (self.image_index + move) % length
        self.update_image()

    def change_patient(self, move):
        # Start by closing previous df
        self.close_dataframe()

        # Change patient
        length = len(self.patients)
        self.patient_index = (self.patient_index + move) % length
        self.image_bar.setRange(0, len(self.data.get_group(self.get_current_patient())))
        self.update_directory()
        self.reset_image()

        # Open new df
        self.open_dataframe()

    def click_event(self, x, y):
        if not self.output:
            self.viewer.mouseRectColorTransition(QColor(Qt.red))
            return
        # Acquire image and it to dataframe
        self.write_patch(x, y)

    def close_dataframe(self):
        file = self.get_current_dataframe()
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

    def define_layer(self):
        # Parent part of windows
        # Build export button
        parent_widget = QWidget()
        parent_layout = QVBoxLayout(parent_widget)
        # Build patient preview
        self.patient_bar = QProgressBar()
        self.patient_bar.setAlignment(Qt.AlignCenter)
        self.patient_bar.setStyleSheet('QProgressBar {border: 2px solid grey;border-radius: 5px;}'
                                       'QProgressBar::chunk {background-color: #bff88b;width: 20px;}')
        self.patient_bar.setRange(0, len(self.patients))
        self.patient_bar.setTextVisible(True)
        parent_layout.addWidget(self.patient_bar)
        # Build image progress bar
        self.image_bar = QProgressBar()
        self.image_bar.setAlignment(Qt.AlignCenter)
        self.image_bar.setStyleSheet('QProgressBar {border: 2px solid grey;border-radius: 5px;}'
                                     'QProgressBar::chunk {background-color: #05B8CC;width: 20px;}')
        self.image_bar.setTextVisible(True)
        parent_layout.addWidget(self.image_bar)

        self.viewer = QtImageViewer()
        self.viewer.grabKeyboard()
        self.viewer.keyPressed.connect(self.key_pressed)
        self.viewer.leftMouseButtonPressed.connect(self.click_event)

        # Export part of windows
        # Build export button
        self.button_output = QPushButton('')
        self.button_output.setEnabled(self.editable)
        self.button_output.released.connect(self.change_output)

        # Build export size component
        self.out_size = QSpinBox()
        self.out_size.setEnabled(self.editable)
        self.out_size.valueChanged.connect(self.viewer.setRectangleSize)
        self.out_size.setRange(0, 1000)
        self.out_size.setSingleStep(10)
        self.out_size.setSuffix("px")
        self.out_size.setValue(250)

        # Then build layout
        export_widget = QWidget()
        export_layout = QGridLayout(export_widget)
        export_layout.addWidget(self.button_output, 0, 0, 1, 2)
        export_layout.addWidget(QLabel('Width/Height'), 1, 0)
        export_layout.addWidget(self.out_size, 1, 1)

        # Build final layout
        global_widget = QWidget()
        # global_widget.setFocusPolicy(Qt.NoFocus)
        global_layout = QGridLayout(global_widget)
        global_layout.addWidget(parent_widget, 0, 0)
        global_layout.addWidget(self.viewer, 1, 0)
        global_layout.addWidget(export_widget, 2, 0)

        self.setCentralWidget(global_widget)

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

    def get_current_folder(self):
        if self.output:
            return join(self.output, self.get_current_patient())
        return ''

    def get_current_dataframe(self):
        if self.output:
            return join(self.get_current_folder(), 'patches.csv')
        return ''

    def get_current_pathology(self):
        return self.data.get_group(self.get_current_patient()).iloc[0].loc['Diagnosis']

    def get_current_patient(self):
        return self.patients[self.patient_index]

    def get_current_image(self, full=True):
        patient = self.data.get_group(self.get_current_patient())
        if self.image_index >= len(patient):
            self.image_index = 0
        if full:
            return patient['Full_path'].iloc[self.image_index]
        else:
            return patient['Path'].iloc[self.image_index]

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
        elif Qt.Key_0 <= key <= Qt.Key_9:
            self.change_label(key-Qt.Key_0)

    def open_dataframe(self):
        file = self.get_current_dataframe()
        if not file:
            return

        if exists(file):
            self.dataframe = pd.read_csv(file)
        else:
            self.dataframe = pd.DataFrame(columns=['Modality', 'Path', 'Height', 'Width',
                                                   'Center_X', 'Center_Y', 'Label', 'Source'])

    def reset_image(self):
        self.image_index = 0
        self.update_image()

    def update_image(self):
        self.image_bar.setValue(self.image_index)
        self.image_bar.setFormat(self.get_current_image(full=False))
        self.viewer.loadImage(self.get_current_image())

    def update_directory(self):
        self.patient_bar.setValue(self.patient_index)
        self.patient_bar.setFormat('Patient: {patient} - Pathology: {pathology}'.format(patient=self.get_current_patient(),
                                                                         pathology=self.get_current_pathology()))

    def update_output(self):
        self.button_output.setText('Output : {path}'.format(path=self.output))

    def write_patch(self, x, y):
        patch = self.extract_patch(x, y)
        if not patch:
            self.viewer.mouseRectColorTransition(QColor(Qt.red))
            return

        filename = '{count}.bmp'.format(count=len(glob.glob(join(self.get_current_folder(), '*.bmp'))))
        patch_path = join(self.get_current_folder(), filename)
        self.dataframe = self.dataframe.append({'Modality': 'Microscopy',
                                                'Path': patch_path,
                                                'Height': patch.height(),
                                                'Width': patch.width(),
                                                'Center_X': int(x),
                                                'Center_Y': int(y),
                                                'Label': self.pathologies[self.pathology_index],
                                                'Source': self.get_current_image()}, ignore_index=True)


        patch_dir = abspath(join(patch_path, os.pardir))
        if not exists(patch_dir):
            os.makedirs(patch_dir)
        patch.save(patch_path, format='bmp')
        self.viewer.mouseRectColorTransition(QColor(Qt.green))


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
        self.text = QGraphicsTextItem()
        self.text.setFont(QFont('Arial', 20))
        self.text.setPos(0, 0)
        self.text.setDefaultTextColor(QColor(255, 0, 0))
        self.mouse_color = QColor(Qt.blue)
        self.mouse_rect = QGraphicsRectItem(-25, -25, 50, 50)
        self.mouse_rect.setPen(QPen(self.mouse_color, 6, Qt.DotLine))
        self.scene.addItem(self.text)
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
        self.canZoom = True
        self.canPan = True

    def change_mouse_default_color(self, color):
        self.mouse_color = QColor.fromRgbF(color[0], color[1], color[2])
        self.mouse_rect.setPen(QPen(self.mouse_color, 6, Qt.DotLine))

    def keyPressEvent(self, event):
        self.keyPressed.emit(event.key())

    def hasImage(self):
        """ Returns whether or not the scene contains an image pixmap.
        """
        return self._pixmapHandle is not None

    def clearImage(self):
        """ Removes the current image pixmap from the scene if it exists.
        """
        if self.hasImage():
            self.scene.removeItem(self._pixmapHandle)
            self._pixmapHandle = None

    def loadImage(self, path):
        """ Load an image from file.
        Without any arguments, loadImageFromFile() will popup a file dialog to choose the image file.
        With a fileName argument, loadImageFromFile(fileName) will attempt to load the specified image file directly.
        """
        if len(path) and isfile(path):
            image = QImage(path)
            self.setImage(image)

    def pixmap(self):
        """ Returns the scene's current image pixmap as a QPixmap, or else None if no image exists.
        :rtype: QPixmap | None
        """
        if self.hasImage():
            return self._pixmapHandle.pixmap()
        return None

    def image(self):
        """ Returns the scene's current image pixmap as a QImage, or else None if no image exists.
        :rtype: QImage | None
        """
        if self.hasImage():
            return self._pixmapHandle.pixmap().toImage()
        return None

    def mouseRectColorTransition(self, color):
        self.animation = QPropertyAnimation(self, b'pcolor')
        self.animation.setDuration(1000)
        self.animation.setStartValue(color)
        self.animation.setEndValue(self.mouse_color)
        self.animation.start()

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
            if self.canPan:
                self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.leftMouseButtonPressed.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.RightButton:
            if self.canZoom:
                self.setDragMode(QGraphicsView.RubberBandDrag)
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
            if self.canZoom:
                viewBBox = self.zoomStack[-1] if len(self.zoomStack) else self.sceneRect()
                selectionBBox = self.scene.selectionArea().boundingRect().intersected(viewBBox)
                self.scene.setSelectionArea(QPainterPath())  # Clear current selection area.
                if selectionBBox.isValid() and (selectionBBox != viewBBox):
                    self.zoomStack.append(selectionBBox)
                    self.updateViewer()
            self.setDragMode(QGraphicsView.NoDrag)
            self.rightMouseButtonReleased.emit(scenePos.x(), scenePos.y())

    def mouseDoubleClickEvent(self, event):
        """ Show entire image.
        """
        scenePos = self.mapToScene(event.pos())
        if event.button() == Qt.LeftButton:
            self.leftMouseButtonDoubleClicked.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.RightButton:
            if self.canZoom:
                self.zoomStack = []  # Clear zoom stack.
                self.updateViewer()
            self.rightMouseButtonDoubleClicked.emit(scenePos.x(), scenePos.y())
        QGraphicsView.mouseDoubleClickEvent(self, event)

    def _set_pcolor(self, color):
        pen = self.mouse_rect.pen()
        pen.setColor(color)
        self.mouse_rect.setPen(pen)

    pcolor = pyqtProperty(QColor, fset=_set_pcolor)
