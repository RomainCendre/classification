import pandas as pd
from pathlib import Path
from PyQt5 import QtWidgets
from natsort import natsorted
from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QRect, QPropertyAnimation, pyqtProperty
from PyQt5.QtGui import QImage, QPixmap, QColor, QPen, QBrush
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGridLayout, QMainWindow, QWidget, QLabel, \
    QSpinBox, QGraphicsRectItem, QProgressBar, QVBoxLayout, \
    QTableWidget, QTabWidget, QTableWidgetItem, QAbstractItemView, QGraphicsItemGroup, QComboBox
from toolbox.IO import dermatology


class QPatchExtractor(QMainWindow):
    # Modes
    LABEL = 0
    PATCH = 1
    BBOX = 2

    def __init__(self, input_folder, pathologies, settings):
        super(QPatchExtractor, self).__init__()
        self.patient_index = 0
        self.image_index = 0
        self.patch_index = 0
        self.patient = None
        self.images = None
        self.patches = None
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
        # First label tool
        self.label_widget = QLabelWidget(self.annotate_widget, self.pathologies, self.settings)
        self.label_widget.change_label.connect(self.change_image_label)
        # Second patch tool
        self.patch_widget = QPatchWidget(self.annotate_widget, self.pathologies[:-1], self.settings)
        self.patch_widget.changed_patch_selection.connect(self.viewer.set_current_patch)
        self.patch_widget.changed_patch_size.connect(self.viewer.setRectangleSize)
        self.patch_widget.changed_mode.connect(self.viewer.change_mouse_color)
        self.patch_widget.set_value(250)
        self.patch_widget.send_key(0)
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
        if self.images is None or len(self.images) == 0 or move is None:
            self.image_index = 0
        else:
            self.image_index = (self.image_index + move) % len(self.images)
        # Open patches
        self.open_patches()
        # Send data to components
        self.label_widget.send_image(self.get_image())
        self.patch_widget.send_patches(self.get_patches())
        self.viewer.set_patches(self.get_patches_draw())
        self.update_image()

    def change_image_label(self, label):
        image = self.get_image()
        # Check everything is fine and needed
        if not (len(image) == 0 or image['Label'] == label):
            self.images.loc[image.name, 'Label'] = label
            # Send data to components
            self.label_widget.send_images(self.images)
        self.change_image(1)

    def change_mode(self):
        # Actions depend on mode
        if self.get_mode() == QPatchExtractor.LABEL:
            self.viewer.change_mouse_state(False)
        else:
            self.viewer.change_mouse_state(True)

    def change_patient(self, move):
        # Start by closing previous df
        self.close_images()
        self.close_patches()
        self.close_patient()
        # Change patient
        length = len(self.patients_directories)
        self.patient_index = (self.patient_index + move) % length
        # Open new df
        self.open_patient_and_images()
        self.update_patient()
        self.reset_image()

    def click_event(self, x, y):
        # Actions depend on mode
        if self.get_mode() == QPatchExtractor.LABEL:
            return
        # Acquire image and it to dataframe
        self.write_patch(x, y)

    def close_images(self, save=True):
        if self.images is not None and save:
            self.images.drop(columns=['Full_Path', 'Type', 'Reference2'], errors='ignore').to_csv(self.get_patient_folder()/'images.csv', index=False)
        self.images = None

    def close_patches(self, save=False):
        if self.patches is not None and save:
            self.patches.drop(columns=['Full_Path', 'Type'], errors='ignore').to_csv(self.get_patient_folder()/'patches.csv', index=False)
        self.patches = None

    def close_patient(self):
        self.patient = None

    def closeEvent(self, event):
        self.close_images()
        self.close_patches()
        self.close_patient()
        super().closeEvent(event)

    def delete_patch(self):
        index = self.patches.index[self.patch_widget.get_patch_selected()]
        # Delete patch and row
        Path(self.patches.loc[index, 'Full_Path']).unlink()
        self.patches = self.patches.drop(index)
        # Close and reopen file
        self.close_patches(save=True)
        self.open_patches()
        self.update_patch()

    def extract_patch(self, x, y):
        # Load image
        raw_image = self.viewer.image()
        image_rect = raw_image.rect()

        # Compute patch position
        size = self.patch_widget.get_size()
        patch_rect = QRect(x - size / 2, y - size / 2, size, size)

        # Test if patch rectangle is full
        if not image_rect.intersected(patch_rect) == patch_rect:
            return None

        # Extract patch
        return raw_image.copy(patch_rect)

    def get_color(self, label):
        color_tuple = self.settings.get_color(label)
        return QColor.fromRgbF(color_tuple[0], color_tuple[1], color_tuple[2], 0.75)

    def get_image(self):
        # If doesnt exist or not valid return empty one
        if self.images is None or self.image_index >= len(self.images):
            return pd.Series()
        return self.images.iloc[self.image_index]

    def get_patches(self):
        if self.patches is None:
            return None
        return self.patches[self.patches.Source == self.get_image().get('Path', 'NA')]

    def get_mode(self):
        return self.annotate_widget.currentIndex()

    def get_patches_draw(self):
        patches = self.get_patches()
        patches_draw = []
        # If not patches found return
        if patches is None:
            return patches_draw
        # Else construct patches
        for index, row in patches.iterrows():
            patches_draw.append((int(row['Center_X']), int(row['Center_Y']),
                                 int(row['Width']), self.get_color(row['Label'])))
        return patches_draw

    def get_patient_folder(self):
        return self.patients_directories[self.patient_index]

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

    def open_patient_and_images(self):
        self.patient = dermatology.Reader.read_patient_file(self.get_patient_folder())
        self.images = dermatology.Reader.read_images_file(self.get_patient_folder(), modality='Microscopy')
        if self.patient is None or len(self.patient) == 0:
            return
        self.patient = self.patient.iloc[0]

    def open_patches(self):
        self.patches = dermatology.Reader.read_patches_file(self.get_patient_folder(),
                                                            modality='Microscopy')

    def reset_image(self):
        self.change_image(None)

    def tool_controls(self, key):
        if self.get_mode() == QPatchExtractor.LABEL:
            self.label_widget.send_key(key - Qt.Key_0)
        else:
            if key == Qt.Key_Delete:
                self.delete_patch()
            else:
                self.patch_widget.send_key(key - Qt.Key_0)

    def update_image(self):
        self.image_bar.setValue(self.image_index)
        self.image_bar.setFormat(str(self.get_image().get('Path', '')))
        self.viewer.loadImage(Path(self.get_image().get('Full_Path', '')))

    def update_patch(self):
        self.patch_widget.send_patches(self.get_patches())
        self.viewer.set_patches(self.get_patches_draw())

    def update_patient(self):
        self.patient_bar.setValue(self.patient_index)
        if self.patient is None:
            id, diagnosis, bdiagnosis = 'NA', 'NA', 'NA'
        else:
            id = self.patient.get('ID', 'NA')
            diagnosis = self.patient.get('Diagnosis', 'NA')
            bdiagnosis = self.patient.get('Binary_Diagnosis', 'NA')
        self.patient_bar.setFormat(F'Patient: {id} - Pathology: {diagnosis}({bdiagnosis})')
        self.label_widget.send_images(self.images)
        # Update image
        if self.images is None:
            self.image_bar.setRange(0, 0)
        else:
            self.image_bar.setRange(0, len(self.images) - 1)

    def write_patch(self, x, y):
        # Get patch
        patch = self.extract_patch(x, y)
        if not patch:
            self.viewer.mouse_color_transition(QColor(Qt.red))
            return
        # If valid check to construct it
        patch_folder = self.get_patient_folder() / 'patches'
        patch_folder.mkdir(exist_ok=True)
        patches = list(patch_folder.glob('*.bmp'))
        patches = [int(patch.stem) for patch in patches]
        if not patches:
            name = 0
        else:
            name = max(patches) + 1
        patch_file = patch_folder / '{name}.bmp'.format(name=name)
        # Prepare and write patch
        patch_data = pd.Series()
        patch_data['Modality'] = 'Microscopy'
        patch_data['Path'] = patch_file.name
        patch_data['Height'] = patch.height()
        patch_data['Width'] = patch.width()
        patch_data['Center_X'] = int(x)
        patch_data['Center_Y'] = int(y)
        patch_data['Label'] = self.patch_widget.get_mode()
        patch_data['Source'] = self.get_image().get('Path', 'NA')
        if self.patches is None:
            self.patches = patch_data.to_frame().transpose()
        else:
            self.patches = self.patches.append(patch_data, ignore_index=True)
        patch.save(str(patch_file), format='bmp')
        # Save and reload
        self.close_patches(save=True)
        self.open_patches()
        self.update_patch()
        # Everything well done
        self.viewer.mouse_color_transition(QColor(Qt.green))


class QLabelWidget(QWidget):
    # Signals
    change_label = pyqtSignal(str)

    def __init__(self, parent, pathologies, settings):
        super(QLabelWidget, self).__init__(parent)
        self.pathologies = pathologies
        self.settings = settings
        self.default_value = len(pathologies) - 1
        self.init_gui()

    def change_mode(self, index):
        self.update_color(index)
        self.change_label.emit(self.pathologies[index])

    def event_changed(self, current, previous):
        self.change_mode(current.row())

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
        self.table.selectionModel().currentRowChanged.connect(self.event_changed, Qt.QueuedConnection)
        patch_layout = QVBoxLayout(self)
        patch_layout.addWidget(self.table)

    def send_image(self, data):
        current = data.get('Label', 'NA')
        try:
            index = self.pathologies.index(current)
        except:
            index = self.default_value
        self.table.selectionModel().blockSignals(True)
        self.table.selectRow(index)
        self.update_color(index)
        self.table.selectionModel().blockSignals(False)

    def send_images(self, data):
        # Count occurrences
        if data is None:
            values = None
        else:
            values = data['Label'].value_counts()
        # Now fill fields
        for index, label in enumerate(self.pathologies):
            if values is None:
                value = 0
            else:
                value = values.get(label, 0)
            self.table.setItem(index, 2, QTableWidgetItem('{value}'.format(value=value)))

    def send_key(self, key):
        if key < 0 or key >= len(self.pathologies):
            return
        self.table.selectionModel().blockSignals(True)
        self.table.selectRow(key)
        self.table.selectionModel().blockSignals(False)
        self.change_mode(key)

    def update_color(self, index):
        color_tuple = self.settings.get_color(self.pathologies[index])
        qcolor = QColor.fromRgbF(color_tuple[0], color_tuple[1], color_tuple[2], 0.75)
        self.table.setStyleSheet('QTableView{selection-background-color: ' + qcolor.name() + '}')


class QPatchWidget(QWidget):
    # Signals
    changed_patch_size = pyqtSignal(int)
    changed_patch_selection = pyqtSignal(int)
    changed_mode = pyqtSignal(QColor)

    def __init__(self, parent, pathologies, settings):
        super(QPatchWidget, self).__init__(parent)
        self.pathologies = pathologies
        self.settings = settings
        self.init_gui()

    def init_gui(self):
        # Create table that list current patches
        self.table = QTableWidget()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(('Center X', 'Center Y', 'Size', 'Label'))
        self.table.resizeColumnsToContents()
        hheader = self.table.horizontalHeader()
        hheader.setStretchLastSection(True)
        self.table.selectionModel().currentRowChanged.connect(self.row_changed)
        # Manage patch mode
        self.mode = QComboBox()
        for index, pathology in enumerate(self.pathologies):
            self.mode.addItem(pathology)
            self.mode.setItemData(index, QBrush(self.get_color(pathology)), Qt.BackgroundColorRole)
        self.mode.setCurrentIndex(-1)
        self.mode.currentIndexChanged.connect(self.mode_changed)
        # Manage patch size
        self.size = QSpinBox()
        self.size.valueChanged.connect(self.changed_patch_size.emit)
        self.size.setRange(0, 1000)
        self.size.setSingleStep(10)
        self.size.setSuffix("px")
        self.size.setEnabled(False)
        # Then build patch tool
        patch_layout = QGridLayout(self)
        patch_layout.addWidget(self.table, 0, 0, 1, 4)
        patch_layout.addWidget(QLabel('Mode'), 1, 0)
        patch_layout.addWidget(self.mode, 1, 1)
        patch_layout.addWidget(QLabel('Width/Height'), 1, 2)
        patch_layout.addWidget(self.size, 1, 3)

    def get_color(self, label):
        color_tuple = self.settings.get_color(label)
        return QColor.fromRgbF(color_tuple[0], color_tuple[1], color_tuple[2], 0.75)

    def get_mode(self):
        return self.mode.currentText()

    def get_patch_selected(self, is_index=True):
        if is_index:
            return self.table.currentRow()
        else:
            return self.table.currentRow()

    def get_size(self):
        return self.size.value()

    def row_changed(self, current, previous):
        if current.row() == -1:
            return
        self.changed_patch_selection.emit(current.row())

    def mode_changed(self, mode):
        self.changed_mode.emit(self.get_color(self.pathologies[mode]))

    def send_key(self, key):
        if key < self.mode.count():
            self.mode.setCurrentIndex(key)

    def send_patches(self, patches):
        # Empty the table
        self.table.setRowCount(0)
        # Check data is valid
        if patches is None or len(patches) == 0:
            return
        self.table.setRowCount(len(patches))
        for index, (rindex, row) in enumerate(patches.iterrows()):
            # Get current label
            current_label = row['Label']
            # Get current color
            qcolor = self.get_color(current_label)
            # Create table items
            item = QTableWidgetItem('{center}'.format(center=row['Center_X']))
            item.setBackground(qcolor)
            item.setData(0, row.name)
            self.table.setItem(index, 0, item)
            item = QTableWidgetItem('{center}'.format(center=row['Center_Y']))
            item.setBackground(qcolor)
            self.table.setItem(index, 1, item)
            item = QTableWidgetItem('{size}'.format(size=row['Width']))
            item.setBackground(qcolor)
            self.table.setItem(index, 2, item)
            item = QTableWidgetItem('{label}'.format(label=row['Label']))
            item.setBackground(qcolor)
            self.table.setItem(index, 3, item)

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
        self.default_color = QColor(Qt.blue)
        self.mouse_rect = QGraphicsRectItem(-25, -25, 50, 50)
        self.mouse_rect.setPen(QPen(self.default_color, 6, Qt.DotLine))
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
        self.mouse_state = True
        self.setMouseTracking(True)

    def change_mouse_color(self, color):
        self.mouse_color = color
        self.mouse_rect.setPen(QPen(self.mouse_color, 6, Qt.DotLine))

    def change_mouse_state(self, enable=True):
        self.mouse_state = enable
        if self.mouse_state:
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

    def mouse_color_transition(self, color):
        if self.mouse_state:
            self.animation = QPropertyAnimation(self, b'pcolor')
            self.animation.setDuration(1000)
            self.animation.setStartValue(color)
            end_color = QColor(color)
            end_color.setAlphaF(0)
            self.animation.setEndValue(end_color)
            self.animation.start()

    def loadImage(self, path):
        """ Load an image from file.
        Without any arguments, loadImageFromFile() will popup a file dialog to choose the image file.
        With a fileName argument, loadImageFromFile(fileName) will attempt to load the specified image file directly.
        """
        if path is None or not path.is_file():
            self.setImage(QImage())
        else:
            image = QImage(str(path))
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
        # Checks
        if index == -1:
            return
        # Delete brush
        no_brush = QBrush()
        for item in self.patches.childItems():
            item.setBrush(no_brush)
        # Create one for selection
        color = QColor(self.default_color)
        color.setAlphaF(0.25)
        self.patches.childItems()[index].setBrush(QBrush(color))

    def set_patches(self, patches):
        # Delete patches items
        for patch in self.patches.childItems():
            self.patches.removeFromGroup(patch)
        # Create new items
        for patch in patches:
            patch_item = QGraphicsRectItem(patch[0] - (patch[2] / 2), patch[1] - (patch[2] / 2), patch[2], patch[2])
            patch_item.setPen(QPen(patch[3], 6, Qt.SolidLine))
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
        brush = self.mouse_rect.brush()
        brush.setColor(color)
        self.mouse_rect.setBrush(color)

    pcolor = pyqtProperty(QColor, fset=_set_pcolor)
