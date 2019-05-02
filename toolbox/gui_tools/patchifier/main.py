import sys

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication
from toolbox.IO.datasets import Dataset
from toolbox.gui_tools.patchifier.sources.gui import QPatchExtractor

if __name__ == '__main__':

    # Create the QApplication.
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('./images/icon.png'))
    inputs = Dataset.full_images()
    inputs = inputs.sub_inputs({'Modality': 'Microscopy'})
    pathologies = ['Healthy', 'Benign', 'Malignant']
    viewer = QPatchExtractor(inputs)

    # Show the viewer and run the application.
    viewer.show()
    sys.exit(app.exec_())
