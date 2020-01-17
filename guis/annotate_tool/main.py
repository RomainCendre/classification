import sys
import qdarkstyle
from pathlib import Path

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication
from guis.annotate_tool.sources.gui import QPatchExtractor
from toolbox.classification.parameters import Settings

if __name__ == '__main__':

    # Create the QApplication.
    icon_path = Path(__file__).parent/'images'/'icon.png'

    # Load resources
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(str(icon_path)))
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    pathologies = ['Normal', 'Benign', 'Malignant', 'Draw']
    home_path = Path().home()
    root_data = home_path/'Data/Skin/'
    input_files = ['Elisa.csv', 'JeanLuc.csv']

    # Launch gui
    viewer = QPatchExtractor(root_data, input_files, pathologies, settings=Settings.get_default_dermatology())

    # Show the viewer and run the application.
    viewer.show()
    sys.exit(app.exec_())
