import sys
import qdarkstyle
from pathlib import Path
from PyQt5.QtGui import QIcon, QPalette
from PyQt5.QtWidgets import QApplication, QFileDialog
from guis.demo_food.sources.gui import QDemo

if __name__ == '__main__':

    # External resources
    icon_path = Path(__file__).parent / 'images' / 'icon.png'

    # Load resources
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(str(icon_path)))
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    # Launch gui
    viewer = QDemo()
    viewer.load_data()
    viewer.show()

    sys.exit(app.exec_())
