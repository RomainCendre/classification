import sys
from pathlib import Path
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QFileDialog
from guis.demo_food.sources.gui import QDemo
from toolbox.classification.common import IO

if __name__ == '__main__':

    # External resources
    icon_path = Path(__file__).parent / 'images' / 'icon.png'
    style_path = Path(__file__).parent / 'style' / 'style.qss'

    # Load resources
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(str(icon_path)))
    app.setStyleSheet(open(style_path, "r").read())

    # Launch gui
    viewer = QDemo()
    # viewer.load_data()
    viewer.show()

    sys.exit(app.exec_())
