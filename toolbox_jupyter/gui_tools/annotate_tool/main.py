import sys
from pathlib import Path

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication
from toolbox.gui_tools.annotate_tool.sources.gui import QPatchExtractor
from toolbox_jupyter.classification.parameters import Settings

if __name__ == '__main__':

    # Create the QApplication.
    app = QApplication(sys.argv)
    icon_path = Path(__file__).parent/'images'/'icon.png'
    app.setWindowIcon(QIcon(str(icon_path)))
    pathologies = ['Normal', 'Benign', 'Malignant', 'Draw']
    home_path = Path().home()
    input_folder = home_path / 'Data/Skin/Saint_Etienne/Patients'
    viewer = QPatchExtractor(input_folder, pathologies, settings=Settings.get_default_dermatology())

    # Show the viewer and run the application.
    viewer.show()
    sys.exit(app.exec_())
