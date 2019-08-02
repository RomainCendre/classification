import sys
from pathlib import Path

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication

from toolbox.core.parameters import BuiltInSettings
from toolbox.gui_tools.patchifier.sources.gui import QPatchExtractor

if __name__ == '__main__':

    # Create the QApplication.
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('./images/icon.png'))
    pathologies = ['Normal', 'Benign', 'Malignant', 'Draw']
    home_path = Path().home()
    input_folder = home_path / 'Data/Skin/Saint_Etienne/Patients'
    viewer = QPatchExtractor(input_folder, pathologies,
                             settings=BuiltInSettings.get_default_dermatology())

    # Show the viewer and run the application.
    viewer.show()
    sys.exit(app.exec_())
