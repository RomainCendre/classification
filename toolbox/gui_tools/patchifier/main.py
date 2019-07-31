import sys
from os import makedirs
from os.path import expanduser, exists, normpath
from pathlib import Path

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication

from toolbox.core.parameters import DermatologyDataset, BuiltInSettings
from toolbox.gui_tools.patchifier.sources.gui import QPatchExtractor

if __name__ == '__main__':

    home_path = expanduser('~')
    patch_folder = '{home}/Data/Skin/Patches'.format(home=home_path)
    patch_folder = normpath(patch_folder)
    if not exists(patch_folder):
        makedirs(patch_folder)
    # Create the QApplication.
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('./images/icon.png'))
    pathologies = ['Normal', 'Benign', 'Malignant', 'Draw']
    home_path = Path().home()
    input_folder = home_path / 'Data/Skin/Saint_Etienne/Patients'
    viewer = QPatchExtractor(input_folder, pathologies,
                             settings=BuiltInSettings.get_default_dermatology(),
                             output=patch_folder)

    # Show the viewer and run the application.
    viewer.show()
    sys.exit(app.exec_())
