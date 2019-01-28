from os.path import expanduser, normpath, exists

from toolbox.IO.dermatology import DataManager

if __name__ == '__main__':

    home_path = expanduser("~")
    # Prepare dermatology from Elisa datas
    input_folder = normpath('{home}/Data/Skin/Saint_Etienne/Elisa_DB/Patients'.format(home=home_path))
    if not exists(input_folder):
        origin_folder = normpath('{home}/Data/Skin/Saint_Etienne/Elisa_DB/Original'.format(home=home_path))
        DataManager(origin_folder).launch_converter(input_folder)

    # Prepare dermatology from Hors serie datas
    input_folder = normpath('{home}/Data/Skin/Saint_Etienne/Hors_DB/Patients'.format(home=home_path))
    if not exists(input_folder):
        origin_folder = normpath('{home}/Data/Skin/Saint_Etienne/Hors_DB/Original'.format(home=home_path))
        DataManager(origin_folder).launch_converter(input_folder)

