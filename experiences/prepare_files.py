from os.path import expanduser, normpath, exists

from toolbox.IO.dermatology import DataManager

if __name__ == '__main__':

    home_path = expanduser("~")
    excluded_meta = ['Name', 'First_Name']
    # Prepare dermatology from Elisa datas
    output_folder = normpath('{home}/Data/Skin/Saint_Etienne/Elisa_DB/Patients'.format(home=home_path))
    if not exists(output_folder):
        origin_folder = normpath('{home}/Data/Skin/Saint_Etienne/Elisa_DB/Original'.format(home=home_path))
        DataManager(origin_folder).launch_converter(output_folder=output_folder, excluded_meta=excluded_meta)

    # Prepare dermatology from Hors serie datas
    output_folder = normpath('{home}/Data/Skin/Saint_Etienne/Hors_DB/Patients'.format(home=home_path))
    if not exists(output_folder):
        origin_folder = normpath('{home}/Data/Skin/Saint_Etienne/Hors_DB/Original'.format(home=home_path))
        DataManager(origin_folder).launch_converter(output_folder=output_folder, excluded_meta=excluded_meta)
