from os.path import expanduser, normpath, exists

from toolbox.IO.dermatology import DataManager

if __name__ == '__main__':

    home_path = expanduser("~")
    excluded_meta = ['Name', 'First_Name']
    # Prepare dermatology from Elisa datas & OutDB
    output_folder = normpath('{home}/Data/Skin/Saint_Etienne/Patients'.format(home=home_path))
    if not exists(output_folder):
        origin_folder = normpath('{home}/Data/Skin/Saint_Etienne/Elisa_DB'.format(home=home_path))
        DataManager(origin_folder).launch_converter(output_folder=output_folder, excluded_meta=excluded_meta)
        origin_folder = normpath('{home}/Data/Skin/Saint_Etienne/JeanLuc_DB'.format(home=home_path))
        DataManager(origin_folder).launch_converter(output_folder=output_folder, excluded_meta=excluded_meta)
