from pathlib import Path

from toolbox.IO.dermatology import DataManager

if __name__ == '__main__':
    excluded_meta = ['Name', 'First_Name']
    # Prepare dermatology from Elisa datas & OutDB
    home_path = Path().home()
    output_folder = home_path/'Data/Skin/Saint_Etienne/Patients'
    if not output_folder.exists():
        # Deal with Elisa database
        origin_folder = home_path/'Data/Skin/Saint_Etienne/Elisa_DB'
        DataManager(origin_folder).launch_converter(output_folder=output_folder, excluded_meta=excluded_meta)
        # Deal with JeanLuc database
        origin_folder = home_path/'Data/Skin/Saint_Etienne/JeanLuc_DB'
        DataManager(origin_folder).launch_converter(output_folder=output_folder, excluded_meta=excluded_meta)
