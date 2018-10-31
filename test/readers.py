from os.path import expanduser

from IO import dermatology
from IO import otorhinolaryngology

if __name__ == '__main__':
    # Load data references
    home_path = expanduser("~")
    patients = dermatology.Reader(';').scan_folder('{home}\\Data\\Skin\\Patients'.format(home=home_path))
    print('Patients {found} founds'.format(found=len(patients)))
    patients = otorhinolaryngology.Reader(';').read_table('{home}\\Data\\Neck\\Patients.csv'.format(home=home_path))
    print('Patients {found} founds'.format(found=len(patients)))
