from genericpath import exists
from os import makedirs
from tempfile import gettempdir

from numpy import arange

from experiments.dermatology.microscopy.full.simple import simple as mic_simple
from experiments.otorhinolaryngology.simple import simple as orl_simple
from toolbox.core.parameters import ORLDataset, DermatologyDataset

if __name__ == "__main__":
    # Output dir
    output_folder = gettempdir()
    if not exists(output_folder):
        makedirs(output_folder)

    # ORL Tests
    # spectra = ORLDataset.test_spectras()
    # spectra.change_wavelength(wavelength=arange(start=445, stop=962, step=1))
    # spectra.apply_average_filter(size=5)
    # spectra.norm_patient()
    # spectra.apply_scaling()
    # orl_simple(spectra, output_folder)

    # Dermatology Tests
    images = DermatologyDataset.test_images()
    mic_simple(images, output_folder)
