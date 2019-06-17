from genericpath import exists
from os import makedirs
from tempfile import gettempdir

from experiments.dermatology.microscopy.descriptors import descriptors
from experiments.dermatology.microscopy.finetune import fine_tune
from experiments.dermatology.microscopy.multiscale_decisions import multiscale_decision
from experiments.dermatology.microscopy.sliding_decisions import sliding_decision
from experiments.dermatology.microscopy.sliding_features import sliding_features
from experiments.dermatology.microscopy.transferlearning import transfer_learning
from toolbox.core.parameters import DermatologyDataset

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
    # Whole
    images = DermatologyDataset.test_images()
    descriptors(images, output_folder)
    transfer_learning(images, output_folder)
    # fine_tune(images, output_folder)

    # Multiscale
    multiresolution_input = DermatologyDataset.test_multiresolution(coefficients=[1, 0.75, 0.5, 0.25])
    multiscale_decision(multiresolution_input, output_folder)

    # Sliding
    windows_inputs = [('NoOverlap', DermatologyDataset.test_sliding_images(size=250, overlap=0))]
    sliding_features(windows_inputs, output_folder)
    sliding_decision(windows_inputs, output_folder)
