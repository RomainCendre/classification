import unittest
from genericpath import exists
from os import makedirs
from tempfile import gettempdir
from numpy.ma import arange
from experiments.dermatology.microscopy.descriptors import descriptors
from experiments.dermatology.microscopy.finetune import fine_tune
from experiments.dermatology.microscopy.multiscale_decisions import multiscale_decision
from experiments.dermatology.microscopy.sliding_decisions import sliding_decisions
from experiments.dermatology.microscopy.sliding_features import sliding_features
from experiments.dermatology.microscopy.transferlearning import transfer_learning
from toolbox.core.parameters import DermatologyDataset, ORLDataset


class TestORL(unittest.TestCase):

    def setUp(self):
        self.spectra = ORLDataset.test_spectras()

    def tearDown(self):
        print('Cleaning...')

    def test_orl_process(self):
        spectra = self.spectra
        spectra.change_wavelength(wavelength=arange(start=445, stop=962, step=1))
        spectra.apply_average_filter(size=5)
        spectra.norm_patient()
        spectra.apply_scaling()
        # orl_simple(spectra, output_folder)


class TestMicroscopyWhole(unittest.TestCase):

    def setUp(self):
        self.microscopy = DermatologyDataset.test_images()
        # Output dir
        self.output_folder = gettempdir()
        if not exists(self.output_folder):
            makedirs(self.output_folder)

    def tearDown(self):
        print('Nettoyage !')

    def test_descriptors(self):
        descriptors(self.microscopy, self.output_folder)

    def test_transfer_learning(self):
        transfer_learning(self.microscopy, self.output_folder)

    def test_fine_tune(self):
        # fine_tune(self.microscopy, self.output_folder)
        print('Nop')


class TestMicroscopyMultiscale(unittest.TestCase):

    def setUp(self):
        self.microscopy = DermatologyDataset.test_multiresolution(coefficients=[1, 0.75, 0.5, 0.25])
        # Output dir
        self.output_folder = gettempdir()
        if not exists(self.output_folder):
            makedirs(self.output_folder)

    def tearDown(self):
        print('Nettoyage !')

    def test_multiscale(self):
        multiscale_decision(self.microscopy, self.output_folder)

class TestMicroscopySliding(unittest.TestCase):

    def setUp(self):
        self.microscopy = DermatologyDataset.test_sliding_images(size=250, overlap=0)
        # Output dir
        self.output_folder = gettempdir()
        if not exists(self.output_folder):
            makedirs(self.output_folder)

    def tearDown(self):
        print('Nettoyage !')

    def test_sliding_features(self):
        sliding_features(self.microscopy, self.output_folder)

    def test_sliding_decisions(self):
        sliding_decisions(self.microscopy, self.output_folder)
