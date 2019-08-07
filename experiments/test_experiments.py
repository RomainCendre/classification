import shutil
import unittest
from experiments.dermatology.microscopy.descriptors import descriptors
from experiments.dermatology.microscopy.finetune import fine_tune
from experiments.dermatology.microscopy.multiscale_decisions import multiscale_decision
from experiments.dermatology.microscopy.sliding_decisions import sliding_decisions
from experiments.dermatology.microscopy.sliding_features import sliding_features
from experiments.dermatology.microscopy.transferlearning import transfer_learning
from experiments.otorhinolaryngology.simple import simple
from toolbox.core.parameters import DermatologyDataset, ORLDataset


class TestORL(unittest.TestCase):

    def setUp(self):
        self.spectra = ORLDataset.test_spectras()
        self.output_folder = ORLDataset.get_results_location(is_test=True)

    def tearDown(self):
        print('Cleaning...')

    def test_orl_process(self):
        spectra = self.spectra
        simple(spectra, self.output_folder)


class TestMicroscopyWhole(unittest.TestCase):

    def setUp(self):
        self.microscopy = DermatologyDataset.test_images()
        self.output_folder = DermatologyDataset.get_results_location(is_test=True)

    def tearDown(self):
        shutil.rmtree(self.microscopy.get_working_folder(), ignore_errors=True)

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
        self.output_folder = DermatologyDataset.get_results_location(is_test=True)

    def tearDown(self):
        shutil.rmtree(self.microscopy.get_working_folder(), ignore_errors=True)

    def test_multiscale(self):
        multiscale_decision(self.microscopy, self.output_folder)


class TestMicroscopySliding(unittest.TestCase):

    def setUp(self):
        self.microscopy = DermatologyDataset.test_sliding_images(size=250, overlap=0)
        self.output_folder = DermatologyDataset.get_results_location(is_test=True)

    def tearDown(self):
        shutil.rmtree(self.microscopy.get_working_folder(), ignore_errors=True)

    def test_sliding_features(self):
        sliding_features([('Test', self.microscopy)], self.output_folder)

    def test_sliding_decisions(self):
        sliding_decisions([('Test', self.microscopy)], self.output_folder)
