import shutil
import unittest
from experiments.dermatology.microscopy.b_manual import manual
from experiments.dermatology.microscopy.d_features import features
from experiments.otorhinolaryngology.simple import simple
from toolbox.core.parameters import DermatologyDataset, ORLDataset


class TestORL(unittest.TestCase):

    def setUp(self):
        self.spectra = ORLDataset.test_spectras()
        self.output_folder = ORLDataset.get_results_location(is_test=True)

    def tearDown(self):
        print('Cleaning...')
        shutil.rmtree(self.output_folder, ignore_errors=True)
        print('... Achieved!')

    def test_orl_process(self):
        spectra = self.spectra
        simple(spectra, self.output_folder)


class TestMicroscopyWhole(unittest.TestCase):

    def setUp(self):
        self.microscopy = DermatologyDataset.test_images()
        self.microscopy = self.microscopy.copy_and_change({'Type': 'Patch'})
        self.output_folder = DermatologyDataset.get_results_location(is_test=True)

    def tearDown(self):
        print('Cleaning...')
        shutil.rmtree(self.microscopy.get_working_folder(), ignore_errors=True)
        shutil.rmtree(self.output_folder, ignore_errors=True)
        print('... Achieved!')

    def test_manual(self):
        manual(self.microscopy, self.output_folder)

    def test_deep(self):
        # fine_tune(self.microscopy_old, self.output_folder)
        print('Nop')


class TestMicroscopySliding(unittest.TestCase):
    def setUp(self):
        self.microscopy = DermatologyDataset.test_sliding_images(size=250, overlap=0)
        self.output_folder = DermatologyDataset.get_results_location(is_test=True)

    def tearDown(self):
        print('Cleaning...')
        shutil.rmtree(self.microscopy.get_working_folder(), ignore_errors=True)
        shutil.rmtree(self.output_folder, ignore_errors=True)
        print('... Achieved!')

    def test_features(self):
        features(self.microscopy, self.output_folder)

    def test_decision(self):
        # fine_tune(self.microscopy_old, self.output_folder)
        print('Nop')
