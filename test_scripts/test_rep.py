# Imports
from pathlib import Path
from latch.types import LatchFile, LatchDir
from latch import workflow
import numpy as np
from typing import Optional, List, Union, Tuple
import sys, os
import unittest

sys.path.append('../wf')
from wf import rep_task, Application, ModelSize

# Get the underlying functions (forgoes the @task since that's hard to run itself)
test_rep_task = rep_task.__wrapped__

class TestInputs(unittest.TestCase):

    def test_large_input(self):
        test_rep_task(
            seqs_and_names = [['LATCH', 'protein1'], ['LATCH', 'protein2']],
            model_size = ModelSize.large,
            model_params = None,
            run_name = "basic test",
        )

        # Validate correct representation sizes
        aggregated_path = Path("/root/outputs/unireps.npy")
        aggregated = np.load(aggregated_path)
        self.assertEqual(aggregated.shape, (2, int(ModelSize.large.value)))

    def test_medium_input(self):
        test_rep_task(
            seqs_and_names = [['LATCH', 'protein1'], ['LATCH', 'protein2']],
            model_size = ModelSize.medium,
            model_params = None,
            run_name = "basic medium test",
        )
        # Validate correct representation sizes
        aggregated_path = Path("/root/outputs/unireps.npy")
        aggregated = np.load(aggregated_path)
        self.assertEqual(aggregated.shape, (2, int(ModelSize.medium.value)))

    def test_small_input(self):
        test_rep_task(
            seqs_and_names = [['LATCH', 'protein1'], ['LATCH', 'protein2']],
            model_size = ModelSize.small,
            model_params = None,
            run_name = "basic small test",
        )
        # Validate correct representation sizes
        aggregated_path = Path("/root/outputs/unireps.npy")
        aggregated = np.load(aggregated_path)
        self.assertEqual(aggregated.shape, (2, int(ModelSize.small.value)))

    def test_large_custom_model(self):
        test_rep_task(
            seqs_and_names = [['LATCH', 'protein1'], ['LATCH', 'protein2']],
            model_size = ModelSize.large,
            model_params = LatchFile("/root/test_scripts/test_data/models/large_model.pkl"),
            run_name = "custom model large test",
        )
        # Validate correct representation sizes
        aggregated_path = Path("/root/outputs/unireps.npy")
        aggregated = np.load(aggregated_path)
        self.assertEqual(aggregated.shape, (2, int(ModelSize.large.value)))

    def test_medium_custom_model(self):
        test_rep_task(
            seqs_and_names = [['LATCH', 'protein1'], ['LATCH', 'protein2']],
            model_size = ModelSize.medium,
            model_params = LatchFile("/root/test_scripts/test_data/models/medium_model.pkl"),
            run_name = "custom model medium test",
        )
        # Validate correct representation sizes
        aggregated_path = Path("/root/outputs/unireps.npy")
        aggregated = np.load(aggregated_path)
        self.assertEqual(aggregated.shape, (2, int(ModelSize.medium.value)))
    
    def test_small_custom_model(self):
        test_rep_task(
            seqs_and_names = [['LATCH', 'protein1'], ['LATCH', 'protein2']],
            model_size = ModelSize.small,
            model_params = LatchFile("/root/test_scripts/test_data/models/small_model.pkl"),
            run_name = "custom model small test",
        )
        # Validate correct representation sizes
        aggregated_path = Path("/root/outputs/unireps.npy")
        aggregated = np.load(aggregated_path)
        self.assertEqual(aggregated.shape, (2, int(ModelSize.small.value)))

    def test_check_model_size_mismatch(self):
        with self.assertRaises(ValueError) as cm:
            test_rep_task(
                seqs_and_names = [['LATCH', 'protein1'], ['LATCH', 'protein2']],
                model_size = ModelSize.small,
                model_params = LatchFile("/root/test_scripts/test_data/models/large_model.pkl"),
                run_name = "custom model size mismatch test",
            )

    
if __name__ == "__main__":
    unittest.main()