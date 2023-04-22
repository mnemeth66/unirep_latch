# Imports
from pathlib import Path
from latch.types import LatchFile, LatchDir
from latch import workflow
import numpy as np
from typing import Optional, List, Union, Tuple
import sys, os
import unittest
import glob
from subprocess import CalledProcessError

sys.path.append('../wf')
from wf import rep_task, Application, ModelSize

# Get the underlying functions (forgoes the @task because I ran into issues with Flyte/Latch
# types while running the task)
test_rep_task = rep_task.__wrapped__

def validate_rep_sizes(self, size, run_name):
    # Get all _unirep.npy files in /root/outputs
    unirep_files = glob.glob(os.path.join(f'/root/outputs/{run_name}/', "*_unirep.npy"))
    # aggregated_path = Path("/root/outputs/unireps.npy")
    print(unirep_files)
    for f in unirep_files:
        aggregated = np.load(f)
        print(aggregated.shape, (size,))
        self.assertEqual(aggregated.shape, (size,))

class TestInputs(unittest.TestCase):

    def test_large_input(self):
        run_name = "large input test"
        test_rep_task(
            seqs_and_names = [['LATCH', 'protein1'], ['LATCH', 'protein2']],
            model_size = ModelSize.large,
            model_params = None,
            run_name = run_name,
        )

        # Validate correct representation sizes
        size = int(ModelSize.large.value)
        validate_rep_sizes(self, size, run_name)

    def test_medium_input(self):
        run_name = "medium input test"
        test_rep_task(
            seqs_and_names = [['LATCH', 'protein1'], ['LATCH', 'protein2']],
            model_size = ModelSize.medium,
            model_params = None,
            run_name = run_name,
        )
        # Validate correct representation sizes
        size = int(ModelSize.medium.value)
        validate_rep_sizes(self, size, run_name)

    def test_small_input(self):
        run_name = "small input test"
        test_rep_task(
            seqs_and_names = [['LATCH', 'protein1'], ['LATCH', 'protein2']],
            model_size = ModelSize.small,
            model_params = None,
            run_name = run_name,
        )
        # Validate correct representation sizes
        size = int(ModelSize.small.value)
        validate_rep_sizes(self, size, run_name)

    def test_large_custom_model(self):
        run_name = 'custom model large test'
        test_rep_task(
            seqs_and_names = [['LATCH', 'protein1'], ['LATCH', 'protein2']],
            model_size = ModelSize.large,
            model_params = LatchFile("s3://latch-public/test-data/3192/unirep_test_data/large_model.pkl"),
            run_name = run_name,
        )
        # Validate correct representation sizes
        size = int(ModelSize.large.value)
        validate_rep_sizes(self, size, run_name)

    def test_medium_custom_model(self):
        run_name = 'custom model medium test'
        test_rep_task(
            seqs_and_names = [['LATCH', 'protein1'], ['LATCH', 'protein2']],
            model_size = ModelSize.medium,
            model_params = LatchFile("s3://latch-public/test-data/3192/unirep_test_data/medium_model.pkl"),
            run_name = run_name,
        )
        # Validate correct representation sizes
        size = int(ModelSize.medium.value)
        validate_rep_sizes(self, size, run_name)
    
    def test_small_custom_model(self):
        run_name = 'custom model small test'
        test_rep_task(
            seqs_and_names = [['LATCH', 'protein1'], ['LATCH', 'protein2']],
            model_size = ModelSize.small,
            model_params = LatchFile("s3://latch-public/test-data/3192/unirep_test_data/small_model.pkl"),
            run_name = run_name,
        )
        # Validate correct representation sizes
        size = int(ModelSize.small.value)
        validate_rep_sizes(self, size, run_name)

    def test_check_model_size_mismatch(self):
        run_name = 'custom model size mismatch test'
        with self.assertRaises(CalledProcessError) as cm:
            test_rep_task(
                seqs_and_names = [['LATCH', 'protein1'], ['LATCH', 'protein2']],
                model_size = ModelSize.small,
                model_params = LatchFile("s3://latch-public/test-data/3192/unirep_test_data/large_model.pkl"),
                run_name = run_name,
            )