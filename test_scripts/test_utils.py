# Imports
from latch.types import LatchFile, LatchDir
from latch import workflow
from typing import Optional, List, Union, Tuple
import sys, os
import unittest
sys.path.append('../wf')
from wf import get_seqs_from_inputs, get_holdouts, check_enum, Application

# Get the underlying functions (forgoes the @task since that's hard to run itself)
test_seqs_from_inputs = get_seqs_from_inputs.__wrapped__
test_get_holdouts = get_holdouts.__wrapped__
test_check_enum = check_enum.__wrapped__

class TestInputs(unittest.TestCase):
    
    def test_string_input(self):
        seqs = test_seqs_from_inputs(sequence=['LATCH', 'BIO'])
        self.assertEqual(seqs, [['LATCH', '0776181c35'], ['BIO', '13a4f1d101']])

    def test_fasta_input(self):
        seqs = test_seqs_from_inputs(sequence=[LatchFile('/root/test_scripts/test_data/seqs.fasta')])
        self.assertEqual(seqs, [['LATCH', 'seqs_protein1'], ['BIO', 'seqs_protein2']])

    def test_txt_input(self):
        seqs = test_seqs_from_inputs(sequence=[LatchFile('/root/test_scripts/test_data/seqs.txt')])
        self.assertEqual(seqs, [['LATCH', 'seqs_0776181c35'], ['BIO', 'seqs_13a4f1d101']])

    def test_dir_input(self):
        seqs = test_seqs_from_inputs(sequence=[LatchDir('/root/test_scripts/test_data')])
        self.assertEqual(seqs, [['LATCH', 'seqs_protein1'], ['BIO', 'seqs_protein2']])

    def test_multiple_input_types(self):
        seqs = test_seqs_from_inputs(sequence=[LatchFile('/root/test_scripts/test_data/seqs.fasta'), LatchFile('/root/test_scripts/test_data/seqs.txt')])
        self.assertEqual(seqs, [['LATCH', 'seqs_protein1'], ['BIO', 'seqs_protein2'], ['LATCH', 'seqs_0776181c35'], ['BIO', 'seqs_13a4f1d101']])


class TestHoldouts(unittest.TestCase):
    # [TODO] Figure out how to get beneath the @task wrappers to test tasks. Since
    # get_holdout calls get_seqs_from_inputs, for now I just made it call the underlying
    # function directly. This might also be better so it doesn't spin up a new node for such
    # a small function.
    def test_inputs(self):
        seqs = test_get_holdouts(sequence=['LATCH', 'BIO'])
        self.assertEqual(seqs, [['LATCH', '0776181c35'], ['BIO', '13a4f1d101']])

    def test_no_inputs(self):
        seqs = test_get_holdouts(sequence=[])
        self.assertEqual(seqs, None)

    def test_none(self):
        seqs = test_get_holdouts(sequence=None)
        self.assertEqual(seqs, None)

class TestCheckEnum(unittest.TestCase):

    def test_protein_rep(self):
        self.assertTrue(test_check_enum(application=Application.protein_rep), [True, False, False])
        self.assertTrue(test_check_enum(application=Application.babble), [False, True, False])
        self.assertTrue(test_check_enum(application=Application.evotune), [False, False, True])

if __name__ == "__main__":
    unittest.main()