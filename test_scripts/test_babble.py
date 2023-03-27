# Imports
from latch.types import LatchFile, LatchDir
from latch import workflow
from typing import Optional, List, Union, Tuple
from pathlib import Path
import sys, os
import unittest
from subprocess import CalledProcessError
import numpy as np
import shutil

sys.path.append('../wf')
from wf import babble_task, Application, ModelSize
from scripts.babble import pkl_to_model


cas9 = "MKRNYILGLDIGITSVGYGIIDYETRDVIDAGVRLFKEANVENNEGRRSKRGARRLKRRRRHRIQRVKKLLFDYNLLTDHSELSGINPYEARVKGLSQKLSEEEFSAALLHLAKRRGVHNVNEVEEDTGNELSTKEQISRNSKALEEKYVAELQLERLKKDGEVRGSINRFKTSDYVKEAKQLLKVQKAYHQLDQSFIDTYIDLLETRRTYYEGPGEGSPFGWKDIKEWYEMLMGHCTYFPEELRSVKYAYNADLYNALNDLNNLVITRDENEKLEYYEKFQIIENVFKQKKKPTLKQIAKEILVNEEDIKGYRVTSTGKPEFTNLKVYHDIKDITARKEIIENAELLDQIAKILTIYQSSEDIQEELTNLNSELTQEEIEQISNLKGYTGTHNLSLKAINLILDELWHTNDNQIAIFNRLKLVPKKVDLSQQKEIPTTLVDDFILSPVVKRSFIQSIKVINAIIKKYGLPNDIIIELAREKNSKDAQKMINEMQKRNRQTNERIEEIIRTTGKENAKYLIEKIKLHDMQEGKCLYSLEAIPLEDLLNNPFNYEVDHIIPRSVSFDNSFNNKVLVKQEENSKKGNRTPFQYLSSSDSKISYETFKKHILNLAKGKGRISKTKKEYLLEERDINRFSVQKDFINRNLVDTRYATRGLMNLLRSYFRVNNLDVKVKSINGGFTSFLRRKWKFKKERNKGYKHHAEDALIIANADFIFKEWKKLDKAKKVMENQMFEEKQAESMPEIETEQEYKEIFITPHQIKHIKDFKDYKYSHRVDKKPNRELINDTLYSTRKDDKGNTLIVNNLNGLYDKDNDKLKKLINKSPEKLLMYHHDPQTYQKLKLIMEQYGDEKNPLYKYYEETGNYLTKYSKKDNGPVIKKIKYYGNKLNAHLDITDDYPNSRNKVVKLSLKPYRFDVYLDNGVYKFVTVKNLDVIKKENYYEVNSKCYEEAKKLKKISNQAEFIASFYNNDLIKINGELYRVIGVNNDLLNRIEVNMIDITYREYLENMNDKRPPRIIKTIASKTQSIKKYSTDILGNLYEVKSKKHPQIIKKG"

# Get the underlying functions (forgoes the @task because I ran into issues with Flyte/Latch
# types while running the task)
test_babble = babble_task.__wrapped__

def validate_babble(self, protein):
    local_dir = Path("/root/outputs/")
    # confirm the babble length is correct by processing the last value in the csv
    with open(local_dir / 'babble_results.csv', 'r') as f:
        lines = f.readlines()
        babble_line = lines[-1]
        babble = babble_line.split(',')[-1].strip()
        self.assertEqual(len(babble), len(protein))

class TestBabble(unittest.TestCase):

    def test_basic_valid_babble(self):
        protein = 'LATCH'
        test_babble(
                seqs_and_names=[[protein, 'seqs_protein1']],
                model_size=ModelSize.small,
                model_params=None,
                run_name="basic test",
                length=10,
                temp=1,
        )
        validate_babble(self, protein)
        # Can't assert equal for now since it seems like the tests run concurrently,
        # rather than in the order they were run.
        # self.assertEqual(files, ['babble_results.csv', 'seqs_protein1'])
        # ie. it wouldn't be test_1 -> test_1 validate -> test_2 -> test_2 validate ..., but
        # eg. test_1 -> test_2 --> test_1 validate -> test_3 -> test_2 validate ...

    def test_small_babble_length(self):
        protein = cas9
        test_babble(
                seqs_and_names=[[protein, 'seqs_protein1']],
                model_size=ModelSize.small,
                model_params=None,
                run_name="small babble length test",
                length=10,
                temp=1,
        )
        validate_babble(self, protein)

    def test_small_babble_length2(self):
        protein = 'LATCHLATCH'
        test_babble(
                seqs_and_names=[[protein, 'seqs_protein1']],
                model_size=ModelSize.small,
                model_params=None,
                run_name="smaller babble length test",
                length=10,
                temp=1,
        )
        validate_babble(self, protein)

    def test_invalid_amino_acids(self):
        protein = 'LATCHBIO'
        test_babble(
                seqs_and_names=[[protein, 'seqs_protein1']],
                model_size=ModelSize.small,
                model_params=None,
                run_name="invalid aa test",
                length=10,
                temp=1,
        )
        expected_invalid_protein = 'invalid sequence'
        validate_babble(self, expected_invalid_protein)

    def test_model_param_input(self):
        protein = 'LATCH'
        test_babble(
                seqs_and_names=[[protein, 'seqs_protein1']],
                model_size=ModelSize.small,
                model_params=LatchFile("s3://latch-public/test-data/3192/unirep_test_data/small_model.pkl"),
                run_name="custom model test",
                length=10,
                temp=1,
        )
        validate_babble(self, protein)

    def test_model_mismatched_size(self):
        protein = 'LATCH'
        with self.assertRaises(CalledProcessError) as cm:
            test_babble(
                    seqs_and_names=[[protein, 'seqs_protein1']],
                    model_size=ModelSize.large,
                    model_params=LatchFile("s3://latch-public/test-data/3192/unirep_test_data/small_model.pkl"),
                    run_name="custom model test",
                    length=10,
                    temp=1,
            )
        # print(cm)


def validate_pkl_to_model(self, pkl_model, tf_model_folder):
    model_folder = pkl_to_model(pkl_model.local_path)
    # Check that they have the same files and that each file has the correct numpy shape
    self.assertEqual(len(os.listdir(model_folder)), len(os.listdir(tf_model_folder)))
    for file in os.listdir(model_folder):
        self.assertEqual(
                np.load(os.path.join(model_folder, file)).shape,
                np.load(os.path.join(tf_model_folder, file)).shape,
        ) 
    shutil.rmtree(model_folder)

class TestModelConversion(unittest.TestCase):

    def test_pkl_to_tf_small(self):
        pkl_model = LatchFile("s3://latch-public/test-data/3192/unirep_test_data/small_model.pkl")
        tf_model_folder = LatchDir("s3://latch-public/test-data/3192/unirep_test_data/small_model_weights").local_path
        validate_pkl_to_model(self, pkl_model, tf_model_folder)
    
    def test_pkl_to_tf_medium(self):
        pkl_model = LatchFile("s3://latch-public/test-data/3192/unirep_test_data/medium_model.pkl")
        tf_model_folder = LatchDir("s3://latch-public/test-data/3192/unirep_test_data/medium_model_weights").local_path
        validate_pkl_to_model(self, pkl_model, tf_model_folder)

    def test_pkl_to_tf_large(self):
        pkl_model = LatchFile("s3://latch-public/test-data/3192/unirep_test_data/large_model.pkl")
        tf_model_folder = LatchDir("s3://latch-public/test-data/3192/unirep_test_data/large_model_weights").local_path
        validate_pkl_to_model(self, pkl_model, tf_model_folder)
