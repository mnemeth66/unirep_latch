# Imports
from latch.types import LatchFile, LatchDir
from latch import workflow
from typing import Optional, List, Union, Tuple
from pathlib import Path
import sys, os
import unittest
from subprocess import CalledProcessError

sys.path.append('../wf')
from wf import babble_task, Application, ModelSize


cas9 = "MKRNYILGLDIGITSVGYGIIDYETRDVIDAGVRLFKEANVENNEGRRSKRGARRLKRRRRHRIQRVKKLLFDYNLLTDHSELSGINPYEARVKGLSQKLSEEEFSAALLHLAKRRGVHNVNEVEEDTGNELSTKEQISRNSKALEEKYVAELQLERLKKDGEVRGSINRFKTSDYVKEAKQLLKVQKAYHQLDQSFIDTYIDLLETRRTYYEGPGEGSPFGWKDIKEWYEMLMGHCTYFPEELRSVKYAYNADLYNALNDLNNLVITRDENEKLEYYEKFQIIENVFKQKKKPTLKQIAKEILVNEEDIKGYRVTSTGKPEFTNLKVYHDIKDITARKEIIENAELLDQIAKILTIYQSSEDIQEELTNLNSELTQEEIEQISNLKGYTGTHNLSLKAINLILDELWHTNDNQIAIFNRLKLVPKKVDLSQQKEIPTTLVDDFILSPVVKRSFIQSIKVINAIIKKYGLPNDIIIELAREKNSKDAQKMINEMQKRNRQTNERIEEIIRTTGKENAKYLIEKIKLHDMQEGKCLYSLEAIPLEDLLNNPFNYEVDHIIPRSVSFDNSFNNKVLVKQEENSKKGNRTPFQYLSSSDSKISYETFKKHILNLAKGKGRISKTKKEYLLEERDINRFSVQKDFINRNLVDTRYATRGLMNLLRSYFRVNNLDVKVKSINGGFTSFLRRKWKFKKERNKGYKHHAEDALIIANADFIFKEWKKLDKAKKVMENQMFEEKQAESMPEIETEQEYKEIFITPHQIKHIKDFKDYKYSHRVDKKPNRELINDTLYSTRKDDKGNTLIVNNLNGLYDKDNDKLKKLINKSPEKLLMYHHDPQTYQKLKLIMEQYGDEKNPLYKYYEETGNYLTKYSKKDNGPVIKKIKYYGNKLNAHLDITDDYPNSRNKVVKLSLKPYRFDVYLDNGVYKFVTVKNLDVIKKENYYEVNSKCYEEAKKLKKISNQAEFIASFYNNDLIKINGELYRVIGVNNDLLNRIEVNMIDITYREYLENMNDKRPPRIIKTIASKTQSIKKYSTDILGNLYEVKSKKHPQIIKKG"

test_babble = babble_task.__wrapped__

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
        # Check the outputs, confirm that the correct files were created
        local_dir = Path("/root/outputs/")
        files = os.listdir(local_dir)
        self.assertEqual(files, ['babble_results.csv', 'seqs_protein1'])

        # confirm the babble length is correct by processing the last value in the csv
        with open(local_dir / 'babble_results.csv', 'r') as f:
            lines = f.readlines()
            babble_line = lines[1]
            babble = babble_line.split(',')[-1].strip()
            self.assertEqual(len(babble), 10)
            self.assertEqual(babble[:len(protein)], protein)

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
        # Check the outputs, confirm that the correct files were created
        local_dir = Path("/root/outputs/")
        files = os.listdir(local_dir)

        # confirm the babble length is correct by processing the last value in the csv
        with open(local_dir / 'babble_results.csv', 'r') as f:
            lines = f.readlines()
            babble_line = lines[-1]
            babble = babble_line.split(',')[-1].strip()
            self.assertEqual(len(babble), len(protein))

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
        # Check the outputs, confirm that the correct files were created
        local_dir = Path("/root/outputs/")
        files = os.listdir(local_dir)

        # confirm the babble length is correct by processing the last value in the csv
        with open(local_dir / 'babble_results.csv', 'r') as f:
            lines = f.readlines()
            babble_line = lines[-1]
            babble = babble_line.split(',')[-1].strip()
            self.assertEqual(len(babble), len(protein))

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
        # Check the outputs, confirm that the correct files were created
        local_dir = Path("/root/outputs/")
        files = os.listdir(local_dir)

        # confirm an invalid output because of invalid amino acids
        with open(local_dir / 'babble_results.csv', 'r') as f:
            lines = f.readlines()
            babble_line = lines[-1]
            babble = babble_line.split(',')[-1].strip()
            self.assertEqual(babble, 'invalid sequence')

    def test_model_param_input(self):
        protein = 'LATCH'
        test_babble(
                seqs_and_names=[[protein, 'seqs_protein1']],
                model_size=ModelSize.small,
                model_params=LatchDir("/root/test_scripts/test_data/test_small_model/"),
                run_name="custom model test",
                length=10,
                temp=1,
        )
        # Check the outputs, confirm that the correct files were created
        local_dir = Path("/root/outputs/")
        files = os.listdir(local_dir)

        # confirm an invalid output because of invalid amino acids
        with open(local_dir / 'babble_results.csv', 'r') as f:
            lines = f.readlines()
            babble_line = lines[-1]
            babble = babble_line.split(',')[-1].strip()
            self.assertEqual(len(babble), 10)

    def test_model_mismatched_size(self):
        protein = 'LATCH'
        with self.assertRaises(CalledProcessError) as cm:
            test_babble(
                    seqs_and_names=[[protein, 'seqs_protein1']],
                    model_size=ModelSize.large,
                    model_params=LatchDir("/root/test_scripts/test_data/test_model/"),
                    run_name="custom model test",
                    length=10,
                    temp=1,
            )
        # print(cm)

if __name__ == '__main__':
    unittest.main()
