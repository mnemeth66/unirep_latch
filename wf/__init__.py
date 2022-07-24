from subprocess import run, Popen, PIPE, CalledProcessError
from enum import Enum
from pathlib import Path
import os
import sys
# print(os.system("ls ."))
# print(os.system("ls /root"))
print(os.system("$PATH"))
print(os.system("conda env list"))
from Bio import SeqIO

from latch import large_task, medium_task, small_task, workflow
from latch.types import LatchDir, LatchFile
from typing import Optional, List, Union


class Application(Enum):
    protein_rep = "UniRep/UniRep Fusion"
    variant_prediction = "Variant Fitness Prediction"
    babble = "Babble"


@small_task
def get_seq_from_latchfile(
    sequence: Union[str, LatchFile],
) -> (str, str):
    if isinstance(sequence, LatchFile):
        local_path = Path(sequence).resolve()
        output_filename = local_path.stem
        for record in SeqIO.parse(sequence, "fasta"):
            return str(record.seq), output_filename
    else:
        return sequence, "protein"

@small_task
def unirep_task(
    sequence: str,
    output_filename: str,
    use_full_1900_dim_model: bool,
    ) -> LatchDir:
    local_dir = "/root/outputs/"
    if not os.path.exists(local_dir):
        os.mkdir(local_dir)
    remote_dir = "latch:///unirep/"
    path = "/root/scripts/var_predict.py"
#     USE_FULL_1900_DIM_MODEL = sys.argv[1] # if 1 (True) use 1900 dimensional model, else use 64 dimensional one.
    # OUTPUT_DIR = sys.argv[2]
    # SEQ = sys.argv[3]
    # OUTPUT_NAME = sys.argv[4]
    # os.system("conda run -n unirep scripts/var_predict.py")
    os.system(f"conda run -n unirep scripts/get_rep.py {use_full_1900_dim_model} {local_dir} {sequence} {output_filename}")

    return LatchDir(local_dir, remote_dir)

@workflow
def unirep(
    sequence: Union[str, LatchFile],
    use_full_1900_dim_model: bool = False,
    application: Application = Application.protein_rep,
    ) -> LatchDir:

    """
    UniRep
    ----
    # UniRep
    UniRep is an a mLSTM "babbler" deep representation learner for protein engineering informatics.
    UniRep out of the box has support for the following applications:
    - generating protein representations from the mLSTM model. This includes both UniRep, a dense representation of the protein, and UniRep_fusion, a larger but more complete representation of the protein.
    - "babbling": using generative modeling to synthesize sequences from a seed
    - further tuning the UniRep model on user-provided sequences 

    # This Workflow 
    This workflow gives the user access to the following applications:
    - generating protein representations from the mLSTM model
    - [IN PROGRESS] "babbling": using generative modeling to synthesize sequences from a seed
    - [FUTURE] variant prediction using UniRep representations

    ## Inputs
    - `sequence`: A string or LatchFile containing a protein sequence.
    - `use_full_1900_dim_model`: A boolean indicating whether to use the 1900 dimensional model or the 64 dimensional model.
    - `application`: A string indicating which application to use.
        - `UniRep/UniRep Fusion`: Generate a protein representation.
        - `Variant Fitness Prediction`: Predict variant fitness.
        - `Babble`: Synthesize sequences from a seed.

    ## Outputs
    - `unirep/{protein_name}_unirep.np`: A numpy array containing the UniRep representation of the protein.
    - `unirep/{protein_name}_unirep_fusion.np`: A numpy array containing the UniRep Fusion representation of the protein.

    ## License
    Copyright 2018, 2019 Ethan Alley, Grigory Khimulya, Surojit Biswas

    All the model weights are licensed under the terms of Creative Commons Attribution-NonCommercial 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

    Otherwise the code in this repository is licensed under the terms of [GPL v3](https://www.gnu.org/licenses/gpl-3.0.html) as specified by the gpl.txt file.
    Args:
        sequence:
            A string or LatchFile containing a protein sequence.
            _metadata_:
                display_name: Sequence
        use_full_1900_dim_model:
            Use 1900-unit model rather than base 64 one
            __metadata__:
                display_name: Use largest model
        application:
            What application of the UniRep model to use.
            __metadata__:
                display_name: Application
        
    """
    (seq, filename) = get_seq_from_latchfile(sequence=sequence)
    return unirep_task(
        sequence=seq,
        output_filename=filename,
        use_full_1900_dim_model=use_full_1900_dim_model
        )
    # return t1()

if __name__ == "__main__":
    unirep(
        sequence='GGVA',
    )