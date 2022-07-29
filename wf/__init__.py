from subprocess import run, Popen, PIPE, CalledProcessError
from enum import Enum
from pathlib import Path
import os
import sys
from Bio import SeqIO

from latch import large_task, medium_task, small_task, workflow, create_conditional_section
from latch.types import LatchDir, LatchFile
from typing import Optional, List, Union, Tuple


class Application(Enum):
    protein_rep = "UniRep/UniRep Fusion"
    variant_prediction = "Variant Fitness Prediction"
    babble = "Babble"

@small_task
def check_enum(
    application: Application,
) -> Tuple[bool, bool]:
    return application == Application.protein_rep, application == Application.babble

@small_task
def get_seq_from_latchfile(
    sequence: Union[str, LatchFile],
) -> Tuple[str, str]:
    if isinstance(sequence, LatchFile):
        local_path = Path(sequence).resolve()
        output_filename = local_path.stem
        for record in SeqIO.parse(sequence, "fasta"):
            return str(record.seq), output_filename
    else:
        return sequence, "protein"

@small_task
def rep_task(
    sequence: str,
    output_filename: str,
    use_full_1900_dim_model: bool,
    output_dir: Optional[LatchDir],
) -> LatchDir:
    local_dir = "/root/outputs/"
    if not os.path.exists(local_dir):
        os.mkdir(local_dir)
    remote_dir = output_dir.remote_directory if output_dir else "latch:///unirep/"
    path = "scripts/get_rep.py"
    os.system(f"conda run -n unirep {path} {use_full_1900_dim_model} {local_dir} {sequence} {output_filename}")
    return LatchDir(local_dir, remote_dir)

@small_task
def babble_task(
    sequence: str,
    output_filename: str,
    use_full_1900_dim_model: bool,
    output_dir: Optional[LatchDir],
    length: Optional[int],
    temp: Optional[float],
) -> LatchDir:
    local_dir = "/root/outputs/"
    if not os.path.exists(local_dir):
        os.mkdir(local_dir)
    remote_dir = output_dir.remote_directory if output_dir else "latch:///unirep/"
    path = "scripts/babble.py"
    os.system(f"conda run -n unirep {path} {use_full_1900_dim_model} {local_dir} {sequence} {output_filename} {length} {temp}")
    return LatchDir(local_dir, remote_dir)

@workflow
def unirep(
    sequence: Union[str, LatchFile],
    application: Application,
    output_dir: Optional[LatchDir],
    use_full_1900_dim_model: bool = False,
    length: Optional[int] = int(250),
    temp: Optional[float] = 1.0,
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
    - "babbling": using generative modeling to synthesize sequences from a seed
    - [FUTURE] variant prediction using UniRep representations
    - [FUTURE, all] multiple sequences at once

    ## Inputs
    - `sequence`: A string or FASTA file containing a protein sequence.
    - `use_full_1900_dim_model`: A boolean indicating whether to use the 1900 dimensional model or the 64 dimensional model.
    - `application`: A dropdown indicating which application to use.
        - `UniRep/UniRep Fusion`: Generate a protein representation.
        - `Babble`: Synthesize sequences from a seed.
        - `Variant Fitness Prediction`: Predict variant fitness.
    - `length`: (Default 250) An integer indicating the length of the sequence to generate (including the original protein length). 
    - `temperature`: (Default 1) A float between 0 and 1 indicating how noisy the babble should be. 1 is the noisiest.

    ## Outputs
    - `unirep/{protein_name}_unirep.np`: A numpy array containing the UniRep representation of the protein.
    - `unirep/{protein_name}_unirep_fusion.np`: A numpy array containing the UniRep Fusion representation of the protein.
    - `unirep/{protein_name}_babble.txt`: A text file containing the babble from a seed protein.

    ## License
    Copyright 2018, 2019 Ethan Alley, Grigory Khimulya, Surojit Biswas

    All the model weights are licensed under the terms of Creative Commons Attribution-NonCommercial 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

    Otherwise the code in this repository is licensed under the terms of [GPL v3](https://www.gnu.org/licenses/gpl-3.0.html) as specified by the gpl.txt file.

    ## Contact
    If you have any questions about the workflow, contact mnemeth6@berkeley.edu.
    For questions about the models, contact the authors.

    __metadata__:
        display_name: UniRep
        author:
            name: Church Lab
            email: 
            github:
        repository: https://github.com/churchlab/UniRep
        license:
            id: GPLv3
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
        output_dir:
            A LatchDir indicating the directory to save the output files to.
            __metadata__:
                display_name: (Optional) Output Directory
        length:
            Length of the sequence to generate (including original protein length). Default: 250
            __metadata__:
                display_name: (Babble) Length
        temp:
            How noisy the babble should be. Default: 1
            __metadata__:
                display_name: (Babble) Temperature
        
    """
    (seq, filename) = get_seq_from_latchfile(sequence=sequence)
    (rep, babble) = check_enum(application=application)
    return (
        create_conditional_section("application")
        .if_((rep.is_true())).then(
            rep_task(
                sequence=seq,
                output_filename=filename,
                use_full_1900_dim_model=use_full_1900_dim_model,
                output_dir=output_dir,
                ))
        .elif_((babble.is_true())).then(
            babble_task(
                sequence=seq,
                output_filename=filename,
                use_full_1900_dim_model=use_full_1900_dim_model,
                output_dir=output_dir,
                length=length,
                temp=temp,
                ))
        .else_().fail("Variant Prediction isn't implemented yet.")
    )

if __name__ == "__main__":
    unirep(
        sequence='GGVA',
        application=Application.protein_rep,
        output_dir=LatchDir("outputs")
    )