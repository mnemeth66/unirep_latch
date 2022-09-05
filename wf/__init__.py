from enum import Enum
from pathlib import Path
import os
from Bio import SeqIO
import jax_unirep
import numpy as np
from datetime import date
import subprocess
import hashlib
import pickle as pkl

from latch import large_task, medium_task, small_task, workflow, create_conditional_section
from latch.types import LatchDir, LatchFile
from latch.resources.launch_plan import LaunchPlan
from typing import Optional, List, Union, Tuple

# Allow extended amino acids: https://en.wikipedia.org/wiki/FASTA_format#Sequence_representation 
aas = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
        'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
        'W', 'X', 'Y', 'Z', '*', '-']

class Application(Enum):
    protein_rep = "UniRep/UniRep Fusion"
    babble = "Babble"
    evotune = "Evotune"
    # variant_prediction = "Variant Fitness Prediction"

class ModelSize(Enum):
    small = "64"
    medium = "256"
    large = "1900"

@small_task
def check_enum(
    application: Application,
) -> Tuple[bool, bool, bool]:
    return tuple([application == a for a in Application])

@small_task
def get_seqs_from_inputs(
    sequence: Optional[List[Union[str, LatchFile, LatchDir]]],
) -> List[List[str]]:
    """
    Return a list of tuple (sequence, sequence_name) from the assorted str/LatchFile inputs.
    If the input is just a string then it is assigned a truncated hash of its contents.
    """
    seqs = []
    latchfile_paths = []
    # for seq in sequence, print string if str or print local_path if latchfile
    for seq in sequence:
        try:
            if isinstance(seq, str):
                print('found a string input')
                # Make sure the sequence is valid
                if not set(seq.upper()).issubset(set(aas)):
                    raise ValueError(f"Invalid sequence: {seq}")

                # Give unnamed sequence a hash
                seq_name = hashlib.sha256(seq.encode('utf-8')).hexdigest()[:10]
                seqs.append([seq, seq_name]) 

            elif isinstance(seq, LatchFile):
                print('found a latchfile input')
                latchfile_paths.append(Path(seq).resolve())
            
            elif isinstance(seq, LatchDir):
                print('found a LatchDir input')
                local_path = Path(seq).resolve()
                fasta_files = [f.resolve() for f in local_path.iterdir() if f.suffix == '.fasta']
                latchfile_paths.extend(fasta_files)
        except Exception as e:
            print(e)
            raise e

    print(f"unrolling {len(latchfile_paths)} latchfiles")
    for latchfile_path in latchfile_paths:
        output_filename = latchfile_path.stem
        for record in SeqIO.parse(latchfile_path, "fasta"):
            seqs.append([str(record.seq), output_filename + "_" + str(record.id)])

    return seqs

@small_task
def get_holdouts(
    sequence: Optional[List[Union[str, LatchFile, LatchDir]]],
) -> Optional[List[str]]:
    """
    A wrapper around get_seqs_from_inputs to account for holdout sequences being optional,
    and working with Type.Optional.
    If no sequences are returned, return None rather than an empty list.
    """
    if sequence is not None:
        seqs = get_seqs_from_inputs(sequence)
        if len(seqs) > 0:
            return seqs
    return None
    

@small_task
def evotune_task(
    seqs_and_names: List[List[str]],
    model_size: ModelSize,
    model_params: Optional[LatchDir],
    run_name: str,
    holdouts: Optional[List[str]],
) -> LatchDir:

    local_dir = Path("/root/outputs/")
    local_dir.mkdir(exist_ok=True)
    local_dir = str(local_dir)
    remote_dir = "latch:///unirep/" + run_name + "/"
    
    model_size = int(model_size.value)
    if model_size == 64:
        from jax_unirep.evotuning_models import mlstm64 as mlstm
    elif model_size == 256:
        from jax_unirep.evotuning_models import mlstm256 as mlstm
    elif model_size == 1900:
        from jax_unirep.evotuning_models import mlstm1900 as mlstm
    else:
        raise ValueError(f"Invalid model size: {model_size.value}")
    
    # Get parameters
    init_func, model_func = mlstm()
    if model_params is not None:
        params = jax_unirep.utils.load_params(folderpath=model_params.local_path)
    else:
        params = jax_unirep.utils.load_params(paper_weights=model_size) 

    # Evotuning
    _, evotuned_params = jax_unirep.evotune(
        sequences=[s[0] for s in seqs_and_names],
        model_func=model_func,
        params=params,
        n_splits=min(len(seqs_and_names), 5),
        out_dom_seqs=holdouts,
    )
    
    # Save the evotuned parameters
    jax_unirep.utils.dump_params(evotuned_params, local_dir)
    return LatchDir(local_dir, remote_dir)

@small_task
def rep_task(
    seqs_and_names: List[List[str]],
    model_size: ModelSize,
    model_params: Optional[LatchDir],
    run_name: str,
) -> LatchDir:

    local_dir = Path("/root/outputs/")
    local_dir.mkdir(exist_ok=True)
    local_dir = str(local_dir)
    remote_dir = "latch:///unirep/" + run_name + "/"

    # Get the reps
    ## Load the model params
    params = None
    if model_params is not None:
        params = jax_unirep.utils.load_params(folderpath=model_params.local_path)
    mlstm_size = int(model_size.value)
    seqs = [seq_and_name[0] for seq_and_name in seqs_and_names]
    h_avg, h_final, c_final = jax_unirep.get_reps(seqs, params=params, mlstm_size=mlstm_size)

    # Save the reps
    for i in range(len(seqs_and_names)):
        seq, name = seqs_and_names[i][0], seqs_and_names[i][1]
        # Make folder in local_dir for each sequence
        local_seq_dir = os.path.join(local_dir, name)
        if not os.path.exists(local_seq_dir):
            os.mkdir(local_seq_dir)
        
        # Save seq as 'original_seq.txt' in local_seq_dir
        with open(os.path.join(local_seq_dir, "original_seq.txt"), "w") as f:
            f.write(seq)

        # Write avg_hidden to output_name + '_unirep'
        np.save(os.path.join(local_seq_dir + '/unirep'), h_avg[i])

        # Write avg_hidden, final_hidden, final_cell to output_name + '_unirep_fusion
        np.save(os.path.join(local_seq_dir + '/unirep_fusion'), np.stack((h_avg[i], h_final[i], c_final[i])))
        
        print('saved files')
    # Save h_avg as unireps.npy in local_dir
    np.save(os.path.join(local_dir, "unireps"), h_avg)

    # Return LatchDir
    return LatchDir(local_dir, remote_dir)

@small_task
def babble_task(
    seqs_and_names: List[List[str]],
    model_size: ModelSize,
    model_params: Optional[LatchDir],
    run_name: str,
    length: Optional[int],
    temp: Optional[float],
) -> LatchDir:

    local_dir = Path("/root/outputs/")
    local_dir.mkdir(exist_ok=True)
    local_dir = str(local_dir)
    remote_dir = "latch:///unirep/" + run_name + "/"

    # Write seqs to 'seqs.csv' file
    with open("seqs.csv", "w") as f:
        for seq_and_name in seqs_and_names:
            f.write(f"{seq_and_name[0]},{seq_and_name[1]}\n")

    # Run babble
    script_path = "scripts/babble.py"
    model_path = "None"
    if model_params is not None:
        model_path = model_params.local_path
    subprocess.run(f"conda run -n unirep {script_path} {model_size.value} {local_dir} {length} {temp} seqs.csv {model_path}".split(), check=True)
    return LatchDir(local_dir, remote_dir)

@workflow
def unirep(
    sequence: Optional[List[Union[str, LatchFile, LatchDir]]],
    application: Application,
    run_name: str = str(date.today()),
    model_size: ModelSize = ModelSize.small,
    model_params: Optional[LatchDir] = None,
    length: Optional[int] = int(250),
    temp: Optional[float] = 1.0,
    holdout: Optional[List[Union[str, LatchFile, LatchDir]]] = None,
    ) -> LatchDir:

    """
    UniRep
    ----
    # UniRep
    UniRep is an a mLSTM "babbler" deep representation learner for protein engineering informatics.
    UniRep out of the box has support for the following applications:
    - "unirep": generating protein representations from the mLSTM model. This includes both UniRep, a dense representation of the protein, and UniRep_fusion, a larger but more complete representation of the protein.
    - "babbling": using generative modeling to synthesize sequences from a seed

    This workflow also benefits from the work done by Eric J. Ma and Arkadij Kummer to reproduce and extend Unirep
    using Jax. From their work we use the following features:
    - "evotuning": further tuning the UniRep model on user-provided sequences 
    - [future] "sampling": sampling from seed sequences using user-defined scoring functions
  
    # This Workflow 
    This workflow gives the user access to the following applications:
    - generating protein representations from the mLSTM model
    - "babbling": using generative modeling to synthesize sequences from a seed
    - "evotuning": further tuning the UniRep model on user-provided sequences

    ## Inputs
    - `run_name`: Name of the run. The files will be stored in latch:///unirep/{run_name}/. If None, run_name will default to the current date and time.
    - `sequence`: Multiple strings or FASTA files containing a protein sequence.
    - `model_size`: Choice between using the 64, 256, or 1900-dimensional model.
    - `model_params`: If not None, this is a LatchDir containing the model parameters in a model_weights.pkl file. If None, the model parameters will be downloaded from the cloud.
    - `application`: A dropdown indicating which application to use.
        - `UniRep/UniRep Fusion`: Generate a protein representation.
        - `Babble`: Synthesize sequences from a seed.
        - `Evotuning`: Further tune the UniRep model on user-provided sequences.
    - `length`: (Default 250) An integer indicating the length of the sequence to generate (including the original protein length). 
    - `temperature`: (Default 1) A float between 0 and 1 indicating how noisy the babble should be. 1 is the noisiest.
    - `holdout`: (Optional) Strings/LatchFiles containing holdout sequences for Evotuning.

    ## Outputs
    - `unirep/{run_name}/{protein_name}/unirep.np`: A numpy array containing the UniRep representation of the protein.
    - `unirep/{run_name}/{protein_name}/unirep_fusion.np`: A numpy array containing the UniRep Fusion representation of the protein.
    - `unirep/{run_name}/{protein_name}/babble{LENGTH}.txt`: A text file containing the babble from a seed protein.
    - `unirep/{run_name}/{protein_name}/original_seq.txt`: A text file containing the original protein sequence.
    - `unirep/{run_name}/babble_results.csv`: A csv containing aggregated babble results.
    - `unirep/{run_name}/model_params.pkl`: A pickle file containing the model parameters.

    ## License
    #### UniRep
    Copyright 2018, 2019 Ethan Alley, Grigory Khimulya, Surojit Biswas

    All the model weights are licensed under the terms of Creative Commons Attribution-NonCommercial 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

    Otherwise the code in this repository is licensed under the terms of [GPL v3](https://www.gnu.org/licenses/gpl-3.0.html) as specified by the gpl.txt file.

    #### Jax-UniRep
    All the model weights are licensed under the terms of Creative Commons Attribution-NonCommercial 4.0 International License. To view a copy of this license, visit here) or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

    Otherwise the code in this repository is licensed under the terms of GPL v3.

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
            A string, fasta file, or directory containing fasta files containing protein sequences.
            _metadata_:
                display_name: Sequence
        model_size:
            What sized model to use.
            __metadata__:
                display_name: Model Size
        model_params:
            LatchDir containing the model_weights.pkl file. Use if you evotuned a new model.
            __metadata__:
                display_name: (Optional) Model Parameters
        application:
            What application of the UniRep model to use.
            __metadata__:
                display_name: Application
        run_name:
            Name of the run where the files will be stored.
            __metadata__:
                display_name: Run Name
        length:
            Length of the sequence to generate (including original protein length). Default: 250
            __metadata__:
                display_name: (Babble) Length
        temp:
            How noisy the babble should be. Default: 1
            __metadata__:
                display_name: (Babble) Temperature
        holdout:
            Holdout sequences for evotuning.
            __metadata__:
                display_name: (Evotuning) Holdout
        
    """
    seqs_and_names = get_seqs_from_inputs(sequence=sequence)
    (rep, babble, evotune) = check_enum(application=application)
    holdouts = get_holdouts(sequence=holdout)
    return (
        create_conditional_section("application")
        .if_((rep.is_true())).then(
            rep_task(
                seqs_and_names=seqs_and_names,
                model_size=model_size,
                model_params=model_params,
                run_name=run_name,
                ))
        .elif_((babble.is_true())).then(
            babble_task(
                seqs_and_names=seqs_and_names,
                model_size=model_size,
                model_params=model_params,
                run_name=run_name,
                length=length,
                temp=temp,
                ))
        .elif_((evotune.is_true())).then(
            evotune_task(
                seqs_and_names=seqs_and_names,
                model_size=model_size,
                model_params=model_params,
                run_name=run_name,
                holdouts=holdouts,
                ))
        .else_().fail("Variant Prediction isn't implemented yet.")
    )

def main():
    unirep(
        sequence=['GGVA', 'GGVB', 'GGVH'],
        application=Application.protein_rep,
        length=5,
        run_name='dubbadoo',
    )
# if __name__ == "__main__":
#     unirep(
#         sequence='GGVA',
#         application=Application.babble,
#         length=10,
#         run_name='yodeleyheehoo',
#     )

# LaunchPlan(
#     unirep,
#     "Sample UniRep",
#     {
#         "sequence":[LatchFile("s3://latch-public/test-data/3192/data/cas9.fasta")],
#         "application":Application.protein_rep,
#         "length":5,
#     },
# )

# LaunchPlan(
#     unirep,
#     "Sample Babble",
#     {
#         "sequence":['LATCH'],
#         "application":Application.babble,
#         "length":10,
#     },
# )