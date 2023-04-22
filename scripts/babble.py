#!/usr/bin/env python3
from pathlib import Path
import pickle
import os
import numpy as np
from tempfile import TemporaryDirectory, mkdtemp


def pkl_to_model(pkl_path):
    # Unpickle model
    with open(pkl_path, "rb") as f:
        model = pickle.load(f)

        # Get the true model size from fully_connected_weights
        model_size = model[-2][0].shape[0]

        # root = Path(pkl_path).parent
        root = Path(mkdtemp())
        model_path = root / f"{model_size}_weights"
        os.mkdir(model_path)

        # Save embed matrix, fully connected biases, weights
        # Embed matrix is always model[0], fcb, fcw are model[-2][1] and model[-2][0] respectively
        np.save(model_path / "embed_matrix:0.npy", model[0])
        np.save(model_path / "fully_connected_biases:0.npy", model[-2][1])
        np.save(model_path / "fully_connected_weights:0.npy", model[-2][0])

        params = ["wx", "wh", "wmx", "wmh", "b", "gx", "gh", "gmx", "gmh"]
        if int(model_size) in [64, 256]:
            # # will be replaced by the stack number, and N will be replaced by the parameter name
            file_skeleton = "rnn_mlstm_stack_mlstm_stack#_mlstm_stack#_N:0.npy"
            for n, label in [(1, 0), (3, 1), (5, 2), (7, 3)]:
                for p in params:
                    file_name = model_path / file_skeleton.replace(
                        "#", str(label)
                    ).replace("N", p)
                    np.save(file_name, model[n][p])
        else:
            file_skeleton = "rnn_mlstm_mlstm_N:0.npy"
            for p in params:
                file_name = model_path / file_skeleton.replace("N", p)
                np.save(file_name, model[1][p])
    return model_path


if __name__ == "__main__":
    import sys
    import tensorflow as tf
    import numpy as np
    import os
    import subprocess

    # Call: conda run -n unirep {script_path} {model_size.value} {local_dir} {length} {temp} seqs.csv {model_path}\
    MODEL_SIZE = int(
        sys.argv[1]
    )  # if 1 (True) use 1900 dimensional model, else use 64 dimensional one.
    OUTPUT_DIR = sys.argv[2]
    LENGTH = int(sys.argv[3])
    TEMP = float(sys.argv[4])
    SEQS_PATH = sys.argv[5]
    MODEL_WEIGHT_PATH = sys.argv[6]
    # Read seqs csv from SEQS_PATH into a list of pairs
    seqs = []
    with open(SEQS_PATH, "r") as f:
        for line in f:
            seqs.append(line.strip().split(","))

    os.chdir("/root")

    # Set seeds
    tf.set_random_seed(42)
    np.random.seed(42)

    if MODEL_WEIGHT_PATH == "None":
        # Get models weights
        subprocess.run(
            [
                "aws",
                "s3",
                "sync",
                "--no-sign-request",
                "--quiet",
                f"s3://unirep-public/{MODEL_SIZE}_weights/",
                f"{MODEL_SIZE}_weights/",
            ]
        )
        MODEL_WEIGHT_PATH = f"./{MODEL_SIZE}_weights"
        # print(f"Got {MODEL_SIZE}_weights")

    if MODEL_SIZE == 64:
        from unirep_source.unirep import babbler64 as babbler
    elif MODEL_SIZE == 256:
        from unirep_source.unirep import babbler256 as babbler
    elif MODEL_SIZE == 1900:
        from unirep_source.unirep import babbler1900 as babbler
    else:
        print("Invalid model size")
        exit(1)

    batch_size = 12
    try:
        b = babbler(batch_size=batch_size, model_path=MODEL_WEIGHT_PATH)
    except Exception as e:
        print(e)
        print(MODEL_WEIGHT_PATH.split("/")[-1], MODEL_SIZE)
        if MODEL_WEIGHT_PATH.split("/")[-1] != f"{MODEL_SIZE}_weights":
            print(
                "Good chance that the model weights you uploaded were for the wrong model size. Please try again."
            )
        exit(1)

    def check_valid_and_get_babble(seq_and_name):
        if b.is_valid_seq(seq_and_name[0]):
            seq_and_name.append(b.get_babble(seq_and_name[0], LENGTH, TEMP))
        else:
            seq_and_name.append("invalid sequence")
        return seq_and_name

    # Get Outputs: [seq, name, babble]
    outputs = []
    for seq in seqs:
        outputs.append(check_valid_and_get_babble(seq))

    # Write results to csv file with headers 'name', 'seq', 'babble'
    # Only add name, seq, babble if it is the file does not exist yet
    babble_outputs_path = os.path.join(OUTPUT_DIR, "babble_results.csv")
    if not os.path.exists(babble_outputs_path):
        with open(babble_outputs_path, "w") as f:
            f.write("name,seq,babble\n")
    with open(os.path.join(OUTPUT_DIR, "babble_results.csv"), "a") as f:
        for output in outputs:
            if output[2] is not None:
                f.write(output[1] + "," + output[0] + "," + output[2] + "\n")
            else:
                f.write(output[1] + "," + output[0] + "," + "None" + "\n")

    # Write results to 'babble.txt' in OUTPUT_DIR/name
    # If babble.txt already exists, append to it
    for output in outputs:
        if not os.path.exists(os.path.join(OUTPUT_DIR, output[1])):
            os.mkdir(os.path.join(OUTPUT_DIR, output[1]))
        with open(os.path.join(OUTPUT_DIR, output[1], f"babble{LENGTH}.txt"), "a") as f:
            f.write(output[2])

        # Write original_seq.txt in OUTPUT_DIR/output[1]
        with open(os.path.join(OUTPUT_DIR, output[1], "original_seq.txt"), "w") as f:
            f.write(output[0])

    # print('saved babbles')
