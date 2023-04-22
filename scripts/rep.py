#!/usr/bin/env python3
if __name__ == "__main__":
    import sys
    import tensorflow as tf
    import numpy as np
    import os
    import subprocess

    # Run using "conda run -n unirep {script_path} {model_size.value} {local_dir} seqs.csv {model_path}"
    MODEL_SIZE = int(
        sys.argv[1]
    )  # if 1 (True) use 1900 dimensional model, else use 64 dimensional one.
    OUTPUT_DIR = sys.argv[2]
    SEQS_PATH = sys.argv[3]
    MODEL_WEIGHT_PATH = sys.argv[4]

    # Read seqs csv from SEQS_PATH into a list of pairs
    seqs = []
    with open(SEQS_PATH, "r") as f:
        for line in f:
            seqs.append(line.strip().split(","))

    os.chdir("/root")

    # Set seeds
    tf.set_random_seed(42)
    np.random.seed(42)

    # Get models weights
    if MODEL_WEIGHT_PATH == "None":
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

    if MODEL_SIZE == 64:
        from unirep_source.unirep import babbler64 as babbler
    elif MODEL_SIZE == 256:
        from unirep_source.unirep import babbler256 as babbler
    elif MODEL_SIZE == 1900:
        from unirep_source.unirep import babbler1900 as babbler
    else:
        print("Invalid model size")
        exit(1)

    # Set up model
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

    # Get the reps
    for seq, name in seqs:
        if b.is_valid_seq(seq):
            # Get the representation of the sequence
            avg_hidden, final_hidden, final_cell = b.get_rep(seq)

            # Write avg_hidden to unirep.npy
            np.save(os.path.join(OUTPUT_DIR, f"{name}_unirep"), avg_hidden)

            # Write avg_hidden, final_hidden, final_cell to unirep_fusion.npy
            np.save(
                os.path.join(OUTPUT_DIR, f"{name}_unirep_fusion"),
                np.stack((avg_hidden, final_hidden, final_cell)),
            )
