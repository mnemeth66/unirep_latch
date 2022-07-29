#!/usr/bin/env python3
if __name__ == "__main__":
    import sys
    import tensorflow as tf
    import numpy as np
    import os
    import subprocess

    USE_FULL_1900_DIM_MODEL = sys.argv[1] # if 1 (True) use 1900 dimensional model, else use 64 dimensional one.
    OUTPUT_DIR = sys.argv[2]
    SEQ = sys.argv[3]
    OUTPUT_NAME = sys.argv[4]
    LENGTH = int(sys.argv[5])
    TEMP = float(sys.argv[6])

    os.chdir("/root")

    # Set seeds
    tf.set_random_seed(42)
    np.random.seed(42)

    if USE_FULL_1900_DIM_MODEL:
        # Sync relevant weight files
        subprocess.run(["aws", "s3", "sync", "--no-sign-request","--quiet", "s3://unirep-public/1900_weights/", "1900_weights/"])

        # Import the mLSTM babbler model
        from unirep_source.unirep import babbler1900 as babbler

        # Where model weights are stored.
        MODEL_WEIGHT_PATH = "./1900_weights"

    else:
        # Sync relevant weight files
        subprocess.run(["aws", "s3", "sync", "--no-sign-request","--quiet", "s3://unirep-public/64_weights/", "64_weights/"])

        # Import the mLSTM babbler model
        from unirep_source.unirep import babbler64 as babbler

        # Where model weights are stored.
        MODEL_WEIGHT_PATH = "./64_weights"

    print('got weights')
    batch_size = 12
    b = babbler(batch_size=batch_size, model_path=MODEL_WEIGHT_PATH)

    assert b.is_valid_seq(SEQ), "The sequence is not valid; either it is too long (<2000 aa's) or has an illegal aa"
    print('checked if sequence is valid')

    # Get babble
    babbled_seq = b.get_babble(SEQ, LENGTH, TEMP)

    # Save babble to output_name + '_babble.txt' in output_dir
    path = os.path.join(OUTPUT_DIR, OUTPUT_NAME + '_babble.txt')
    print(path)
    with open(path, 'w') as f:
        f.write(babbled_seq)
    print('saved babble')