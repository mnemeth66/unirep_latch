#!/usr/bin/env python3
if __name__ == "__main__":
    import sys
    import tensorflow as tf
    import numpy as np
    import os
    import subprocess

    MODEL_SIZE = int(sys.argv[1]) # if 1 (True) use 1900 dimensional model, else use 64 dimensional one.
    OUTPUT_DIR = sys.argv[2]
    LENGTH = int(sys.argv[3])
    TEMP = float(sys.argv[4])
    SEQS_PATH = sys.argv[5]
    MODEL_WEIGHT_PATH = sys.argv[6]
    # Read seqs csv from SEQS_PATH into a list of pairs
    seqs = []
    with open(SEQS_PATH, 'r') as f:
        for line in f:
            seqs.append(line.strip().split(','))

    os.chdir("/root")

    # Set seeds
    tf.set_random_seed(42)
    np.random.seed(42)

    if MODEL_WEIGHT_PATH == "None":
        # Get models weights
        subprocess.run(["aws", "s3", "sync", "--no-sign-request","--quiet", f"s3://unirep-public/{MODEL_SIZE}_weights/", f"{MODEL_SIZE}_weights/"])
        MODEL_WEIGHT_PATH = f"./{MODEL_SIZE}_weights"
        print(f"Got {MODEL_SIZE}_weights")

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
        if MODEL_WEIGHT_PATH != f"./{MODEL_SIZE}_weights":
            print("Good chance that the model weights you uploaded were for the wrong model size. Please try again.")
        exit(1)

    def check_valid_and_get_babble(seq_and_name):
        if b.is_valid_seq(seq_and_name[0]):
            seq_and_name.append(b.get_babble(seq_and_name[0], LENGTH, TEMP))
        else:
            seq_and_name.append('invalid sequence')
        return seq_and_name

    # Get Outputs: [seq, name, babble]
    outputs = []
    for seq in seqs:
        outputs.append(check_valid_and_get_babble(seq))
    
    # Write results to csv file with headers 'name', 'seq', 'babble'
    with open(os.path.join(OUTPUT_DIR, 'babble_results.csv'), 'a') as f:
        f.write('name, seq ,babble\n')
        for output in outputs:
            if output[2] is not None:
                f.write(output[1] + ',' + output[0] + ',' + output[2] + '\n')
            else:
                f.write(output[1] + ',' + output[0] + ',' + 'None' + '\n')

    # Write results to 'babble.txt' in OUTPUT_DIR/name
    # If babble.txt already exists, append to it
    for output in outputs:
        if not os.path.exists(os.path.join(OUTPUT_DIR, output[1])):
            os.mkdir(os.path.join(OUTPUT_DIR, output[1]))
        with open(os.path.join(OUTPUT_DIR, output[1], f'babble{LENGTH}.txt'), 'a') as f:
            f.write(output[2])      
        
        # Write original_seq.txt in OUTPUT_DIR/output[1]
        with open(os.path.join(OUTPUT_DIR, output[1], 'original_seq.txt'), 'w') as f:
            f.write(output[0])

    print('saved babbles')