#!/bin/bash

python Achilles_main.py --path_to_data='./data/2011 Census Microdata Teaching File_OG.csv' \
                --path_to_metadata='./data/2011 Census Microdata Teaching Discretized.json' \
                --target_record_id=44444 \
                --output_dir='./results_experiments' \
                --name_generator='SYNTHPOP' \
                --n_aux=50000 \
                --n_test=25000 \
                --n_original=1000 \
                --n_synthetic=1000 \
                --n_pos_train=500 \
                --n_pos_test=100