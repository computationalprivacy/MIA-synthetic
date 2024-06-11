#!/bin/bash

python synthetic_only_main.py --path_to_data='./data/2011 Census Microdata Teaching File_OG.csv' \
                --path_to_metadata='./data/2011 Census Microdata Teaching Discretized.json' \
                    --target_record_id=266579\
                    --synthetic_scenario=3\
                    --n_original=1000\
                    --n_pos_test=5\
                    --n_pos_train=10\
                    --nbr_cores=20\
                    --unique='True'\
                    --name_generator='SYNTHPOP'\
                    --cols_to_select="['all']"
