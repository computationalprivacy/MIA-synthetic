#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=20:mem=200gb

module load anaconda3/personal
source activate featuresnout

cd /rds/general/user/frg20/home/featuresnout/Git/featuresnout/
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