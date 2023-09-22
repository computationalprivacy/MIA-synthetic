import argparse
import random
import torch
import numpy as np
import datetime
import multiprocessing as mp
from time import time
from tqdm import tqdm

from src.data_prep import read_data, read_metadata, select_columns, discretize_dataset, \
    get_target_record, normalize_cont_cols
from src.generators import get_generator
from src.shadow_data import create_shadow_training_data_membership, create_shadow_training_data_membership_synthetic_only_s3
from src.utils import blockPrint, enablePrint, str2bool, str2list
from src.synthetic_only_parallel import run_pipeline_parallel

# ---------- Data Setup ------------ #
# Read the input of the user
parser = argparse.ArgumentParser()
parser.add_argument("--target_record_id", type = int, default = None,
                    help = "the index of the target record to be considered in the attack")
parser.add_argument("--path_to_data", type = str, default = 'data/2011 Census Microdata Teaching File_OG.csv',
                    help = "path to all original data in csv format")
parser.add_argument("--path_to_metadata", type = str, default = 'data/2011 Census Microdata Teaching Discretized.json',
                    help = "path to metadata of the csv in json format")
parser.add_argument("--cols_to_select", type = str2list, default = "['Age',  'Ethnic Group',  'Residence Type',  'Industry', 'Health',\
 'Hours worked per week' , 'Country of Birth',  'Religion', \
 'Approximated Social Grade',  'Family Composition' , 'Marital Status']",
                    help = "if not all, specify a list of cols to include in the pipeline")
parser.add_argument("--output_path", type = str, default = None, help = "path to save the output in csv format")
parser.add_argument("--name_generator", type = str, default = 'privbayes', help = "name of the synthetic data generator")
parser.add_argument("--epsilon", type = float, default = 1000.0, help = "epsilon value for DP synthetic data generator")
parser.add_argument("--n_aux", type = int, default = 50000, help = "number of records in the auxiliary data")
parser.add_argument("--n_test", type = int, default = 25000, help = "number of records in the test data")
parser.add_argument("--n_original", type = int, default = 1000,
                    help = "number of records in the original data, from which synthetic data is generated")
parser.add_argument("--n_synthetic", type = int, default = 1000,
                    help = "number of records in the generated synthetic dataset")
parser.add_argument("--n_pos_train", type = int, default = 1000,
                    help = "number of shadow datasets with a positive label for training, the total number of train shadow datasets is twice this number")
parser.add_argument("--n_pos_test", type = int, default = 100,
                    help = "number of shadow datasets with a positive label for testing, the total number of test shadow datasets is twice this number")
parser.add_argument("--feature_extractors", type = list, default = ['set_based_classifier'], #['naive', ('query', range(1, 12), 10 ** 8, (1, -1))],
                    help = "a list of specifications for the feature extractor types to be used")
parser.add_argument("--models", type = list, default = ['logistic_regression', 'random_forest'],
                    help = "a list of strings corresponding to the model types to be used")
parser.add_argument("--cv", type = str2bool, default = 'False',
                    help = "whether or not cross validation should be applied")
parser.add_argument("--feat_selection", type = str2bool, default = 'False',
                    help = "whether or not feature selection in the meta model should be applied")
parser.add_argument("--seed", type = int, default = 42,
                    help = "the random seed to be applied for reproducibility")
parser.add_argument("--synthetic_scenario", type = int, default = 1,
                    help = "In which scenario are we ?")
parser.add_argument("--nbr_cores", type = int, default = 1,
                    help = "On how many cores we want to run the attck with Option 3 (Optimal)")
parser.add_argument("--unique", type=str2bool, default='True',
                    help= "If we want to use only a single prediction in the case of the synthetic option")
parser.add_argument("--m", type=int, default=50,
                    help= "The number of time we generate N_SYNTH")



args = parser.parse_args()

PATH_TO_DATA = args.path_to_data
PATH_TO_METADATA = args.path_to_metadata
COLS_TO_SELECT = args.cols_to_select
OUTPUT_PATH = args.output_path
if OUTPUT_PATH is None:
    OUTPUT_PATH = './results_experiments/output_' + str(datetime.datetime.now()) + '.csv'
TARGET_RECORD_ID = args.target_record_id
NAME_GENERATOR = args.name_generator
EPSILON = args.epsilon
N_AUX, N_TEST = args.n_aux, args.n_test
N_ORIGINAL, N_SYNTHETIC  = args.n_original, args.n_synthetic
N_POS_TRAIN, N_POS_TEST = args.n_pos_train, args.n_pos_test
FEATURE_EXTRACTORS = args.feature_extractors
MODELS = args.models
CV = args.cv
FEAT_SELECTION = args.feat_selection
SEED = args.seed
SCENARIO = args.synthetic_scenario
UNIQUE = args.unique
M = args.m


assert SCENARIO in [1,2,3]

# set the seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def main():
    """
    Main function to run our code. This main is call directly when running the .py file. 
    """

    # read data
    meta_data_og, categorical_cols, continuous_cols = read_metadata(PATH_TO_METADATA)
    df = read_data(PATH_TO_DATA, categorical_cols, continuous_cols)
    df = discretize_dataset(df, categorical_cols)
    df = normalize_cont_cols(df, meta_data_og, df_aux=df)
    df, categorical_cols, continuous_cols, meta_data = select_columns(df, categorical_cols, continuous_cols,
                                                                      COLS_TO_SELECT, meta_data_og)

    # get a target record, for now just by selecting an index
    target_record = get_target_record(df, TARGET_RECORD_ID)

    # specify a generator
    generator = get_generator(NAME_GENERATOR, epsilon = EPSILON)

    blockPrint()
    # get all datasets for shadow training, for now only MIA
    #if we are in the synthetic only case (scenario S1, S2 and S3)
    train_seeds = list(range(SEED, SEED + N_POS_TRAIN * 2))
    test_seeds = list(range(SEED + N_POS_TRAIN * 2, SEED + N_POS_TRAIN * 2 + N_POS_TEST * 2)) # make it non overlapping
    test_seeds_s3 = list(range(SEED + N_POS_TRAIN * 2 + N_POS_TEST * 2, SEED + N_POS_TRAIN * 2 + 2*N_POS_TEST * 2)) # make it non overlapping
        
    cols_equal_to_target = (df[df.columns].values == target_record[df.columns].values).sum(axis = 1)
    df_wo_target = df[cols_equal_to_target != len(df.columns)]
    # check if this got rid of at least the target record
    assert len(df) - len(df_wo_target) >= 1
        
    if SCENARIO == 3 :
        #Scenario S3
        datasets_test, datasets_test_s3, labels_test = create_shadow_training_data_membership_synthetic_only_s3(df = df_wo_target, meta_data = meta_data, generator = generator, n_original = N_ORIGINAL,
                                   n_synth = N_SYNTHETIC, n_pos = N_POS_TEST, seeds = test_seeds, target_record = target_record, m = M)
    else :
        #Scenario S1 and S2
        datasets_test, labels_test, _ = create_shadow_training_data_membership(df = df_wo_target, meta_data = meta_data,
                                  target_record = target_record, generator = generator, n_original = N_ORIGINAL,
                                   n_synth = M*N_SYNTHETIC, n_pos = N_POS_TEST, seeds = test_seeds)

    #Argument for the // code
    t1 = time()
    args_list = []
    results_=[]
    for l,dataset_test in enumerate(datasets_test):
        if SCENARIO == 2:
            #In scenario S2, we train a generator on top of the released dataset to perform the pipeline of attack
            dataset_aux  = generator.fit_generate(dataset=dataset_test, metadata=meta_data, size= M*N_SYNTHETIC, seed = SEED + N_POS_TRAIN * 2 + 2*N_POS_TEST * 2 + 42)
        else :
            if SCENARIO == 3:
                #In scenario S3, we use a version without the target of the released dataset
                dataset_aux = datasets_test_s3[l]
            else :
                #In scenario S1, we use directly the released dataset
                dataset_aux = dataset_test

        #disjonction of possibilities, according to arguments:
        #UNIQUE: if we want a unique prediction from the classifier on the test_dataset
        if SCENARIO == 3:
            if UNIQUE:
                args_list.append((l, dataset_aux, target_record, [dataset_test], [labels_test[l]], TARGET_RECORD_ID, FEAT_SELECTION, MODELS, CV, SEED, continuous_cols, categorical_cols, meta_data, generator, N_ORIGINAL,N_SYNTHETIC,N_POS_TRAIN,train_seeds, SCENARIO, NAME_GENERATOR, args))
            else :
                args_list.append((l, dataset_aux, target_record, [dataset_test.sample(n=N_SYNTHETIC, random_state=10*j) for j in range(1,11)], [labels_test[l] for _ in range(10) ], TARGET_RECORD_ID, FEAT_SELECTION, MODELS, CV, SEED, continuous_cols, categorical_cols, meta_data, generator, N_ORIGINAL,N_SYNTHETIC,N_POS_TRAIN,train_seeds, SCENARIO, NAME_GENERATOR, args))
        else:
            if UNIQUE:
                args_list.append((l, dataset_aux, target_record, [dataset_test.head(N_SYNTHETIC)], [labels_test[l]], TARGET_RECORD_ID, FEAT_SELECTION, MODELS, CV, SEED, continuous_cols, categorical_cols, meta_data, generator, N_ORIGINAL,N_SYNTHETIC,N_POS_TRAIN,train_seeds, SCENARIO, NAME_GENERATOR, args))
            else :
                args_list.append((l, dataset_aux, target_record, [dataset_test.sample(n=N_SYNTHETIC, random_state=10*j) for j in range(1,11)], [labels_test[l] for _ in range(10) ], TARGET_RECORD_ID, FEAT_SELECTION, MODELS, CV, SEED, continuous_cols, categorical_cols, meta_data, generator, N_ORIGINAL,N_SYNTHETIC,N_POS_TRAIN,train_seeds, SCENARIO, NAME_GENERATOR, args))

    results = []
    #we perform the experiment by batch of size nbr_cores
    for batch_start in range(0, len(args_list), args.nbr_cores):
        batch_end = min(batch_start + args.nbr_cores, len(args_list))
        args_list_batch = args_list[batch_start:batch_end]
        with mp.Pool(args.nbr_cores) as pool:
            for result in tqdm(pool.imap(run_pipeline_parallel, args_list_batch),total=len(args_list_batch)):
                results.append(result)
    enablePrint()
    if len(results)==len(datasets_test):
        print('success!')
    t2 = time()
    print(f'Execution time : {t2-t1}')
    return 0

if __name__ == "__main__":
    main()