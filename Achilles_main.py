import argparse
import random
import torch
import numpy as np
import datetime

from src.data_prep import read_data, read_metadata, select_columns, discretize_dataset, \
    normalize_cont_cols, sample_split_data_for_attack, get_target_record
from src.generators import get_generator
from src.shadow_data import create_shadow_training_data_membership
from src.set_based_classifier import fit_set_based_classifier
from src.feature_extractors import fit_ohe, get_feature_extractors, apply_feature_extractor
from src.classifiers import drop_zero_cols, scale_features, select_features, fit_validate_classifiers
from src.output import save_output
from src.utility import compute_utility
from src.utils import enablePrint, blockPrint, str2bool, str2list, ignore_depreciation

# ---------- Data Setup ------------ #
# Read the input of the user
        
parser = argparse.ArgumentParser()
parser.add_argument("--target_record_id", type = int, default = None,
                    help = "the index of the target record to be considered in the attack")
parser.add_argument("--path_to_data", type = str,
                    ## FOR ADULT
                    #default = 'data/Adult_dataset.csv',
                    ## For UK census
                    default = 'data/2011 Census Microdata Teaching File_OG.csv',
                    help = "path to all original data in csv format")
parser.add_argument("--path_to_metadata", type = str,
                    ## FOR ADULT
                    #default = 'data/Adult_metadata_discretized.json',
                    ## For UK census
                    default = 'data/2011 Census Microdata Teaching Discretized.json',
                    help = "path to metadata of the csv in json format")
parser.add_argument("--cols_to_select", type = str2list, default = "['all']",
                    help = "if not all, specify a list of cols to include in the pipeline")
parser.add_argument("--output_dir", type = str, default = './results_experiments',
                    help = "path to dir to save the output in csv format")
parser.add_argument("--name_generator", type = str, default = 'privbayes', help = "name of the synthetic data generator")
parser.add_argument("--epsilon", type = float, default = 1000.0, help = "epsilon value for DP synthetic data generator")
parser.add_argument("--n_aux", type = int, default = 50000, help = "number of records in the auxiliary data")
parser.add_argument("--n_test", type = int, default = 25000, help = "number of records in the test data")
parser.add_argument("--n_original", type = int, default = 1000,
                    help = "number of records in the original data, from which synthetic data is generated")
parser.add_argument("--n_synthetic", type = int, default = 1000,
                    help = "number of records in the generated synthetic dataset")
parser.add_argument("--n_pos_train", type = int, default = 500,
                    help = "number of shadow datasets with a positive label for training, the total number of train shadow datasets is twice this number")
parser.add_argument("--n_pos_test", type = int, default = 100,
                    help = "number of shadow datasets with a positive label for testing, the total number of test shadow datasets is twice this number")
parser.add_argument("--models", type = list, default = ['random_forest'],
                    help = "a list of strings corresponding to the model types to be used")
parser.add_argument("--cv", type = str2bool, default = 'False',
                    help = "whether or not cross validation should be applied")
parser.add_argument("--feat_selection", type = str2bool, default = 'False',
                    help = "whether or not feature selection in the meta model should be applied")
parser.add_argument("--compute_utility", type = str2bool, default = 'False',
                    help = "Compute the utility of the given generator")
parser.add_argument("--seed", type = int, default = 42,
                    help = "the random seed to be applied for reproducibility")

args = parser.parse_args()

PATH_TO_DATA = args.path_to_data
PATH_TO_METADATA = args.path_to_metadata
COLS_TO_SELECT = args.cols_to_select
OUTPUT_DIR = args.output_dir
TARGET_RECORD_ID = args.target_record_id
NAME_GENERATOR = args.name_generator
EPSILON = args.epsilon
N_AUX, N_TEST = args.n_aux, args.n_test
N_ORIGINAL, N_SYNTHETIC  = args.n_original, args.n_synthetic
N_POS_TRAIN, N_POS_TEST = args.n_pos_train, args.n_pos_test
MODELS = args.models
CV = args.cv
FEAT_SELECTION = args.feat_selection
COMPUTE_UTILITY = args.compute_utility
SEED = args.seed

# set the seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def main():

    # read data
    print('Reading and preparing the data...')
    meta_data_og, categorical_cols, continuous_cols = read_metadata(PATH_TO_METADATA)
    df = read_data(PATH_TO_DATA, categorical_cols, continuous_cols)
    df = discretize_dataset(df, categorical_cols)
    df = normalize_cont_cols(df, meta_data_og, df_aux=df)
    df, categorical_cols, continuous_cols, meta_data = select_columns(df, categorical_cols, continuous_cols,
                                                                      COLS_TO_SELECT, meta_data_og)

    # get a target record, for now just by selecting an index
    target_record = get_target_record(df, TARGET_RECORD_ID)

    # split data into auxiliary and test set
    df_aux, df_test = sample_split_data_for_attack(df, target_record, N_AUX, N_TEST)

    # specify a generator
    generator = get_generator(NAME_GENERATOR, epsilon = EPSILON)

    print('Creating shadow datasets...')
    blockPrint()
    # get all datasets for shadow training, for now only MIA
    train_seeds = list(range(N_POS_TRAIN * 2))
    datasets_train, labels_train, datasets_train_utility = create_shadow_training_data_membership(df = df_aux, meta_data = meta_data,
                                    target_record = target_record, generator = generator, n_original = N_ORIGINAL,
                                    n_synth = N_SYNTHETIC, n_pos = N_POS_TRAIN, seeds = train_seeds)
    test_seeds = list(range(N_POS_TRAIN * 2, N_POS_TRAIN * 2 + N_POS_TEST * 2)) # make it non overlapping
    datasets_test, labels_test, _ = create_shadow_training_data_membership(df = df_test, meta_data = meta_data,
                                  target_record = target_record, generator = generator, n_original = N_ORIGINAL,
                                   n_synth = N_SYNTHETIC, n_pos = N_POS_TEST, seeds = test_seeds)
    enablePrint()

    # fit one hot encoding
    ohe, ohe_column_names = fit_ohe(df_aux, categorical_cols, meta_data)

    # compute utility
    if COMPUTE_UTILITY:
        print('Computing utility...')
        # only consider X=2*10 datasets to compute the utility of
        train_utility = compute_utility(datasets=datasets_train_utility[:10], dataset_path=PATH_TO_DATA,
                                        cont_cols=continuous_cols, cat_cols=categorical_cols,
                                        ohe=ohe, ohe_column_names=ohe_column_names)
        utility_results = {'Train':train_utility}
    else:
        utility_results = None

    ### (1) first do the naive feature extraction
    print('Running naive attack with simple features...')
    NAIVE_FEATURE_EXTRACTORS = ['naive']
    feature_extractors, do_ohe = get_feature_extractors(NAIVE_FEATURE_EXTRACTORS)
    X_train, y_train = apply_feature_extractor(datasets = datasets_train.copy(), target_record = target_record,
                                       labels = labels_train,
                                       ohe = ohe, ohe_columns = categorical_cols, ohe_column_names = ohe_column_names,
                                       continuous_cols=continuous_cols,
                                       feature_extractors = feature_extractors, do_ohe = do_ohe)
    X_test, y_test = apply_feature_extractor(datasets = datasets_test.copy(), target_record = target_record,
                                       labels = labels_test,
                                       ohe = ohe, ohe_columns = categorical_cols, ohe_column_names = ohe_column_names,
                                       continuous_cols=continuous_cols,
                                       feature_extractors = feature_extractors, do_ohe = do_ohe)
    X_train, X_test = drop_zero_cols(X_train, X_test)
    X_train, X_test = scale_features(X_train, X_test)

    if FEAT_SELECTION:
       valid_cols = select_features(X_train, y_train)
       X_train, X_test = X_train[valid_cols], X_test[valid_cols]

    trained_models, all_results = fit_validate_classifiers(X_train, y_train, X_test, y_test, models = MODELS, cv = CV)

    NAIVE_OUTPUT_PATH = f'{OUTPUT_DIR}/output_{datetime.datetime.now()}_naive.csv'
    save_output(NAIVE_OUTPUT_PATH, vars(args), all_results, utility_results)

    ### (2) Then do the query based extraction
    print('Running query-based attack...')
    QUERY_FEATURE_EXTRACTORS = [('query', range(1, df.shape[1] + 1), 1e6, {'categorical':(1,), 'continuous': (3,)})]

    feature_extractors, do_ohe = get_feature_extractors(QUERY_FEATURE_EXTRACTORS)
    ignore_depreciation()
    X_train, y_train = apply_feature_extractor(datasets = datasets_train.copy(), target_record = target_record,
                                        labels = labels_train,
                                        ohe = ohe, ohe_columns = categorical_cols, ohe_column_names = ohe_column_names,
                                        continuous_cols=continuous_cols,
                                        feature_extractors = feature_extractors, do_ohe = do_ohe)
    X_test, y_test = apply_feature_extractor(datasets = datasets_test.copy(), target_record = target_record,
                                        labels = labels_test,
                                        ohe = ohe, ohe_columns = categorical_cols, ohe_column_names = ohe_column_names,
                                        continuous_cols=continuous_cols,
                                        feature_extractors = feature_extractors, do_ohe = do_ohe)

    X_train, X_test = drop_zero_cols(X_train, X_test)
    X_train, X_test = scale_features(X_train, X_test)

    if FEAT_SELECTION:
        valid_cols = select_features(X_train, y_train)
        X_train, X_test = X_train[valid_cols], X_test[valid_cols]

    trained_models, all_results = fit_validate_classifiers(X_train, y_train, X_test, y_test, models = MODELS, cv = CV)

    QUERY_OUTPUT_PATH = f'{OUTPUT_DIR}/output_{datetime.datetime.now()}_query.csv'
    save_output(QUERY_OUTPUT_PATH, vars(args), all_results, utility_results)

    ### (3) Then do set based classifier with TARGETattention
    print('Running target attention attack...')
    TARGET_ATTENTION_MODEL_PARAMS = {'num_in_features':len(ohe_column_names) + len(continuous_cols) + 2,
                                     'embedding_hidden_size':10, 'embedding_size':10,
                                    'attention_size': 10, 'n_records': 1000,
                                    'prediction_hidden_size':5, 'dropout_rate':0.15}

    trained_models, all_results = fit_set_based_classifier(all_datasets_train=datasets_train.copy(), datasets_test=datasets_test.copy(),
                                            all_labels_train=labels_train, labels_test=labels_test,
                                            model_params=TARGET_ATTENTION_MODEL_PARAMS,
                                            target_record=target_record,
                                            ohe=ohe, categorical_cols=categorical_cols, ohe_column_names=ohe_column_names,
                                            continuous_cols=continuous_cols, meta_data=meta_data, df_aux=df_aux,
                                            path_best_model=f'{OUTPUT_DIR}/best_model_{TARGET_RECORD_ID}_targetattention.pt',
                                            model_type='TargetAttention', top_X=100)

    TARGETATTENTION_OUTPUT_PATH = f'{OUTPUT_DIR}/output_{datetime.datetime.now()}_targetattention.csv'
    save_output(TARGETATTENTION_OUTPUT_PATH, vars(args), all_results, utility_results)
    
    print('succes!')

if __name__ == "__main__":
    main()
