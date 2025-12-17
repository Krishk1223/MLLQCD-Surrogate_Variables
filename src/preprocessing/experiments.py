import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys
import argparse

"""
Preprocessing step 4: Experiment configuration and data generation for ML tasks on 2 point 
                      and 3 point correlators.
"""

# DEFAULT EXPERIMENT CONFIGURATIONS FOR 2 POINT CORRELATOR PREDICTION TASKS:

CONFIG = {
    "time_sources" : 4,
}

TWO_PT_EXPERIMENT_ONE = {
        'type' : '2pt',
        'name': '2 point correlator OTSP: Gold to Non-Gold Operator Prediction',
        'description' : 'Cheap symmetric operators used to predict expensive symmetry breaking operators',
        'input filepath': 'data/processed/unaveraged_data/',
        'num features': 1,
        'input filename': '2pt_D_gold_fine.csv',
        'num targets': 1,
        'target filename': '2pt_D_nongold_fine.csv',
        'output path': 'data/experiments/Two_pt_ML_OTSP_D_gold_to_nongold_Experiment/',
        'training feature file': 'train_data_X.npy',
        'training target file': 'train_data_y.npy',
        'evaluation feature file': 'evaluation_data_X.npy',
        'evaluation target file': 'evaluation_data_y.npy',
        'bias correction feature file': 'bias_correction_data_X.npy',
        'bias correction target file': 'bias_correction_data_y.npy',
        'test feature file': 'test_data_X.npy',
        'test target file': 'test_data_y.npy' 
}

TWO_PT_EXPERIMENT_TWO = {
        'type' : '2pt',
        'name' : '2 point correlator Kaon Momentum Transfer Interpolation Prediction: K qmax to qsq0',
        'description' : 'Predicting Kaon correlators at q^2=0 using correlators at highest momentum',
        'input filepath': 'data/processed/unaveraged_data/',
        'num features': 1,
        'input filename': '2pt_K_fine.csv',
        'num targets': 1,
        'target filename': '2pt_K_fine_qsq0.csv',
        'output path': 'data/experiments/Two_pt_ML_Kaon_qmax_to_qsq0_Experiment/',
        'training feature file': 'train_data_X.npy',
        'training target file': 'train_data_y.npy',
        'evaluation feature file': 'evaluation_data_X.npy',
        'evaluation target file': 'evaluation_data_y.npy',
        'bias correction feature file': 'bias_correction_data_X.npy',
        'bias correction target file': 'bias_correction_data_y.npy',
        'test feature file': 'test_data_X.npy',
        'test target file': 'test_data_y.npy'
}

TWO_PT_EXPERIMENT_THREE = {
        'type' : '2pt',
        'name' : '2 point correlator Kaon Momentum Transfer Interpolation Prediction: K qmax to qsqmaxby3',
        'description' : 'Predicting Kaon correlators at q^2=q^2_max/3 using correlators at highest momentum',
        'input filepath': 'data/processed/unaveraged_data/',
        'num features': 1,
        'input filename': '2pt_K_fine.csv',
        'num targets': 1,
        'target filename': '2pt_K_fine_qsqmaxby3.csv',
        'output path': 'data/experiments/Two_pt_ML_Kaon_qmax_to_qsqmaxby3_Experiment/',
        'training feature file': 'train_data_X.npy',
        'training target file': 'train_data_y.npy',
        'evaluation feature file': 'evaluation_data_X.npy',
        'evaluation target file': 'evaluation_data_y.npy',
        'bias correction feature file': 'bias_correction_data_X.npy',
        'bias correction target file': 'bias_correction_data_y.npy',
        'test feature file': 'test_data_X.npy',
        'test target file': 'test_data_y.npy'
}

TWO_PT_EXPERIMENT_FOUR = {
        'type' : '2pt',
        'name' : '2 point correlator Kaon Momentum Transfer Interpolation Prediction: K qmax to 2qsqmaxby3',
        'description' : 'Predicting Kaon correlators at q^2=2*q^2_max/3 using correlators at highest momentum',
        'input filepath': 'data/processed/unaveraged_data/',
        'num features': 1,
        'input filename': '2pt_K_fine.csv',
        'num targets': 1,
        'target filename': '2pt_K_fine_2qsqmaxby3.csv',
        'output path': 'data/experiments/Two_pt_ML_Kaon_qmax_to_qsq2maxby3_Experiment/',
        'training feature file': 'train_data_X.npy',
        'training target file': 'train_data_y.npy',
        'evaluation feature file': 'evaluation_data_X.npy',
        'evaluation target file': 'evaluation_data_y.npy',
        'bias correction feature file': 'bias_correction_data_X.npy',
        'bias correction target file': 'bias_correction_data_y.npy',
        'test feature file': 'test_data_X.npy',
        'test target file': 'test_data_y.npy'
}

TWO_PT_EXPERIMENT_FIVE = {
        'type' : '2pt',
        'name' : '2 point correlator Cross Channel Extrapolation Prediction: D_gold and D_nongold to K qmax',
        'description' : 'Using D meson correlators to predict Kaon correlators at highest momentum',
        'input filepath': 'data/processed/unaveraged_data/',
        'num features': 2,
        'input filename': ['2pt_D_gold_fine.csv', '2pt_D_nongold_fine.csv'],
        'num targets': 1,
        'target filename': '2pt_K_fine.csv',
        'output path': 'data/experiments/Two_pt_ML_Cross_Channel_StationaryExtrapolation_Experiment/',
        'training feature file': 'train_data_X.npy',
        'training target file': 'train_data_y.npy',
        'evaluation feature file': 'evaluation_data_X.npy',
        'evaluation target file': 'evaluation_data_y.npy',
        'bias correction feature file': 'bias_correction_data_X.npy',
        'bias correction target file': 'bias_correction_data_y.npy',
        'test feature file': 'test_data_X.npy',
        'test target file': 'test_data_y.npy'
}
# DEFAULT EXPERIMENT CONFIGURATIONS FOR 3 POINT CORRELATOR PREDICTION TASKS:
# To be implemented

experiments = {
    1: TWO_PT_EXPERIMENT_ONE,
    2: TWO_PT_EXPERIMENT_TWO,
    3: TWO_PT_EXPERIMENT_THREE,
    4: TWO_PT_EXPERIMENT_FOUR,
    5: TWO_PT_EXPERIMENT_FIVE
}

def valid_file(file: Path):
    """Checks if a file exists and is a valid CSV file."""
    if file.exists() and file.suffix == '.csv':
        return True
    else:
        return False

def missing_data(df, strategy='remove'):
    """
    Checks for missing data in dataframes of a given experiment. Handles via strategy.
    strategy: 'remove' to drop rows with missing data, 'warn' to warn user of missing data.
    """
    #Nan checks:
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f"Warning: Dataframe contains {nan_count} missing values.")
        if strategy == 'remove':
            df = df.dropna()
            print(f"Dropped configs with missing values. New dataframe shape: {df.shape}")
        elif strategy == 'warn':
            print("Please handle missing data before proceeding.")
            sys.exit(1)
    else: 
        print("No missing data found in dataframe.")
    return df
    
def add_experiment(experiments_dict=experiments):
    """
    Adds a new experiment to the existing experiments dictionary.
    Probably will not be used right now so leave it be as it will not be invoked yet.
    """
    #type of experiment:
    type_int = 0
    while type_int not in [2,3]:
        try:
            type_int = int(input("Enter experiment type (2 for 2pt, 3 for 3pt): "))
        except ValueError:
            print("Invalid input. Please enter 2 or 3.")
            type_int = 0
    if type_int == 2:
        type = '2pt'
    elif type_int == 3:
        type = '3pt'

    #experiment name and description:
    name = str(input("Enter experiment name: "))
    description = input("Enter experiment description: ")

    #file path and filenames checking:
    while True:
        input_filepath = input("Enter input filepath: ")
        project_root = Path(__file__).resolve().parent.parent.parent
        input_path = project_root / input_filepath
        if not input_path.exists():
            print(f"Input path {input_path} does not exist. Please enter a valid path.")
        else:
            break
    
    num_files = int(input("Enter number of input files: "))
    if num_files > 1:
        input_filenames = []
        for i in range(num_files):
            file_flag = True
            while file_flag:
                filename = input(f"Enter input filename {i+1}: ")
                filepath = project_root / input_filepath / filename
                if valid_file(filepath):
                    file_flag = False
                else:
                    print(f"Either the file {filepath} does not exist or is not a valid CSV file. Please enter a valid filename.")
            input_filenames.append(filename)
    else:
        file_flag = True
        while file_flag:
            filename = input("Enter input filename: ")
            file = project_root / input_filepath / filename
            if valid_file(file):
                file_flag = False
            else:
                print(f"Either the file {file} does not exist or is not a valid CSV file. Please enter a valid filename.")
        input_filenames = filename
    
    target_flag = True
    while target_flag:
        target_filename = input("Enter target filename: ")
        target_file = project_root / input_filepath / target_filename
        if valid_file(target_file):
            target_flag = False
        else:
            print(f"Either the file {target_file} does not exist or is not a valid CSV file. Please enter a valid filename.")
    output_filepath = input("Enter output path: ")
    print("Output filenames will be set to default names to ensure consistency.")
    experiment_number = max(experiments_dict.keys()) + 1
    experiment = {
        'type' : type,
        'name': name,
        'description' : description,
        'input filepath': input_filepath,
        'num features': num_files,
        'input filename': input_filenames,
        'target filename': target_filename,
        'num targets': 1,
        'output path': output_filepath,
        'training feature file': 'train_data_X.npy',
        'training target file': 'train_data_y.npy',
        'evaluation feature file': 'evaluation_data_X.npy',
        'evaluation target file': 'evaluation_data_y.npy',
        'bias correction feature file': 'bias_correction_data_X.npy',
        'bias correction target file': 'bias_correction_data_y.npy',
        'test feature file': 'test_data_X.npy',
        'test target file': 'test_data_y.npy'
    }
    experiments_dict[experiment_number] = experiment
    print(f"Experiment {experiment_number} added successfully.")
    return experiments_dict

def two_point_extract_identifier(filename):
    """Extract the 2 point identifier from the filename"""
    file_stem = Path(filename).stem
    identifier = file_stem.replace("2pt_", '').replace('_fine', '')
    return identifier

def two_point_data_for_experiment(experiment_number, tau_max=None, experiment_dict=experiments, strategy='remove'):
    """Generates the data for a given experiment."""
    experiment = experiment_dict.get(experiment_number)
    if experiment is None:
        raise ValueError(f"Experiment {experiment_number} not defined.")
    
    #filepath setup for experiment:
    project_root = Path(__file__).resolve().parent.parent.parent
    input_path = project_root / experiment['input filepath']
    output_path = project_root / experiment['output path']
    output_path.mkdir(exist_ok=True, parents=True)

    time_sources = CONFIG['time_sources']

    #Loading input data:
    if isinstance(experiment['input filename'], list):
        input_dfs = []
        #Multi input file case:
        for filename in experiment['input filename']:
            input_file = input_path / filename
            if not input_file.exists():
                raise FileNotFoundError(f"Input file {input_file} does not exist.")
            df = pd.read_csv(input_file, header=None)
            input_dfs.append(df)
        input_data = pd.concat(input_dfs, axis=1, ignore_index=True)
    else: #Single input case:
        input_file = input_path / experiment['input filename']
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        input_data = pd.read_csv(input_file, header=None)

    #loading target data:
    target_file = input_path / experiment['target filename']
    if not target_file.exists():
        raise FileNotFoundError(f"Target file {target_file} does not exist.")
    target_data = pd.read_csv(target_file, header=None)
    
    min_rows = min(len(input_data), len(target_data))
    aligned_rows = (min_rows // time_sources) * time_sources
    
    print(f"Input rows: {len(input_data)}, Target rows: {len(target_data)}")
    print(f"Aligning to {aligned_rows} rows ({aligned_rows // time_sources} complete configs)")
    
    input_data = input_data.iloc[:aligned_rows].reset_index(drop=True)
    target_data = target_data.iloc[:aligned_rows].reset_index(drop=True)
    
    config_ids = np.repeat(np.arange(1, aligned_rows // time_sources + 1), time_sources)
    time_source_ids = np.tile(np.arange(time_sources), aligned_rows // time_sources)
    
    multi_index = pd.MultiIndex.from_arrays([config_ids, time_source_ids], names=['config_id', 'time_source'])
    
    input_data.index = multi_index
    target_data.index = multi_index
    
    input_cols_names = []
    if isinstance(experiment['input filename'], list):
        for filename in experiment['input filename']:
            identifier = two_point_extract_identifier(filename)
            df_temp = pd.read_csv(input_path / filename, header=None, nrows=1)
            n_cols = df_temp.shape[1]
            for i in range(1, n_cols + 1):
                input_cols_names.append(f"input_{identifier}_tau_{i}")
    else:
        identifier = two_point_extract_identifier(experiment['input filename'])
        n_cols = input_data.shape[1]
        input_cols_names = [f"input_{identifier}_tau_{i}" for i in range(1, n_cols + 1)]
    
    input_data.columns = input_cols_names
    
    identifier = two_point_extract_identifier(experiment['target filename'])
    n_cols = target_data.shape[1]
    target_data.columns = [f"target_{identifier}_tau_{i}" for i in range(1, n_cols + 1)]
    
    paired_data = input_data.join(target_data, how='inner')
    
    config_counts = paired_data.groupby(level='config_id').size()
    complete_configs = config_counts[config_counts == time_sources].index
    paired_data = paired_data[paired_data.index.get_level_values('config_id').isin(complete_configs)]
    
    # Truncate to tau_max if specified:
    if tau_max is not None:
        input_cols = [col for col in paired_data.columns if col.startswith('input_')][:tau_max]
        target_cols = [col for col in paired_data.columns if col.startswith('target_')][:tau_max]
        paired_data = paired_data[input_cols + target_cols]

    paired_data = missing_data(paired_data, strategy=strategy)
    return paired_data

def three_point_extract_identifier(filename):
    """Extract the 3 point identifier from the filename"""
    file_stem = Path(filename).stem
    #to be improved:
    identifier = file_stem.replace("3pt_", '').replace('_fine', '').replace('_averaged', '')
    return identifier

def three_point_experiment_data(experiment_number, tau_max=None,experiment_dict=experiments, strategy='remove'):
    """Generates the data for a given 3 point correlator experiment."""
    experiment = experiment_dict.get(experiment_number)
    if experiment is None:
        raise ValueError(f"Experiment {experiment_number} not defined.")
    
    #filepath setup for experiment:
    project_root = Path(__file__).resolve().parent.parent.parent
    input_path = project_root / experiment['input filepath']
    output_path = project_root / experiment['output path']
    output_path.mkdir(exist_ok=True, parents=True)

    #to be continued...
    raise NotImplementedError("3 point correlator data generation not yet implemented. Please standby.")

def split_data(combined_data:pd.DataFrame, train_fraction=0.15, eval_fraction=0.15, bias_correction_fraction=0.15, test_fraction=0.55, seed=42, strategy='remove'):
    """
    Splits the combined dataset into its train, eval, bias correction and test fraction.
    Ensures all time sources of a config stay together in the same split.
    """

    combined_data = missing_data(combined_data, strategy=strategy)
    
    #normalisation check:
    total_fraction = train_fraction + eval_fraction + bias_correction_fraction + test_fraction
    if not np.isclose(total_fraction, 1.0):
        train_fraction /= total_fraction
        eval_fraction /= total_fraction
        bias_correction_fraction /= total_fraction
        test_fraction /= total_fraction

    unique_configs = combined_data.index.get_level_values('config_id').unique()
    n_configs = len(unique_configs)
    
    rng = np.random.default_rng(seed=seed)
    config_idx = np.arange(n_configs)
    rng.shuffle(config_idx)

    n_train_configs = int(train_fraction * n_configs)
    n_eval_configs = int(eval_fraction * n_configs)
    n_bias_correction_configs = int(bias_correction_fraction * n_configs)
    n_test_configs = n_configs - (n_train_configs + n_eval_configs + n_bias_correction_configs)
    
    #config ids for each split:
    train_config_ids = unique_configs[config_idx[:n_train_configs]]
    eval_config_ids = unique_configs[config_idx[n_train_configs:n_train_configs + n_eval_configs]]
    bias_correction_config_ids = unique_configs[config_idx[n_train_configs + n_eval_configs:n_train_configs + n_eval_configs + n_bias_correction_configs]]
    test_config_ids = unique_configs[config_idx[n_train_configs + n_eval_configs + n_bias_correction_configs:]]

    input_cols = [col for col in combined_data.columns if col.startswith("input_")]
    target_cols = [col for col in combined_data.columns if col.startswith("target_")]
    
    train_data = combined_data[combined_data.index.get_level_values('config_id').isin(train_config_ids)][input_cols + target_cols]
    eval_data = combined_data[combined_data.index.get_level_values('config_id').isin(eval_config_ids)][input_cols + target_cols]
    bias_correction_data = combined_data[combined_data.index.get_level_values('config_id').isin(bias_correction_config_ids)][input_cols + target_cols]
    test_data = combined_data[combined_data.index.get_level_values('config_id').isin(test_config_ids)][input_cols + target_cols]

    return train_data, eval_data, bias_correction_data, test_data

def save_split(df: pd.DataFrame, features_path: Path, target_path: Path):
    """Saves the split dataframes into numpy files."""
    input_cols = [col for col in df.columns if col.startswith("input_")]
    target_cols = [col for col in df.columns if col.startswith("target_")]
    X = df[input_cols].to_numpy()
    y = df[target_cols].to_numpy()
    np.save(features_path, X)
    np.save(target_path, y)

def validate_experiment_files(experiment_number, tau_max=None, experiment_dict=experiments, strategy='remove'):
    """ Validates data for given experiment and checks for consistency
    if all files are valid, returns the combined dataframe by calling 
    two_point_data_for_experiment if the files are not valid it an raises error."""
    experiment = experiment_dict.get(experiment_number)
    if experiment is None:
        raise ValueError(f"Experiment {experiment_number} not defined.")
    project_root = Path(__file__).resolve().parent.parent.parent
    input_path = project_root / experiment['input filepath']

    #input file checks:
    if isinstance(experiment['input filename'], list):
        true_list = []
        for filename in experiment['input filename']:
            input_file = input_path / filename
            if valid_file(input_file):
                print(f"Input file {input_file} is valid.")
                true_list.append(True)
            else:
                print(f"Input file {input_file} is invalid.")
                true_list.append(False)
        if all(true_list):
            input_valid = True
    else:
        input_file = input_path / experiment['input filename']
        if valid_file(input_file):
            print(f"Input file {input_file} is valid.")
            input_valid = True
        else:
            print(f"Input file {input_file} is invalid.")
            input_valid = False
    
    target_file = input_path / experiment['target filename']
    if valid_file(target_file):
        print(f"Target file {target_file} is valid.")
        target_valid = True
    else:
        print(f"Target file {target_file} is invalid.")
        target_valid = False
    
    if (input_valid and target_valid):
        if experiment['type'] == '2pt':
            return two_point_data_for_experiment(experiment_number, tau_max=tau_max, experiment_dict=experiment_dict, strategy=strategy)
        elif experiment['type'] == '3pt':
            return three_point_experiment_data(experiment_number, tau_max=tau_max, experiment_dict=experiment_dict, strategy=strategy)
        else:
            raise ValueError(f"Experiment type {experiment['type']} not recognized.")
    else:
        raise FileNotFoundError("One or more input/target files are invalid.")
        return False

def write_experiment_metadata(experiment_number, experiment_dict=experiments):
    """Writes the experiment metadata to a JSON file in output directory."""
    experiment = experiment_dict.get(experiment_number)
    if experiment is None:
        raise ValueError(f"Experiment {experiment_number} not defined.")
    
    project_root = Path(__file__).resolve().parent.parent.parent
    output_path = project_root / experiment['output path']
    output_path.mkdir(exist_ok=True, parents=True)

    metadata = {
        'experiment_number': experiment_number,
        'name': experiment['name'],
        'description': experiment['description'],
        'input filepath': experiment['input filepath'],
        'num features': experiment['num features'],
        'input filename': experiment['input filename'],
        'num targets': experiment['num targets'],
        'target filename': experiment['target filename'],
        'output path': experiment['output path'],
        'training feature file': experiment['training feature file'],
        'training target file': experiment['training target file'],
        'evaluation feature file': experiment['evaluation feature file'],
        'evaluation target file': experiment['evaluation target file'],
        'bias correction feature file': experiment['bias correction feature file'],
        'bias correction target file': experiment['bias correction target file'],
        'test feature file': experiment['test feature file'],
        'test target file': experiment['test target file']
    }

    metadata_file = output_path / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Experiment metadata written to {metadata_file}")

def check_experiment_output_exists(experiment_number, experiment_dict=experiments):
    """Checks if the output files for a given experiment already exist."""
    experiment = experiment_dict.get(experiment_number)
    if experiment is None:
        raise ValueError(f"Experiment {experiment_number} not defined.")
    
    project_root = Path(__file__).resolve().parent.parent.parent
    output_path = project_root / experiment['output path']

    output_files = [
        output_path / experiment['training feature file'],
        output_path / experiment['training target file'],
        output_path / experiment['evaluation feature file'],
        output_path / experiment['evaluation target file'],
        output_path / experiment['bias correction feature file'],
        output_path / experiment['bias correction target file'],
        output_path / experiment['test feature file'],
        output_path / experiment['test target file'], 
        output_path / 'metadata.json'
    ]

    all_exist = all(file.exists() for file in output_files)
    return all_exist

def single_experiment_data_pipeline(experiment_number, tau_max=None, train_fraction=0.15, eval_fraction=0.15, bias_correction_fraction=0.15, test_fraction=0.55, experiment_dict=experiments, strategy='remove'):
    """Generates, splits and saves the data for a given experiment."""
    experiment = experiment_dict.get(experiment_number)
    if experiment is None:
        raise ValueError(f"Experiment {experiment_number} not defined.")
    combined_data = validate_experiment_files(experiment_number, tau_max=tau_max, experiment_dict=experiment_dict)
    if combined_data is False:
        print("Data validation failed. Cannot proceed with data pipeline.")
        sys.exit(1)
    
    train_data, eval_data, bias_correction_data, test_data = split_data(
    combined_data=combined_data, 
    train_fraction=train_fraction,
    eval_fraction=eval_fraction,
    bias_correction_fraction=bias_correction_fraction,
    test_fraction=test_fraction,
    seed=42,
    strategy=strategy
    )

    #filepath setup for saving:
    project_root = Path(__file__).resolve().parent.parent.parent
    output_path = project_root / experiment['output path']
    output_path.mkdir(exist_ok=True, parents=True)

    #data shape checks:
    print(f"Experiment {experiment_number} Data Shapes:")
    print(f"Training Data Shape: {train_data.shape}")
    print(f"Evaluation Data Shape: {eval_data.shape}")
    print(f"Bias Correction Data Shape: {bias_correction_data.shape}")
    print(f"Test Data Shape: {test_data.shape}")

    #output data path setup:
    train_feature_path = output_path / experiment['training feature file']
    train_target_path = output_path / experiment['training target file']
    eval_feature_path = output_path / experiment['evaluation feature file']
    eval_target_path = output_path / experiment['evaluation target file']
    bias_correction_feature_path = output_path / experiment['bias correction feature file']
    bias_correction_target_path = output_path / experiment['bias correction target file']
    test_feature_path = output_path / experiment['test feature file']
    test_target_path = output_path / experiment['test target file']

    #saving data:
    save_split(train_data, train_feature_path, train_target_path)
    save_split(eval_data, eval_feature_path, eval_target_path)
    save_split(bias_correction_data, bias_correction_feature_path, bias_correction_target_path)
    save_split(test_data, test_feature_path, test_target_path)

    #writing metadata:
    write_experiment_metadata(experiment_number, experiment_dict=experiment_dict)

    #check if experiment was saved successfully:
    if check_experiment_output_exists(experiment_number, experiment_dict=experiment_dict):
        print(f"Experiment {experiment_number} data saved successfully.")
    else:
        print(f"Error saving Experiment {experiment_number} data.")

def load_experiment_data(experiment_number, split='train', experiment_dict=experiments):
    """Loads and returns the data for a given experiment and split."""
    experiment = experiment_dict.get(experiment_number)
    if experiment is None:
        raise ValueError(f"Experiment {experiment_number} not defined.")

    project_root = Path(__file__).resolve().parent.parent.parent
    data_path = project_root / experiment['output path']

    if split == 'train':
        X = np.load(data_path / experiment['training feature file'])
        y = np.load(data_path / experiment['training target file'])
    elif split == 'eval':
        X = np.load(data_path / experiment['evaluation feature file'])
        y = np.load(data_path / experiment['evaluation target file'])
    elif split == 'bias_correction':
        X = np.load(data_path / experiment['bias correction feature file'])
        y = np.load(data_path / experiment['bias correction target file'])
    elif split == 'test':
        X = np.load(data_path / experiment['test feature file'])
        y = np.load(data_path / experiment['test target file'])
    else:
        raise ValueError(f"Invalid split name: {split}. Please select from 'train', 'eval', 'bias_correction', 'test'.")
    
    return X, y

def all_experiments_data_pipeline(tau_max=None, train_fraction=0.15, eval_fraction=0.15, bias_correction_fraction=0.15, test_fraction=0.55, experiment_dict=experiments, strategy='remove'):
    """Generates, splits and saves the data for all experiments."""
    for experiment_number in experiment_dict.keys():
        print(f"Processing Experiment {experiment_number}...")
        single_experiment_data_pipeline(
            experiment_number,
            tau_max=tau_max,
            train_fraction=train_fraction,
            eval_fraction=eval_fraction,
            bias_correction_fraction=bias_correction_fraction,
            test_fraction=test_fraction,
            experiment_dict=experiment_dict,
            strategy=strategy
        )
        print(f"Experiment {experiment_number} data saved.\n")

def main(experiment_dict=experiments, tau_max=None, strategy='remove'):
    """Main function to generate and save data for all experiments."""
    all_experiments_data_pipeline(tau_max=tau_max, experiment_dict=experiment_dict, strategy=strategy)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ML experiment data from averaged correlator CSVs")
    parser.add_argument('--tau_max', type=int, default=None, help="Max tau cuttoff value")
    parser.add_argument('--e', type=int, help="Experiment number for single specific experiment data generation")
    parser.add_argument('--strategy', choices=['remove', 'warn'], type=str, default='remove', help="Missing data handling strategy: 'remove' or 'warn' allowed")
    args = parser.parse_args()

    if args.e is not None:
        print(f"Processing Experiment {args.e}...")
        experiment_data_pipeline(args.e, tau_max=args.tau_max, experiment_dict=experiments, strategy=args.strategy)
        print(f"Experiment {args.e} data saved.\n")
    else:
        main(tau_max=args.tau_max, strategy=args.strategy)
