"""
Experiment data generation from YAML config.
Supports 2pt and 3pt correlator experiments.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CONFIG_PATHS = {
    '2pt': PROJECT_ROOT / "configs" / "experiments_two_pt.yaml",
    '3pt': PROJECT_ROOT / "configs" / "experiments_three_pt.yaml"
}


def load_config(config_type='2pt'):
    """Load experiment config from YAML."""
    if config_type not in CONFIG_PATHS:
        raise ValueError(f"Unknown config type: {config_type}. Use '2pt' or '3pt'")
    with open(CONFIG_PATHS[config_type]) as f:
        return yaml.safe_load(f)


def save_config(config, config_type='2pt'):
    """Save experiment config to YAML."""
    if config_type not in CONFIG_PATHS:
        raise ValueError(f"Unknown config type: {config_type}. Use '2pt' or '3pt'")
    with open(CONFIG_PATHS[config_type], 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_experiment(experiment_num, config_type='2pt'):
    """Get experiment config by number."""
    config = load_config(config_type)
    exp = config['experiments'].get(experiment_num)
    if exp is None:
        raise ValueError(f"Experiment {experiment_num} not found in {config_type}. "
                        f"Available: {list(config['experiments'].keys())}")
    return exp, config


def extract_identifier(filename):
    """Extract identifier from filename. Handles both 2pt and 3pt formats."""
    stem = Path(filename).stem
    stem = stem.replace("_fine", "").replace("_averaged", "")
    
    # 3pt format: localscalar_3pt_T19_qsq0 -> localscalar_T19_qsq0
    if "3pt" in stem:
        parts = stem.split("_")
        operator = parts[0]  # localscalar or localtempvector
        t_val = next((p for p in parts if p.startswith('T')), '')
        momentum = next((p for p in parts if 'qsq' in p or 'qmax' in p), '')
        ident = f"{operator}_{t_val}"
        if momentum:
            ident += f"_{momentum}"
        return ident
    
    # 2pt format: 2pt_K_fine -> K
    return stem.replace("2pt_", "")


def load_and_align_data(exp, config):
    """Load input and target CSVs, align by config."""
    time_sources = config['settings']['time_sources']
    input_path = PROJECT_ROOT / exp['input_path']
    
    # Load inputs
    if len(exp['input_files']) > 1:
        dfs = [pd.read_csv(input_path / f, header=None) for f in exp['input_files']]
        input_data = pd.concat(dfs, axis=1, ignore_index=True)
    else:
        input_data = pd.read_csv(input_path / exp['input_files'][0], header=None)
    
    # Load target
    target_data = pd.read_csv(input_path / exp['target_file'], header=None)
    
    # Align rows
    min_rows = min(len(input_data), len(target_data))
    aligned_rows = (min_rows // time_sources) * time_sources
    
    input_data = input_data.iloc[:aligned_rows].reset_index(drop=True)
    target_data = target_data.iloc[:aligned_rows].reset_index(drop=True)
    
    # Create multi-index
    n_configs = aligned_rows // time_sources
    config_ids = np.repeat(np.arange(1, n_configs + 1), time_sources)
    time_source_ids = np.tile(np.arange(time_sources), n_configs)
    
    multi_index = pd.MultiIndex.from_arrays([config_ids, time_source_ids], names=['config_id', 'time_source'])
    input_data.index = multi_index
    target_data.index = multi_index
    
    # Name columns
    input_cols = []
    for filename in exp['input_files']:
        ident = extract_identifier(filename)
        n_cols = pd.read_csv(input_path / filename, header=None, nrows=1).shape[1]
        input_cols.extend([f"input_{ident}_tau_{i}" for i in range(1, n_cols + 1)])
    input_data.columns = input_cols[:input_data.shape[1]]
    
    ident = extract_identifier(exp['target_file'])
    target_data.columns = [f"target_{ident}_tau_{i}" for i in range(1, target_data.shape[1] + 1)]
    
    # Join and filter complete configs
    paired = input_data.join(target_data, how='inner')
    config_counts = paired.groupby(level='config_id').size()
    complete = config_counts[config_counts == time_sources].index
    paired = paired[paired.index.get_level_values('config_id').isin(complete)]
    
    # Drop NaN rows
    nan_count = paired.isna().sum().sum()
    if nan_count > 0:
        print(f"Dropping {nan_count} NaN values")
        paired = paired.dropna()
    
    print(f"Loaded {len(paired) // time_sources} configs, {paired.shape[1]} columns")
    return paired


def split_data(df, config):
    """Split data into train/eval/bias/test sets."""
    splits = config['splits']
    seed = config['settings']['seed']
    
    unique_configs = df.index.get_level_values('config_id').unique()
    n = len(unique_configs)
    
    rng = np.random.default_rng(seed=seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    
    n_train = int(splits['train'] * n)
    n_eval = int(splits['eval'] * n)
    n_bias = int(splits['bias_correction'] * n)
    
    config_ids = {
        'train': unique_configs[idx[:n_train]],
        'eval': unique_configs[idx[n_train:n_train + n_eval]],
        'bias_correction': unique_configs[idx[n_train + n_eval:n_train + n_eval + n_bias]],
        'test': unique_configs[idx[n_train + n_eval + n_bias:]]
    }
    
    result = {}
    for name, ids in config_ids.items():
        mask = df.index.get_level_values('config_id').isin(ids)
        result[name] = df[mask]
    
    return result


def save_split(df, output_path, split_name):
    """Save split as X and y numpy files."""
    input_cols = [c for c in df.columns if c.startswith('input_')]
    target_cols = [c for c in df.columns if c.startswith('target_')]
    
    X = df[input_cols].to_numpy()
    y = df[target_cols].to_numpy()
    
    file_names = {
        'train': ('train_data_X.npy', 'train_data_y.npy'),
        'eval': ('evaluation_data_X.npy', 'evaluation_data_y.npy'),
        'bias_correction': ('bias_correction_data_X.npy', 'bias_correction_data_y.npy'),
        'test': ('test_data_X.npy', 'test_data_y.npy')
    }
    
    x_file, y_file = file_names[split_name]
    np.save(output_path / x_file, X)
    np.save(output_path / y_file, y)


def generate_experiment(experiment_num, config_type='2pt', tau_max=None):
    """Generate and save data for a single experiment."""
    exp, config = get_experiment(experiment_num, config_type)
    print(f"\n=== [{config_type.upper()}] Experiment {experiment_num}: {exp['name']} ===")
    
    data = load_and_align_data(exp, config)
    
    if tau_max:
        input_cols = [c for c in data.columns if c.startswith('input_')][:tau_max]
        target_cols = [c for c in data.columns if c.startswith('target_')][:tau_max]
        data = data[input_cols + target_cols]
    
    splits = split_data(data, config)
    
    output_path = PROJECT_ROOT / exp['output_path']
    output_path.mkdir(parents=True, exist_ok=True)
    
    for name, df in splits.items():
        save_split(df, output_path, name)
        print(f"  {name}: {df.shape[0]} rows")
    
    print(f"Saved to: {output_path}")
    return True


def load_experiment_data(experiment_num, split='train', config_type='2pt'):
    """Load saved experiment data."""
    exp, _ = get_experiment(experiment_num, config_type)
    data_path = PROJECT_ROOT / exp['output_path']
    
    file_map = {
        'train': ('train_data_X.npy', 'train_data_y.npy'),
        'eval': ('evaluation_data_X.npy', 'evaluation_data_y.npy'),
        'bias_correction': ('bias_correction_data_X.npy', 'bias_correction_data_y.npy'),
        'test': ('test_data_X.npy', 'test_data_y.npy')
    }
    
    x_file, y_file = file_map[split]
    return np.load(data_path / x_file), np.load(data_path / y_file)


def get_experiment_folder(experiment_num, config_type='2pt'):
    """Get the experiment folder name from experiment number."""
    exp, _ = get_experiment(experiment_num, config_type)
    return Path(exp['output_path']).name


def list_experiments(config_type='2pt'):
    """List all available experiments."""
    config = load_config(config_type)
    print(f"\nAvailable {config_type.upper()} Experiments:")
    print("-" * 60)
    for num, exp in config['experiments'].items():
        transfer = " [TRANSFER]" if exp.get('transfer_learning') else ""
        print(f"  {num}: {exp['name']}{transfer}")
    print()


def generate_all(config_type='2pt', tau_max=None):
    """Generate data for all experiments."""
    config = load_config(config_type)
    for num in config['experiments']:
        generate_experiment(num, config_type, tau_max)


def add_experiment_interactive(config_type='2pt'):
    """Interactively add a new experiment to the YAML config."""
    config = load_config(config_type)
    
    # Get next experiment number
    existing_nums = list(config['experiments'].keys())
    next_num = max(existing_nums) + 1 if existing_nums else 1
    
    print(f"\n=== Add New {config_type.upper()} Experiment (will be #{next_num}) ===\n")
    
    name = input("Experiment name: ").strip()
    description = input("Description: ").strip()
    
    exp_type = config_type
    if config_type == '3pt':
        operator = input("Operator (localscalar/localtempvector): ").strip()
        t_value = int(input("T value (e.g., 19): ").strip())
    
    input_path = input("Input path [data/processed/unaveraged_data]: ").strip()
    input_path = input_path or "data/processed/unaveraged_data"
    
    # Input files
    input_files = []
    while True:
        f = input(f"Input file {len(input_files)+1} (empty to finish): ").strip()
        if not f:
            break
        input_files.append(f)
    
    if not input_files:
        print("Error: At least one input file required")
        return False
    
    target_file = input("Target file: ").strip()
    
    # Auto-generate output path
    default_output = f"data/experiments/{config_type}_exp_{next_num}"
    output_path = input(f"Output path [{default_output}]: ").strip()
    output_path = output_path or default_output
    
    # Build experiment dict
    new_exp = {
        'name': name,
        'description': description,
        'type': exp_type,
        'input_path': input_path,
        'output_path': output_path,
        'input_files': input_files,
        'target_file': target_file
    }
    
    if config_type == '3pt':
        new_exp['operator'] = operator
        new_exp['T_value'] = t_value
        
        # Transfer learning?
        transfer = input("Use transfer learning? (y/n): ").strip().lower()
        if transfer == 'y':
            pretrained = int(input("Pretrained experiment number: ").strip())
            freeze = input("Freeze encoder? (y/n): ").strip().lower() == 'y'
            new_exp['transfer_learning'] = {
                'pretrained_experiment': pretrained,
                'freeze_encoder': freeze
            }
    
    config['experiments'][next_num] = new_exp
    save_config(config, config_type)
    print(f"\nExperiment {next_num} added to {CONFIG_PATHS[config_type]}")
    return True


def delete_experiment(experiment_num, config_type='2pt', override=False):
    """Delete an experiment from the YAML config."""
    if not override:
        print("Error: --override flag required to delete experiments")
        return False
    
    config = load_config(config_type)
    if experiment_num not in config['experiments']:
        print(f"Experiment {experiment_num} not found")
        return False
    
    del config['experiments'][experiment_num]
    save_config(config, config_type)
    print(f"Experiment {experiment_num} deleted from {config_type}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate experiment data from YAML config")
    parser.add_argument('--experiment', '-e', type=int, help="Experiment number (omit for all)")
    parser.add_argument('--config-type', '-c', choices=['2pt', '3pt'], default='2pt',
                       help="Config type: 2pt or 3pt (default: 2pt)")
    parser.add_argument('--tau_max', type=int, help="Max tau cutoff")
    parser.add_argument('--list', '-l', action='store_true', help="List available experiments")
    parser.add_argument('--interactive', '-i', action='store_true', help="Add experiment interactively")
    parser.add_argument('--delete', '-d', type=int, help="Delete experiment by number")
    parser.add_argument('--override', action='store_true', help="Required for delete operations")
    args = parser.parse_args()
    
    if args.list:
        list_experiments(args.config_type)
    elif args.interactive:
        add_experiment_interactive(args.config_type)
    elif args.delete:
        delete_experiment(args.delete, args.config_type, args.override)
    elif args.experiment:
        generate_experiment(args.experiment, args.config_type, args.tau_max)
    else:
        generate_all(args.config_type, args.tau_max)
