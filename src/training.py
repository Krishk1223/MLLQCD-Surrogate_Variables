import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
from sklearn.preprocessing import StandardScaler
from model_scripts.transformer.transformer_model import Transformer
from model_scripts.cnn.cnn_model import CNN
from model_scripts.mlp.mlp_model import MLP
from model_scripts.gbr.gbr_model import GBR
from preprocessing.experiments import get_experiment, get_experiment_folder, load_experiment_data


def arcsinh_transform(data, scale=None):
    """Sign-preserving log-like transform. Inverse: sinh(x) * scale"""
    if scale is None:
        nonzero = np.abs(data[data != 0])
        scale = np.median(nonzero) if len(nonzero) > 0 else 1.0
    return np.arcsinh(data / scale), scale


def load_data(experiment_num, split, config_type='2pt'):
    """Load data from experiment by number."""
    return load_experiment_data(experiment_num, split, config_type)


def prepare_data_for_model(features, targets, model_type, arcsinh_scales=None):
    """
    Prepare data for a specific model type.
    
    Args:
        features, targets: Raw numpy arrays (N, seq_len)
        model_type: 'transformer', 'cnn', 'mlp', or 'gbr'
        arcsinh_scales: dict with 'feature' and 'target' scales (None for training set)
    
    Returns:
        features, targets: Transformed arrays
        arcsinh_scales: dict with scales (for reuse on eval/test)
    """
    if arcsinh_scales is None:
        arcsinh_scales = {}
    
    # Apply arcsinh transform
    if 'feature' not in arcsinh_scales:
        features, arcsinh_scales['feature'] = arcsinh_transform(features)
    else:
        features, _ = arcsinh_transform(features, arcsinh_scales['feature'])
    
    if 'target' not in arcsinh_scales:
        targets, arcsinh_scales['target'] = arcsinh_transform(targets)
    else:
        targets, _ = arcsinh_transform(targets, arcsinh_scales['target'])
    
    # Reshape based on model type
    if model_type in ['transformer', 'cnn']:
        # Sequence models: (N, seq_len, 1)
        features = features.reshape(features.shape[0], features.shape[1], 1)
        targets = targets.reshape(targets.shape[0], targets.shape[1], 1)
    else:
        # Flat models (mlp, gbr): keep as (N, seq_len)
        pass
    
    return features, targets, arcsinh_scales


def scale_data(train_features, train_targets, eval_features, eval_targets,
               test_features, test_targets, model_type):
    """Fit scalers on training data and transform all splits."""
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    if model_type in ['transformer', 'cnn']:
        # 3D data: (N, seq_len, 1)
        n_train, seq_len, n_features = train_features.shape
        
        train_features_scaled = feature_scaler.fit_transform(
            train_features.reshape(-1, n_features)
        ).reshape(n_train, seq_len, n_features)
        
        eval_features_scaled = feature_scaler.transform(
            eval_features.reshape(-1, n_features)
        ).reshape(eval_features.shape)
        
        test_features_scaled = feature_scaler.transform(
            test_features.reshape(-1, n_features)
        ).reshape(test_features.shape)
        
        train_targets_scaled = target_scaler.fit_transform(
            train_targets.reshape(-1, 1)
        ).reshape(train_targets.shape)
        
        eval_targets_scaled = target_scaler.transform(
            eval_targets.reshape(-1, 1)
        ).reshape(eval_targets.shape)
        
        test_targets_scaled = target_scaler.transform(
            test_targets.reshape(-1, 1)
        ).reshape(test_targets.shape)
    else:
        # 2D data: (N, seq_len)
        train_features_scaled = feature_scaler.fit_transform(train_features)
        eval_features_scaled = feature_scaler.transform(eval_features)
        test_features_scaled = feature_scaler.transform(test_features)
        
        train_targets_scaled = target_scaler.fit_transform(train_targets)
        eval_targets_scaled = target_scaler.transform(eval_targets)
        test_targets_scaled = target_scaler.transform(test_targets)
    
    return (train_features_scaled, train_targets_scaled,
            eval_features_scaled, eval_targets_scaled,
            test_features_scaled, test_targets_scaled,
            feature_scaler, target_scaler)


class DictDataset(torch.utils.data.Dataset):
    """Dataset that returns dicts with 'features' and 'targets' keys."""
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {'features': self.features[idx], 'targets': self.targets[idx]}


def create_dataloaders(train_features, train_targets, eval_features, eval_targets,
                       test_features, test_targets, batch_size):
    """Create PyTorch DataLoaders for sequence models."""
    train_loader = DataLoader(
        DictDataset(train_features, train_targets), 
        batch_size=batch_size, shuffle=True
    )
    eval_loader = DataLoader(
        DictDataset(eval_features, eval_targets), 
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        DictDataset(test_features, test_targets), 
        batch_size=batch_size, shuffle=False
    )
    return train_loader, eval_loader, test_loader


def save_scalers(feature_scaler, target_scaler, arcsinh_scales, model_dir):
    """Save fitted scalers."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez(model_dir / 'scalers.npz',
             feature_mean=feature_scaler.mean_,
             feature_scale=feature_scaler.scale_,
             target_mean=target_scaler.mean_,
             target_scale=target_scaler.scale_,
             arcsinh_feature_scale=arcsinh_scales['feature'],
             arcsinh_target_scale=arcsinh_scales['target'])
    print(f"Scalers saved to: {model_dir / 'scalers.npz'}")


def train_transformer(config, experiment_folder, train_loader, eval_loader, test_loader,
                      bias_loader, feature_scaler, target_scaler, arcsinh_scales,
                      use_bias_correction=False, pretrained_path=None, freeze_encoder=False):
    """Train a Transformer model."""
    print("Initializing Transformer model:")
    
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch['features'].shape[-1]
    
    model = Transformer(config, experiment_folder, input_dim=input_dim)
    model.build_model()
    
    # Load pretrained weights if specified
    if pretrained_path and Path(pretrained_path).exists():
        print(f"Loading pretrained weights from: {pretrained_path}")
        model.model.load_state_dict(torch.load(pretrained_path, map_location=model.device))
        
        if freeze_encoder:
            print("Freezing encoder layers")
            for name, param in model.model.named_parameters():
                if 'regression_head' not in name and 'output' not in name:
                    param.requires_grad = False
    
    bias_subdir = 'bias_corrected' if use_bias_correction else 'bias_uncorrected'
    model.results_dir = model.results_dir / bias_subdir
    model.results_dir.mkdir(parents=True, exist_ok=True)
    
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    total = model.count_parameters()
    print(f"Model has {trainable:,}/{total:,} trainable parameters\n")
    
    model.train_model(train_loader, eval_loader)
    save_scalers(feature_scaler, target_scaler, arcsinh_scales, model.model_dir)
    
    bias_vector = None
    if use_bias_correction and bias_loader is not None:
        print("\nComputing bias correction:")
        bias_vector = model.compute_bias_correction(bias_loader, target_scaler)
    
    print("\nEvaluating on test set:")
    result = model.evaluate_model(test_loader, target_scaler, bias_vector, save_predictions=True)
    metrics = result[0]
    predictions = result[2] if len(result) == 4 else result[1]
    
    print(f"\nResults saved to {model.results_dir}")
    return model, metrics, predictions, bias_vector


def train_cnn(config, experiment_folder, train_loader, eval_loader, test_loader,
              bias_loader, feature_scaler, target_scaler, arcsinh_scales,
              use_bias_correction=False, pretrained_path=None):
    """Train a CNN model."""
    print("Initializing CNN model:")
    
    sample_batch = next(iter(train_loader))
    input_channels = sample_batch['features'].shape[-1]
    
    model = CNN(config, experiment_folder, input_channels=input_channels)
    model.build_model()
    
    # Load pretrained weights if specified
    if pretrained_path and Path(pretrained_path).exists():
        print(f"Loading pretrained weights from: {pretrained_path}")
        model.model.load_state_dict(torch.load(pretrained_path, map_location=model.device))
    
    bias_subdir = 'bias_corrected' if use_bias_correction else 'bias_uncorrected'
    model.results_dir = model.results_dir / bias_subdir
    model.results_dir.mkdir(parents=True, exist_ok=True)
    
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    total = model.count_parameters()
    print(f"Model has {trainable:,}/{total:,} trainable parameters\n")
    
    model.train_model(train_loader, eval_loader)
    save_scalers(feature_scaler, target_scaler, arcsinh_scales, model.model_dir)
    
    bias_vector = None
    if use_bias_correction and bias_loader is not None:
        print("\nComputing bias correction:")
        bias_vector = model.compute_bias_correction(bias_loader, target_scaler)
    
    print("\nEvaluating on test set:")
    result = model.evaluate_model(test_loader, target_scaler, bias_vector, save_predictions=True)
    metrics = result[0]
    predictions = result[2] if len(result) == 4 else result[1]
    
    print(f"\nResults saved to {model.results_dir}")
    return model, metrics, predictions, bias_vector


def train_mlp(config, experiment_folder, train_X, train_y, eval_X, eval_y, test_X, test_y,
              feature_scaler, target_scaler, arcsinh_scales,
              use_bias_correction=False, bias_X=None, bias_y=None, pretrained_path=None):
    """Train an MLP model."""
    print("Initializing MLP model:")
    
    input_dim = train_X.shape[1]
    output_dim = train_y.shape[1]
    
    model = MLP(config, experiment_folder, input_dim=input_dim, output_dim=output_dim)
    model.build_model()
    
    # Load pretrained weights if specified
    if pretrained_path and Path(pretrained_path).exists():
        print(f"Loading pretrained weights from: {pretrained_path}")
        model.model.load_state_dict(torch.load(pretrained_path, map_location=model.device))
    
    bias_subdir = 'bias_corrected' if use_bias_correction else 'bias_uncorrected'
    model.results_dir = model.results_dir / bias_subdir
    model.results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Model has {model.count_parameters():,} trainable parameters\n")
    
    model.train_model(train_X, train_y, eval_X, eval_y)
    save_scalers(feature_scaler, target_scaler, arcsinh_scales, model.model_dir)
    
    bias_vector = None
    if use_bias_correction and bias_X is not None:
        print("\nComputing bias correction:")
        bias_vector = model.compute_bias_correction(bias_X, bias_y, target_scaler)
    
    print("\nEvaluating on test set:")
    metrics, predictions = model.evaluate_model(test_X, test_y, target_scaler, bias_vector, save_predictions=True)
    
    print(f"\nResults saved to {model.results_dir}")
    return model, metrics, predictions, bias_vector


def train_gbr(config, experiment_folder, train_X, train_y, test_X, test_y,
              feature_scaler, target_scaler, arcsinh_scales,
              use_bias_correction=False, bias_X=None, bias_y=None):
    """Train a Gradient Boosted Regressor."""
    print("Initializing GBR model:")
    
    model = GBR(config, experiment_folder)
    model.build_model(output_dim=train_y.shape[1])
    
    bias_subdir = 'bias_corrected' if use_bias_correction else 'bias_uncorrected'
    model.results_dir = model.results_dir / bias_subdir
    model.results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Model complexity: {model.count_parameters()} estimators × timeslices\n")
    
    model.train_model(train_X, train_y)
    save_scalers(feature_scaler, target_scaler, arcsinh_scales, model.model_dir)
    model.save_model()
    
    bias_vector = None
    if use_bias_correction and bias_X is not None:
        print("\nComputing bias correction:")
        bias_vector = model.compute_bias_correction(bias_X, bias_y, target_scaler)
    
    print("\nEvaluating on test set:")
    metrics, predictions = model.evaluate_model(test_X, test_y, target_scaler, bias_vector, save_predictions=True)
    
    print(f"\nResults saved to {model.results_dir}")
    return model, metrics, predictions, bias_vector


def train_experiment(config_path, experiment_num, model_type='transformer', 
                     use_bias_correction=False, config_type='2pt',
                     pretrained_path=None, freeze_encoder=False):
    """
    Complete training pipeline for any model type.
    
    Args:
        config_path: Path to YAML config file
        experiment_num: Experiment number from experiments YAML
        model_type: 'transformer', 'cnn', 'mlp', or 'gbr'
        use_bias_correction: Whether to apply bias correction
        config_type: '2pt' or '3pt' experiment config
        pretrained_path: Path to pretrained model weights for transfer learning
        freeze_encoder: If True, freeze encoder layers during transfer learning
        
    Returns:
        model, metrics, predictions, bias_vector, feature_scaler, target_scaler
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    experiment_folder = get_experiment_folder(experiment_num, config_type)
    exp_config, full_config = get_experiment(experiment_num, config_type)
    
    # Check for transfer learning config in experiment
    if exp_config.get('transfer_learning') and pretrained_path is None:
        tl = exp_config['transfer_learning']
        pretrained_exp = tl['pretrained_experiment']
        source_config = tl.get('source_config_type', config_type)  # Support cross-config transfer
        pretrained_folder = get_experiment_folder(pretrained_exp, source_config)
        project_root = Path(__file__).resolve().parent.parent
        pretrained_path = project_root / "models" / pretrained_folder / f"{model_type}_model" / "best_model.pth"
        if not pretrained_path.exists():
            print(f"Warning: Pretrained model not found at {pretrained_path}")
            pretrained_path = None
        else:
            freeze_encoder = tl.get('freeze_encoder', freeze_encoder)
            print(f"Transfer learning from experiment {pretrained_exp} ({source_config}), freeze_encoder={freeze_encoder}")
    
    exp_type_label = f"[{config_type.upper()}]"
    transfer_label = " [TRANSFER]" if pretrained_path else ""
    print(f"Training {model_type.upper()} on {exp_type_label} Experiment {experiment_num}: {exp_config['name']}{transfer_label}\n")
    
    # Load raw data
    print("Loading data:")
    train_features, train_targets = load_data(experiment_num, 'train', config_type)
    eval_features, eval_targets = load_data(experiment_num, 'eval', config_type)
    test_features, test_targets = load_data(experiment_num, 'test', config_type)
    
    bias_features, bias_targets = None, None
    if use_bias_correction:
        bias_features, bias_targets = load_data(experiment_num, 'bias_correction', config_type)
    
    print(f"Train: {train_features.shape}, Eval: {eval_features.shape}, Test: {test_features.shape}")
    if use_bias_correction:
        print(f"Bias: {bias_features.shape}")
    print()
    
    # Transform data for model type
    train_features, train_targets, arcsinh_scales = prepare_data_for_model(
        train_features, train_targets, model_type
    )
    eval_features, eval_targets, _ = prepare_data_for_model(
        eval_features, eval_targets, model_type, arcsinh_scales
    )
    test_features, test_targets, _ = prepare_data_for_model(
        test_features, test_targets, model_type, arcsinh_scales
    )
    if use_bias_correction:
        bias_features, bias_targets, _ = prepare_data_for_model(
            bias_features, bias_targets, model_type, arcsinh_scales
        )
    
    # Scale data
    print("Scaling data:\n")
    (train_features_scaled, train_targets_scaled,
     eval_features_scaled, eval_targets_scaled,
     test_features_scaled, test_targets_scaled,
     feature_scaler, target_scaler) = scale_data(
        train_features, train_targets,
        eval_features, eval_targets,
        test_features, test_targets,
        model_type
    )
    
    # Scale bias data if needed
    bias_features_scaled, bias_targets_scaled = None, None
    if use_bias_correction:
        if model_type in ['transformer', 'cnn']:
            bias_features_scaled = feature_scaler.transform(
                bias_features.reshape(-1, bias_features.shape[-1])
            ).reshape(bias_features.shape)
            bias_targets_scaled = target_scaler.transform(
                bias_targets.reshape(-1, 1)
            ).reshape(bias_targets.shape)
        else:
            bias_features_scaled = feature_scaler.transform(bias_features)
            bias_targets_scaled = target_scaler.transform(bias_targets)
    
    # Train based on model type
    if model_type in ['transformer', 'cnn']:
        # Create dataloaders for sequence models
        batch_size = config['training']['batch_size']
        train_loader, eval_loader, test_loader = create_dataloaders(
            train_features_scaled, train_targets_scaled,
            eval_features_scaled, eval_targets_scaled,
            test_features_scaled, test_targets_scaled,
            batch_size
        )
        
        bias_loader = None
        if use_bias_correction:
            bias_loader = DataLoader(
                DictDataset(bias_features_scaled, bias_targets_scaled),
                batch_size=batch_size, shuffle=False
            )
        
        if model_type == 'transformer':
            result = train_transformer(
                config, experiment_folder, train_loader, eval_loader, test_loader,
                bias_loader, feature_scaler, target_scaler, arcsinh_scales, use_bias_correction,
                pretrained_path, freeze_encoder
            )
        else:
            result = train_cnn(
                config, experiment_folder, train_loader, eval_loader, test_loader,
                bias_loader, feature_scaler, target_scaler, arcsinh_scales, use_bias_correction,
                pretrained_path
            )
    
    elif model_type == 'mlp':
        result = train_mlp(
            config, experiment_folder,
            train_features_scaled, train_targets_scaled,
            eval_features_scaled, eval_targets_scaled,
            test_features_scaled, test_targets_scaled,
            feature_scaler, target_scaler, arcsinh_scales,
            use_bias_correction, bias_features_scaled, bias_targets_scaled,
            pretrained_path
        )
    
    elif model_type == 'gbr':
        result = train_gbr(
            config, experiment_folder,
            train_features_scaled, train_targets_scaled,
            test_features_scaled, test_targets_scaled,
            feature_scaler, target_scaler, arcsinh_scales,
            use_bias_correction, bias_features_scaled, bias_targets_scaled
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose: transformer, cnn, mlp, gbr")
    
    model, metrics, predictions, bias_vector = result
    return model, metrics, predictions, bias_vector, feature_scaler, target_scaler


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train ML models on LQCD experiments")
    parser.add_argument('--experiment', '-e', type=int, default=2, help="Experiment number")
    parser.add_argument('--model', '-m', type=str, default='transformer', 
                       choices=['transformer', 'cnn', 'mlp', 'gbr'], help="Model type")
    parser.add_argument('--config', type=str, default='../configs/transformer.yaml', 
                       help="Path to model config YAML")
    parser.add_argument('--config-type', '-t', type=str, default='2pt', choices=['2pt', '3pt'],
                       help="Experiment config type: 2pt or 3pt (default: 2pt)")
    parser.add_argument('--bias', '-b', action='store_true', help="Use bias correction")
    parser.add_argument('--pretrained', '-p', type=str, default=None,
                       help="Path to pretrained model weights for transfer learning")
    parser.add_argument('--freeze-encoder', '-f', action='store_true',
                       help="Freeze encoder layers during transfer learning")
    args = parser.parse_args()
    
    model, metrics, preds, bias_vec, feat_scaler, targ_scaler = train_experiment(
        config_path=args.config,
        experiment_num=args.experiment,
        model_type=args.model,
        use_bias_correction=args.bias,
        config_type=args.config_type,
        pretrained_path=args.pretrained,
        freeze_encoder=args.freeze_encoder
    )
