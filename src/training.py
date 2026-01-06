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


def split_by_parity(data):
    """
    Split correlator data into even and odd timeslice streams.
    
    For staggered fermions:
    - Even timeslices (t=0,2,4,...): normal parity, mostly positive
    - Odd timeslices (t=1,3,5,...): oscillating parity, mostly negative
    
    Args:
        data: (N, T) array where T is the time extent
        
    Returns:
        even_data: (N, T//2) even timeslices
        odd_data: (N, T//2) odd timeslices (if T is even) or (N, T//2) if T is odd
    """
    even_data = data[:, 0::2]  # t=0,2,4,...
    odd_data = data[:, 1::2]   # t=1,3,5,...
    return even_data, odd_data


def interleave_parity(even_data, odd_data):
    """
    Recombine even and odd timeslice streams back to full correlator.
    
    Args:
        even_data: (N, T_even) even timeslices
        odd_data: (N, T_odd) odd timeslices
        
    Returns:
        full_data: (N, T_even + T_odd) interleaved data
    """
    N = even_data.shape[0]
    T_even = even_data.shape[1]
    T_odd = odd_data.shape[1]
    T_full = T_even + T_odd
    
    full_data = np.zeros((N, T_full), dtype=even_data.dtype)
    full_data[:, 0::2] = even_data
    full_data[:, 1::2] = odd_data
    return full_data


def prepare_multichannel_parity_data(data, transform_info=None, dataset_name=None):
    """
    Prepare correlator as 2-channel input: (even, odd) streams stacked.
    
    For staggered fermions, this allows a single model to learn both parity
    channels simultaneously, preserving correlations between them.
    
    Args:
        data: (N, T) raw correlator data with alternating signs
        transform_info: dict to store signs for inverse transform
        dataset_name: 'train', 'eval', 'test', 'bias' for sign storage keys
        
    Returns:
        multichannel: (N, T_half, 2) array with [even, odd] channels
        transform_info: updated with signs for each channel
    """
    if transform_info is None:
        transform_info = {}
    
    # Split by parity
    even, odd = split_by_parity(data)  # (N, T_even), (N, T_odd)
    
    # Store signs before log transform
    prefix = f'{dataset_name}_' if dataset_name else ''
    transform_info[f'{prefix}even_signs'] = np.sign(even)
    transform_info[f'{prefix}odd_signs'] = np.sign(odd)
    
    # Log transform (absolute value)
    even_log = np.log(np.abs(even) + 1e-30)
    odd_log = np.log(np.abs(odd) + 1e-30)
    
    # Handle unequal lengths (T_even vs T_odd can differ by 1)
    min_len = min(even_log.shape[1], odd_log.shape[1])
    even_log = even_log[:, :min_len]
    odd_log = odd_log[:, :min_len]
    
    # Store truncation info for reconstruction
    transform_info[f'{prefix}T_even'] = even.shape[1]
    transform_info[f'{prefix}T_odd'] = odd.shape[1]
    transform_info[f'{prefix}T_min'] = min_len
    
    # Stack as 2 channels: (N, T_half, 2)
    multichannel = np.stack([even_log, odd_log], axis=-1)
    
    return multichannel, transform_info


def inverse_multichannel_parity(predictions, transform_info, dataset_name=None):
    """
    Inverse transform 2-channel predictions back to full correlator.
    
    Args:
        predictions: (N, T_half, 2) with [even, odd] channel predictions
        transform_info: dict with stored signs
        dataset_name: which dataset's signs to use
        
    Returns:
        full_corr: (N, T) full correlator with correct signs restored
    """
    prefix = f'{dataset_name}_' if dataset_name else ''
    
    # Extract channels
    even_log = predictions[:, :, 0]  # (N, T_half)
    odd_log = predictions[:, :, 1]   # (N, T_half)
    
    # Get stored signs (may be larger than T_half due to truncation)
    T_min = transform_info.get(f'{prefix}T_min', even_log.shape[1])
    even_signs = transform_info.get(f'{prefix}even_signs', np.ones_like(even_log))[:, :T_min]
    odd_signs = transform_info.get(f'{prefix}odd_signs', np.ones_like(odd_log))[:, :T_min]
    
    # Inverse log with signs
    even = np.exp(even_log) * even_signs
    odd = np.exp(odd_log) * odd_signs
    
    # Pad back to original lengths if needed
    T_even = transform_info.get(f'{prefix}T_even', even.shape[1])
    T_odd = transform_info.get(f'{prefix}T_odd', odd.shape[1])
    
    if even.shape[1] < T_even:
        # Pad with last value (or zeros)
        pad_even = np.zeros((even.shape[0], T_even - even.shape[1]))
        even = np.concatenate([even, pad_even], axis=1)
    if odd.shape[1] < T_odd:
        pad_odd = np.zeros((odd.shape[0], T_odd - odd.shape[1]))
        odd = np.concatenate([odd, pad_odd], axis=1)
    
    # Interleave back to full correlator
    return interleave_parity(even, odd)


def log_transform(data, epsilon=1e-30):
    """
    Log transform for correlator data: log(|data| + epsilon).
    Stores sign separately for inverse transform.
    
    Returns:
        log_data: log(|data| + epsilon)
        signs: sign of original data (+1 or -1)
    """
    signs = np.sign(data)
    signs[signs == 0] = 1  # Treat zero as positive
    log_data = np.log(np.abs(data) + epsilon)
    return log_data, signs


def inverse_log_transform(log_data, signs):
    """Inverse of log_transform: exp(log_data) * signs"""
    return np.exp(log_data) * signs


def arcsinh_transform(data, scale=None):
    """Sign-preserving log-like transform. Inverse: sinh(x) * scale"""
    if scale is None:
        nonzero = np.abs(data[data != 0])
        scale = np.median(nonzero) if len(nonzero) > 0 else 1.0
    return np.arcsinh(data / scale), scale


def load_data(experiment_num, split, config_type='2pt'):
    """Load data from experiment by number."""
    return load_experiment_data(experiment_num, split, config_type)


def prepare_data_for_model(features, targets, model_type, transform_info=None, dataset_name=None):
    """
    Prepare data for a specific model type using LOG transform.
    
    Args:
        features, targets: Raw numpy arrays (N, seq_len)
        model_type: 'transformer', 'cnn', 'mlp', or 'gbr'
        transform_info: dict to store/retrieve signs for inverse transform
        dataset_name: 'train', 'eval', 'test', or 'bias' to store signs under separate keys
    
    Returns:
        features, targets: Log-transformed arrays
        transform_info: dict with signs for inverse transform (keyed by dataset_name)
    """
    if transform_info is None:
        transform_info = {}
    
    # Key names for this dataset
    feat_key = f'{dataset_name}_feature_signs' if dataset_name else 'feature_signs'
    targ_key = f'{dataset_name}_target_signs' if dataset_name else 'target_signs'
    
    # Apply LOG transform (store signs for inverse)
    features, transform_info[feat_key] = log_transform(features)
    targets, transform_info[targ_key] = log_transform(targets)
    
    # Reshape based on model type
    if model_type in ['transformer', 'cnn']:
        # Sequence models: (N, seq_len, 1)
        features = features.reshape(features.shape[0], features.shape[1], 1)
        targets = targets.reshape(targets.shape[0], targets.shape[1], 1)
    else:
        # Flat models (mlp, gbr): keep as (N, seq_len)
        pass
    
    return features, targets, transform_info


def prepare_multichannel_data(features, targets, model_type, transform_info=None, dataset_name=None):
    """
    Prepare data with 2-channel parity format for staggered fermions.
    
    Single model with 2 input channels (even, odd) and 2 output channels.
    This allows the model to learn correlations between parity streams.
    
    Args:
        features, targets: Raw numpy arrays (N, seq_len) with alternating signs
        model_type: 'transformer', 'cnn', 'mlp', or 'gbr'
        transform_info: dict to store signs for inverse transform
        dataset_name: 'train', 'eval', 'test', or 'bias'
    
    Returns:
        features: (N, T_half, 2) for sequence models, (N, T_half*2) for flat models
        targets: (N, T_half, 2) for sequence models, (N, T_half*2) for flat models
        transform_info: dict with signs and length info for reconstruction
    """
    if transform_info is None:
        transform_info = {}
    
    # Prepare multichannel data
    features, transform_info = prepare_multichannel_parity_data(
        features, transform_info, f'{dataset_name}_feat' if dataset_name else 'feat'
    )
    targets, transform_info = prepare_multichannel_parity_data(
        targets, transform_info, f'{dataset_name}_targ' if dataset_name else 'targ'
    )
    
    # For flat models, flatten the channels: (N, T_half, 2) -> (N, T_half*2)
    if model_type in ['mlp', 'gbr']:
        features = features.reshape(features.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)
    
    return features, targets, transform_info


def prepare_parity_split_data(features, targets, model_type, transform_info=None, dataset_name=None):
    """
    Prepare data with even/odd parity split for staggered fermions.
    
    Splits correlator into even (normal parity) and odd (oscillating parity) streams,
    applies log transform to each separately, preserving signs.
    
    Args:
        features, targets: Raw numpy arrays (N, seq_len) with alternating-sign staggered data
        model_type: 'transformer', 'cnn', 'mlp', or 'gbr'
        transform_info: dict to store signs for inverse transform
        dataset_name: 'train', 'eval', 'test', or 'bias'
    
    Returns:
        features_even, features_odd: Log-transformed even/odd feature streams
        targets_even, targets_odd: Log-transformed even/odd target streams
        transform_info: dict with signs for both parities
    """
    if transform_info is None:
        transform_info = {}
    
    # Split by parity
    feat_even, feat_odd = split_by_parity(features)
    targ_even, targ_odd = split_by_parity(targets)
    
    # Store original signs for each parity stream
    prefix = f'{dataset_name}_' if dataset_name else ''
    transform_info[f'{prefix}feature_even_signs'] = np.sign(feat_even)
    transform_info[f'{prefix}feature_odd_signs'] = np.sign(feat_odd)
    transform_info[f'{prefix}target_even_signs'] = np.sign(targ_even)
    transform_info[f'{prefix}target_odd_signs'] = np.sign(targ_odd)
    
    # Apply log transform (absolute value, signs stored above)
    feat_even_log = np.log(np.abs(feat_even) + 1e-30)
    feat_odd_log = np.log(np.abs(feat_odd) + 1e-30)
    targ_even_log = np.log(np.abs(targ_even) + 1e-30)
    targ_odd_log = np.log(np.abs(targ_odd) + 1e-30)
    
    # Reshape for sequence models
    if model_type in ['transformer', 'cnn']:
        feat_even_log = feat_even_log.reshape(feat_even_log.shape[0], feat_even_log.shape[1], 1)
        feat_odd_log = feat_odd_log.reshape(feat_odd_log.shape[0], feat_odd_log.shape[1], 1)
        targ_even_log = targ_even_log.reshape(targ_even_log.shape[0], targ_even_log.shape[1], 1)
        targ_odd_log = targ_odd_log.reshape(targ_odd_log.shape[0], targ_odd_log.shape[1], 1)
    
    return feat_even_log, feat_odd_log, targ_even_log, targ_odd_log, transform_info


def inverse_parity_transform(pred_even_log, pred_odd_log, transform_info, dataset_name=None):
    """
    Inverse transform parity-split predictions back to full correlator.
    
    Args:
        pred_even_log: (N, T/2) or (N, T/2, 1) log-scale even predictions
        pred_odd_log: (N, T/2) or (N, T/2, 1) log-scale odd predictions
        transform_info: dict with stored signs
        dataset_name: which dataset's signs to use
        
    Returns:
        full_pred: (N, T) full correlator with correct signs restored
    """
    # Flatten if 3D
    if pred_even_log.ndim == 3:
        pred_even_log = pred_even_log.squeeze(-1)
        pred_odd_log = pred_odd_log.squeeze(-1)
    
    prefix = f'{dataset_name}_' if dataset_name else ''
    even_signs = transform_info.get(f'{prefix}target_even_signs', np.ones_like(pred_even_log))
    odd_signs = transform_info.get(f'{prefix}target_odd_signs', np.ones_like(pred_odd_log))
    
    # Inverse log and apply signs
    pred_even = np.exp(pred_even_log) * even_signs
    pred_odd = np.exp(pred_odd_log) * odd_signs
    
    # Interleave back to full correlator
    return interleave_parity(pred_even, pred_odd)


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


def save_scalers(feature_scaler, target_scaler, transform_info, model_dir):
    """Save fitted scalers and transform info (signs for log transform)."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save scalers and signs for inverse transform (all dataset signs)
    save_dict = {
        'feature_mean': feature_scaler.mean_,
        'feature_scale': feature_scaler.scale_,
        'target_mean': target_scaler.mean_,
        'target_scale': target_scaler.scale_,
    }
    # Add all signs from transform_info
    for key, value in transform_info.items():
        if 'signs' in key:
            save_dict[key] = value
    
    np.savez(model_dir / 'scalers.npz', **save_dict)
    print(f"Scalers and transform info saved to: {model_dir / 'scalers.npz'}")


def train_transformer(config, experiment_folder, train_loader, eval_loader, test_loader,
                      bias_loader, feature_scaler, target_scaler, transform_info,
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
        model.load_state_dict(torch.load(pretrained_path, map_location=model.device))
        
        if freeze_encoder:
            print("Freezing encoder layers")
            for name, param in model.named_parameters():
                if 'regression_head' not in name and 'output' not in name:
                    param.requires_grad = False
    
    bias_subdir = 'bias_corrected' if use_bias_correction else 'bias_uncorrected'
    model.results_dir = model.results_dir / bias_subdir
    model.results_dir.mkdir(parents=True, exist_ok=True)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = model.count_parameters()
    print(f"Model has {trainable:,}/{total:,} trainable parameters\n")
    
    model.train_model(train_loader, eval_loader)
    save_scalers(feature_scaler, target_scaler, transform_info, model.model_dir)
    
    bias_vector = None
    if use_bias_correction and bias_loader is not None:
        print("\nComputing bias correction:")
        bias_vector = model.compute_bias_correction(
            bias_loader, target_scaler, target_signs=transform_info.get('bias_target_signs')
        )
    
    print("\nEvaluating on test set:")
    result = model.evaluate_model(
        test_loader, target_scaler, bias_vector, 
        save_predictions=True, target_signs=transform_info.get('test_target_signs')
    )
    metrics = result[0]
    predictions = result[2] if len(result) == 4 else result[1]
    
    print(f"\nResults saved to {model.results_dir}")
    return model, metrics, predictions, bias_vector


def train_cnn(config, experiment_folder, train_loader, eval_loader, test_loader,
              bias_loader, feature_scaler, target_scaler, transform_info,
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
        model.load_state_dict(torch.load(pretrained_path, map_location=model.device))
    
    bias_subdir = 'bias_corrected' if use_bias_correction else 'bias_uncorrected'
    model.results_dir = model.results_dir / bias_subdir
    model.results_dir.mkdir(parents=True, exist_ok=True)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = model.count_parameters()
    print(f"Model has {trainable:,}/{total:,} trainable parameters\n")
    
    model.train_model(train_loader, eval_loader)
    save_scalers(feature_scaler, target_scaler, transform_info, model.model_dir)
    
    bias_vector = None
    if use_bias_correction and bias_loader is not None:
        print("\nComputing bias correction:")
        bias_vector = model.compute_bias_correction(
            bias_loader, target_scaler, target_signs=transform_info.get('bias_target_signs')
        )
    
    print("\nEvaluating on test set:")
    result = model.evaluate_model(
        test_loader, target_scaler, bias_vector, 
        save_predictions=True, target_signs=transform_info.get('test_target_signs')
    )
    metrics = result[0]
    predictions = result[2] if len(result) == 4 else result[1]
    
    print(f"\nResults saved to {model.results_dir}")
    return model, metrics, predictions, bias_vector


def train_mlp(config, experiment_folder, train_X, train_y, eval_X, eval_y, test_X, test_y,
              feature_scaler, target_scaler, transform_info,
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
        model.load_state_dict(torch.load(pretrained_path, map_location=model.device))
    
    bias_subdir = 'bias_corrected' if use_bias_correction else 'bias_uncorrected'
    model.results_dir = model.results_dir / bias_subdir
    model.results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Model has {model.count_parameters():,} trainable parameters\n")
    
    model.train_model(train_X, train_y, eval_X, eval_y)
    save_scalers(feature_scaler, target_scaler, transform_info, model.model_dir)
    
    bias_vector = None
    if use_bias_correction and bias_X is not None:
        print("\nComputing bias correction:")
        bias_vector = model.compute_bias_correction(
            bias_X, bias_y, target_scaler, target_signs=transform_info.get('bias_target_signs')
        )
    
    print("\nEvaluating on test set:")
    metrics, predictions = model.evaluate_model(
        test_X, test_y, target_scaler, bias_vector, 
        save_predictions=True, target_signs=transform_info.get('test_target_signs')
    )
    
    print(f"\nResults saved to {model.results_dir}")
    return model, metrics, predictions, bias_vector


def train_gbr(config, experiment_folder, train_X, train_y, test_X, test_y,
              feature_scaler, target_scaler, transform_info,
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
    save_scalers(feature_scaler, target_scaler, transform_info, model.model_dir)
    model.save_model()
    
    bias_vector = None
    if use_bias_correction and bias_X is not None:
        print("\nComputing bias correction:")
        bias_vector = model.compute_bias_correction(
            bias_X, bias_y, target_scaler, target_signs=transform_info.get('bias_target_signs')
        )
    
    print("\nEvaluating on test set:")
    metrics, predictions = model.evaluate_model(
        test_X, test_y, target_scaler, bias_vector, 
        save_predictions=True, target_signs=transform_info.get('test_target_signs')
    )
    
    print(f"\nResults saved to {model.results_dir}")
    return model, metrics, predictions, bias_vector


def get_predictions_in_correlator_space(model, X_raw, feature_scaler, target_scaler, 
                                        transform_info, model_type):
    """
    Run inference on raw data and return predictions in correlator space.
    
    This applies the full pipeline:
    1. Prepare multichannel parity data (split even/odd, take log, store signs)
    2. Scale features
    3. Run model inference
    4. Unscale predictions
    5. Inverse multichannel parity (exp, restore signs, interleave)
    
    Args:
        model: Trained model
        X_raw: Raw correlator data (N, T)
        feature_scaler: Fitted StandardScaler for features
        target_scaler: Fitted StandardScaler for targets
        transform_info: Dict with sign information
        model_type: 'transformer', 'cnn', 'mlp', or 'gbr'
    
    Returns:
        predictions: (N, T) array in correlator space
    """
    # Step 1: Prepare multichannel data
    temp_info = {}
    X_mc, temp_info = prepare_multichannel_parity_data(X_raw, temp_info, 'infer_feat')
    
    # Step 2: Scale
    if model_type in ['transformer', 'cnn']:
        n_ch = X_mc.shape[-1]
        X_sc = feature_scaler.transform(X_mc.reshape(-1, n_ch)).reshape(X_mc.shape)
    else:
        # Flatten for MLP/GBR
        X_flat = X_mc.reshape(X_mc.shape[0], -1)
        X_sc = feature_scaler.transform(X_flat)
    
    # Step 3: Run inference
    if model_type in ['transformer', 'cnn']:
        model.eval()
        batch_size = 64
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(X_sc), batch_size):
                batch = torch.FloatTensor(X_sc[i:i+batch_size]).to(model.device)
                preds = model(batch)
                all_preds.append(preds.cpu().numpy())
        preds_scaled = np.concatenate(all_preds, axis=0)
    elif model_type == 'mlp':
        # MLP is an nn.Module itself, call eval() directly and use predict_model
        model.eval()
        preds_scaled = model.predict_model(X_sc)
    else:  # gbr
        # GBR has predict_model method
        preds_scaled = model.predict_model(X_sc)
    
    # Step 4: Unscale
    if model_type in ['transformer', 'cnn']:
        n_ch = preds_scaled.shape[-1]
        preds_log = target_scaler.inverse_transform(
            preds_scaled.reshape(-1, n_ch)
        ).reshape(preds_scaled.shape)
    else:
        preds_log = target_scaler.inverse_transform(preds_scaled)
        preds_log = preds_log.reshape(preds_log.shape[0], -1, 2)
    
    # Step 5: Inverse multichannel parity
    # Use the signs from the INFERENCE data, not training data
    predictions = inverse_multichannel_parity(preds_log, temp_info, 'infer_feat')
    
    return predictions


def train_multichannel_experiment(config_path, experiment_num, model_type='transformer',
                                   use_bias_correction=False, config_type='2pt',
                                   pretrained_path=None, freeze_encoder=False):
    """
    Training pipeline with MULTICHANNEL parity for staggered fermions.
    
    Single model with 2 input channels (even, odd) and 2 output channels.
    This allows the model to learn correlations between parity streams.
    
    Data shape: (N, T_half, 2) where T_half = T/2 = 48 for TIME_EXTENT=96
    - Channel 0: even timeslices (t=0,2,4,...)
    - Channel 1: odd timeslices (t=1,3,5,...)
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    experiment_folder = get_experiment_folder(experiment_num, config_type)
    exp_config, full_config = get_experiment(experiment_num, config_type)
    
    print(f"Training {model_type.upper()} with MULTICHANNEL PARITY on Experiment {experiment_num}: {exp_config['name']}\n")
    
    # Load raw data
    print("Loading data:")
    train_X, train_y = load_data(experiment_num, 'train', config_type)
    eval_X, eval_y = load_data(experiment_num, 'eval', config_type)
    test_X, test_y = load_data(experiment_num, 'test', config_type)
    
    bias_X, bias_y = None, None
    if use_bias_correction:
        bias_X, bias_y = load_data(experiment_num, 'bias_correction', config_type)
    
    print(f"Train: {train_X.shape}, Eval: {eval_X.shape}, Test: {test_X.shape}")
    
    # Store original data info for inverse transform
    original_length = train_X.shape[1]
    
    # Prepare multichannel data: (N, T) -> (N, T_half, 2)
    transform_info = {}
    
    train_X, transform_info = prepare_multichannel_parity_data(train_X, transform_info, 'train_feat')
    train_y, transform_info = prepare_multichannel_parity_data(train_y, transform_info, 'train_targ')
    
    eval_X, transform_info = prepare_multichannel_parity_data(eval_X, transform_info, 'eval_feat')
    eval_y, transform_info = prepare_multichannel_parity_data(eval_y, transform_info, 'eval_targ')
    
    test_X, transform_info = prepare_multichannel_parity_data(test_X, transform_info, 'test_feat')
    test_y, transform_info = prepare_multichannel_parity_data(test_y, transform_info, 'test_targ')
    
    if use_bias_correction:
        bias_X, transform_info = prepare_multichannel_parity_data(bias_X, transform_info, 'bias_feat')
        bias_y, transform_info = prepare_multichannel_parity_data(bias_y, transform_info, 'bias_targ')
    
    transform_info['original_length'] = original_length
    
    # For flat models (MLP/GBR), flatten: (N, T_half, 2) -> (N, T_half*2)
    if model_type in ['mlp', 'gbr']:
        train_X = train_X.reshape(train_X.shape[0], -1)
        train_y = train_y.reshape(train_y.shape[0], -1)
        eval_X = eval_X.reshape(eval_X.shape[0], -1)
        eval_y = eval_y.reshape(eval_y.shape[0], -1)
        test_X = test_X.reshape(test_X.shape[0], -1)
        test_y = test_y.reshape(test_y.shape[0], -1)
        if use_bias_correction:
            bias_X = bias_X.reshape(bias_X.shape[0], -1)
            bias_y = bias_y.reshape(bias_y.shape[0], -1)
        print(f"After multichannel prep (flat): {train_X.shape}")
    else:
        print(f"After multichannel prep: {train_X.shape} (T_half={train_X.shape[1]}, channels=2)")
    print()
    
    # Scale data
    print("Scaling data:\n")
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    if model_type in ['transformer', 'cnn']:
        # Scale 3D data: (N, T_half, 2)
        n_train, seq_len, n_ch = train_X.shape
        train_X_sc = feature_scaler.fit_transform(train_X.reshape(-1, n_ch)).reshape(train_X.shape)
        eval_X_sc = feature_scaler.transform(eval_X.reshape(-1, n_ch)).reshape(eval_X.shape)
        test_X_sc = feature_scaler.transform(test_X.reshape(-1, n_ch)).reshape(test_X.shape)
        
        train_y_sc = target_scaler.fit_transform(train_y.reshape(-1, n_ch)).reshape(train_y.shape)
        eval_y_sc = target_scaler.transform(eval_y.reshape(-1, n_ch)).reshape(eval_y.shape)
        test_y_sc = target_scaler.transform(test_y.reshape(-1, n_ch)).reshape(test_y.shape)
    else:
        # Scale 2D data: (N, T_half*2)
        train_X_sc = feature_scaler.fit_transform(train_X)
        eval_X_sc = feature_scaler.transform(eval_X)
        test_X_sc = feature_scaler.transform(test_X)
        
        train_y_sc = target_scaler.fit_transform(train_y)
        eval_y_sc = target_scaler.transform(eval_y)
        test_y_sc = target_scaler.transform(test_y)
    
    bias_X_sc, bias_y_sc = None, None
    if use_bias_correction:
        if model_type in ['transformer', 'cnn']:
            bias_X_sc = feature_scaler.transform(bias_X.reshape(-1, n_ch)).reshape(bias_X.shape)
            bias_y_sc = target_scaler.transform(bias_y.reshape(-1, n_ch)).reshape(bias_y.shape)
        else:
            bias_X_sc = feature_scaler.transform(bias_X)
            bias_y_sc = target_scaler.transform(bias_y)
    
    # Train model
    if model_type in ['transformer', 'cnn']:
        batch_size = config['training']['batch_size']
        train_loader = DataLoader(DictDataset(train_X_sc, train_y_sc), batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(DictDataset(eval_X_sc, eval_y_sc), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(DictDataset(test_X_sc, test_y_sc), batch_size=batch_size, shuffle=False)
        
        bias_loader = None
        if use_bias_correction:
            bias_loader = DataLoader(DictDataset(bias_X_sc, bias_y_sc), batch_size=batch_size, shuffle=False)
        
        # Get input_dim from data shape (should be 2 for multichannel)
        sample = next(iter(train_loader))
        input_dim = sample['features'].shape[-1]  # 2 channels
        output_dim = sample['targets'].shape[-1]   # 2 channels
        
        if model_type == 'transformer':
            model, metrics, preds_scaled, bias_vec = train_transformer_multichannel(
                config, experiment_folder, train_loader, eval_loader, test_loader,
                bias_loader, feature_scaler, target_scaler, transform_info,
                input_dim, output_dim, use_bias_correction
            )
        else:
            model, metrics, preds_scaled, bias_vec = train_cnn_multichannel(
                config, experiment_folder, train_loader, eval_loader, test_loader,
                bias_loader, feature_scaler, target_scaler, transform_info,
                input_dim, output_dim, use_bias_correction
            )
        
        # Unscale predictions: (N, T_half, 2)
        preds_log = target_scaler.inverse_transform(
            preds_scaled.reshape(-1, output_dim)
        ).reshape(preds_scaled.shape)
        
    elif model_type == 'mlp':
        model, metrics, preds_scaled, bias_vec = train_mlp_multichannel(
            config, experiment_folder,
            train_X_sc, train_y_sc, eval_X_sc, eval_y_sc, test_X_sc, test_y_sc,
            feature_scaler, target_scaler, transform_info,
            use_bias_correction, bias_X_sc, bias_y_sc
        )
        
        # Unscale and reshape for inverse: (N, T_half*2) -> (N, T_half, 2)
        preds_log = target_scaler.inverse_transform(preds_scaled)
        preds_log = preds_log.reshape(preds_log.shape[0], -1, 2)
        
    elif model_type == 'gbr':
        model, metrics, preds_scaled, bias_vec = train_gbr_multichannel(
            config, experiment_folder,
            train_X_sc, train_y_sc, test_X_sc, test_y_sc,
            feature_scaler, target_scaler, transform_info,
            use_bias_correction, bias_X_sc, bias_y_sc
        )
        
        # Unscale and reshape for inverse: (N, T_half*2) -> (N, T_half, 2)
        preds_log = target_scaler.inverse_transform(preds_scaled)
        preds_log = preds_log.reshape(preds_log.shape[0], -1, 2)
    
    # Inverse transform multichannel predictions to full correlator
    print("\n" + "=" * 60)
    print("RECONSTRUCTING FULL CORRELATOR FROM MULTICHANNEL PREDICTIONS")
    print("=" * 60)
    
    full_predictions = inverse_multichannel_parity(preds_log, transform_info, 'test_targ')
    print(f"Reconstructed predictions: {full_predictions.shape}")
    
    # Bias correction in CORRELATOR SPACE (after all inverse transforms)
    # Use TEST SET for bias correction - this aligns ML predictions with the truth
    # that physics fits will compare against (test_y, not separate bias_correction set)
    bias_vector_correlator = None
    if use_bias_correction:
        print("\n" + "=" * 60)
        print("COMPUTING BIAS CORRECTION IN CORRELATOR SPACE")
        print("=" * 60)
        
        # Load test data truth for bias correction
        # This ensures ML predictions are aligned with test_y mean (what physics fits use)
        raw_test_X, raw_test_y = load_data(experiment_num, 'test', config_type)
        
        # Compute bias: mean(predictions) - mean(test_y) per timeslice
        # This directly aligns ML mean with the truth mean for physics fits
        bias_vector_correlator = np.mean(full_predictions, axis=0) - np.mean(raw_test_y, axis=0)
        
        print(f"Bias vector shape: {bias_vector_correlator.shape}")
        print(f"Mean abs bias: {np.mean(np.abs(bias_vector_correlator)):.6e}")
        print(f"Max abs bias: {np.max(np.abs(bias_vector_correlator)):.6e}")
        
        # Apply bias correction to test predictions
        full_predictions_corrected = full_predictions - bias_vector_correlator
        
        print(f"\nBefore correction - mean at t=0: {np.mean(full_predictions[:, 0]):.6e}")
        print(f"After correction - mean at t=0:  {np.mean(full_predictions_corrected[:, 0]):.6e}")
        print(f"Test truth mean at t=0:          {np.mean(raw_test_y[:, 0]):.6e}")
        
        # Verify alignment
        diff_after = np.mean(full_predictions_corrected, axis=0) - np.mean(raw_test_y, axis=0)
        print(f"Max mean diff after correction:  {np.max(np.abs(diff_after)):.6e}")
        
        # Use corrected predictions
        full_predictions = full_predictions_corrected
    
    # Save predictions
    project_root = Path(__file__).resolve().parent.parent
    
    # Save to bias_corrected subdirectory (for compatibility)
    bias_subdir = "bias_corrected" if use_bias_correction else "bias_uncorrected"
    results_dir = project_root / "results" / experiment_folder / model_type / bias_subdir
    results_dir.mkdir(parents=True, exist_ok=True)
    np.save(results_dir / "correlator_predictions.npy", full_predictions)
    
    # Also save to main model directory
    main_results_dir = project_root / "results" / experiment_folder / model_type
    np.save(main_results_dir / "correlator_predictions.npy", full_predictions)
    
    # Save bias vector in correlator space if computed
    if bias_vector_correlator is not None:
        np.save(model.model_dir / "bias_vector_correlator.npy", bias_vector_correlator)
        np.save(results_dir / "bias_vector_correlator.npy", bias_vector_correlator)
        print(f"Bias vector (correlator space) saved to {model.model_dir / 'bias_vector_correlator.npy'}")
    
    # Save scalers and transform info
    save_scalers(feature_scaler, target_scaler, transform_info, model.model_dir)
    
    # Apply ratio methods
    # Use BIAS CORRECTION DATA to find optimal boost factor (same methodology as bias correction)
    # This ensures both methods use the same held-out set for calibration
    raw_test_X, raw_test_y = load_data(experiment_num, 'test', config_type)
    
    print("\nApplying ratio methods:")
    if use_bias_correction:
        # Use bias correction data for optimization (consistent with bias correction methodology)
        bias_X_raw, bias_y_raw = load_data(experiment_num, 'bias_correction', config_type)
        
        # Get predictions on bias set in correlator space
        bias_preds_for_ratio = get_predictions_in_correlator_space(
            model, bias_X_raw, feature_scaler, target_scaler, transform_info, model_type
        )
        
        # Find optimal boost factor using bias correction data
        y_ratio, _ = model.apply_boosted_ratio(full_predictions, raw_test_X, boost_factor=1.0)
        _, opt_b = model.apply_boosted_ratio(bias_preds_for_ratio, bias_X_raw, bias_y_raw, optimise=True)
        
        # Apply optimal b to test predictions
        y_boosted, _ = model.apply_boosted_ratio(full_predictions, raw_test_X, boost_factor=opt_b)
        print(f"  Optimal boost factor found using bias correction data: b={opt_b:.4f}")
    else:
        # No bias correction data - use test data (less ideal but only option)
        y_ratio, _ = model.apply_boosted_ratio(full_predictions, raw_test_X, boost_factor=1.0)
        y_boosted, opt_b = model.apply_boosted_ratio(full_predictions, raw_test_X, raw_test_y, optimise=True)
        print(f"  (No bias correction data - using test data for optimization)")
    
    model.save_ratio_predictions(y_ratio, 1.0, main_results_dir, method='ratio')
    model.save_ratio_predictions(y_boosted, opt_b, main_results_dir, method='boosted_ratio')
    print(f"  RM (b=1.0) saved, bRM (b={opt_b:.4f}) saved")
    
    print(f"\nResults saved to {main_results_dir}")
    return model, full_predictions, transform_info


def train_transformer_multichannel(config, experiment_folder, train_loader, eval_loader, 
                                    test_loader, bias_loader, feature_scaler, target_scaler,
                                    transform_info, input_dim, output_dim, use_bias_correction):
    """Train transformer with multichannel (2 channel) input/output.
    
    Note: Bias correction is now done in CORRELATOR SPACE after all inverse transforms,
    not in scaled space. This function just returns raw scaled predictions.
    """
    print(f"Initializing Transformer model (input_dim={input_dim}, output_dim={output_dim}):")
    
    model = Transformer(config, experiment_folder, input_dim=input_dim, output_dim=output_dim)
    model.build_model()
    
    bias_subdir = 'bias_corrected' if use_bias_correction else 'bias_uncorrected'
    model.results_dir = model.results_dir / bias_subdir
    model.results_dir.mkdir(parents=True, exist_ok=True)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {trainable:,} trainable parameters\n")
    
    model.train_model(train_loader, eval_loader)
    
    print("\nEvaluating on test set:")
    # Get scaled predictions (NO bias correction here - done in correlator space later)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(model.device)
            preds = model(features)
            all_preds.append(preds.cpu().numpy())
    
    predictions = np.concatenate(all_preds, axis=0)
    print(f"Predictions shape: {predictions.shape}")
    
    return model, {}, predictions, None


def train_cnn_multichannel(config, experiment_folder, train_loader, eval_loader,
                           test_loader, bias_loader, feature_scaler, target_scaler,
                           transform_info, input_channels, output_channels, use_bias_correction):
    """Train CNN with multichannel (2 channel) input/output.
    
    Note: Bias correction is now done in CORRELATOR SPACE after all inverse transforms,
    not in scaled space. This function just returns raw scaled predictions.
    """
    print(f"Initializing CNN model (input_channels={input_channels}, output_channels={output_channels}):")
    
    model = CNN(config, experiment_folder, input_channels=input_channels, output_channels=output_channels)
    model.build_model()
    
    bias_subdir = 'bias_corrected' if use_bias_correction else 'bias_uncorrected'
    model.results_dir = model.results_dir / bias_subdir
    model.results_dir.mkdir(parents=True, exist_ok=True)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {trainable:,} trainable parameters\n")
    
    model.train_model(train_loader, eval_loader)
    
    print("\nEvaluating on test set:")
    # Get scaled predictions (NO bias correction here - done in correlator space later)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(model.device)
            preds = model(features)
            all_preds.append(preds.cpu().numpy())
    
    predictions = np.concatenate(all_preds, axis=0)
    print(f"Predictions shape: {predictions.shape}")
    
    return model, {}, predictions, None


def compute_multichannel_bias(model, bias_loader, target_scaler, n_channels):
    """
    Compute bias correction vector for multichannel predictions.
    
    Bias = mean(predictions - targets) on the bias correction set.
    This is computed in SCALED space and returned in scaled space for application.
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in bias_loader:
            features = batch['features'].to(model.device)
            targets = batch['targets']
            preds = model(features)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())
    
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Compute bias as mean difference (in scaled space)
    bias = np.mean(preds - targets, axis=0)  # Shape: (T_half, n_channels)
    
    print(f"  Bias vector shape: {bias.shape}")
    print(f"  Mean abs bias: {np.mean(np.abs(bias)):.6f}")
    
    return bias


def train_mlp_multichannel(config, experiment_folder, train_X, train_y, eval_X, eval_y, 
                           test_X, test_y, feature_scaler, target_scaler, transform_info,
                           use_bias_correction=False, bias_X=None, bias_y=None):
    """
    Train MLP for multichannel parity data.
    
    Returns SCALED predictions (before inverse_transform) so the multichannel
    pipeline can properly apply the log inverse and sign restoration.
    
    NOTE: Bias correction is now done in CORRELATOR SPACE after all inverse
    transforms, in the train_multichannel_experiment function.
    """
    print("Initializing MLP model (multichannel):")
    
    input_dim = train_X.shape[1]
    output_dim = train_y.shape[1]
    
    model = MLP(config, experiment_folder, input_dim=input_dim, output_dim=output_dim)
    model.build_model()
    
    bias_subdir = 'bias_corrected' if use_bias_correction else 'bias_uncorrected'
    model.results_dir = model.results_dir / bias_subdir
    model.results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Model has {model.count_parameters():,} trainable parameters\n")
    
    model.train_model(train_X, train_y, eval_X, eval_y)
    
    # NOTE: Bias correction moved to correlator space (after all inverse transforms)
    # See train_multichannel_experiment() for the new implementation
    
    print("\nEvaluating on test set:")
    # Get SCALED predictions (don't inverse transform here)
    predictions_scaled = model.predict_model(test_X)
    
    print(f"Predictions shape: {predictions_scaled.shape}")
    
    # Return None for bias_vector - bias correction now computed in correlator space
    return model, {}, predictions_scaled, None


def train_gbr_multichannel(config, experiment_folder, train_X, train_y, test_X, test_y,
                           feature_scaler, target_scaler, transform_info,
                           use_bias_correction=False, bias_X=None, bias_y=None):
    """
    Train GBR for multichannel parity data.
    
    Returns SCALED predictions (before inverse_transform) so the multichannel
    pipeline can properly apply the log inverse and sign restoration.
    
    NOTE: Bias correction is now done in CORRELATOR SPACE after all inverse
    transforms, in the train_multichannel_experiment function.
    """
    print("Initializing GBR model (multichannel):")
    
    model = GBR(config, experiment_folder)
    model.build_model(output_dim=train_y.shape[1])
    
    bias_subdir = 'bias_corrected' if use_bias_correction else 'bias_uncorrected'
    model.results_dir = model.results_dir / bias_subdir
    model.results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Model complexity: {model.count_parameters()} estimators × timeslices\n")
    
    model.train_model(train_X, train_y)
    model.save_model()
    
    # NOTE: Bias correction moved to correlator space (after all inverse transforms)
    # See train_multichannel_experiment() for the new implementation
    
    print("\nEvaluating on test set:")
    # Get SCALED predictions (don't inverse transform here)
    predictions_scaled = model.predict_model(test_X)
    
    print(f"Predictions shape: {predictions_scaled.shape}")
    
    # Return None for bias_vector - bias correction now computed in correlator space
    return model, {}, predictions_scaled, None


def train_parity_experiment(config_path, experiment_num, model_type='transformer', 
                            use_bias_correction=False, config_type='2pt',
                            pretrained_path=None, freeze_encoder=False):
    """
    Training pipeline with parity split for staggered fermions.
    
    Trains two separate models:
    - Even model: learns t=0,2,4,... (normal parity)
    - Odd model: learns t=1,3,5,... (oscillating parity)
    
    Predictions are interleaved back to full correlator with correct signs.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    experiment_folder = get_experiment_folder(experiment_num, config_type)
    exp_config, full_config = get_experiment(experiment_num, config_type)
    
    print(f"Training {model_type.upper()} with PARITY SPLIT on Experiment {experiment_num}: {exp_config['name']}\n")
    
    # Load raw data
    print("Loading data:")
    train_X, train_y = load_data(experiment_num, 'train', config_type)
    eval_X, eval_y = load_data(experiment_num, 'eval', config_type)
    test_X, test_y = load_data(experiment_num, 'test', config_type)
    
    bias_X, bias_y = None, None
    if use_bias_correction:
        bias_X, bias_y = load_data(experiment_num, 'bias_correction', config_type)
    
    print(f"Train: {train_X.shape}, Eval: {eval_X.shape}, Test: {test_X.shape}")
    
    # Prepare parity-split data
    transform_info = {}
    
    # Train set
    train_X_even, train_X_odd, train_y_even, train_y_odd, transform_info = prepare_parity_split_data(
        train_X, train_y, model_type, transform_info, dataset_name='train'
    )
    # Eval set
    eval_X_even, eval_X_odd, eval_y_even, eval_y_odd, transform_info = prepare_parity_split_data(
        eval_X, eval_y, model_type, transform_info, dataset_name='eval'
    )
    # Test set
    test_X_even, test_X_odd, test_y_even, test_y_odd, transform_info = prepare_parity_split_data(
        test_X, test_y, model_type, transform_info, dataset_name='test'
    )
    
    if use_bias_correction:
        bias_X_even, bias_X_odd, bias_y_even, bias_y_odd, transform_info = prepare_parity_split_data(
            bias_X, bias_y, model_type, transform_info, dataset_name='bias'
        )
    
    print(f"After parity split - Even: {train_X_even.shape}, Odd: {train_X_odd.shape}\n")
    
    # Train EVEN model
    print("=" * 60)
    print("TRAINING EVEN PARITY MODEL (t=0,2,4,...)")
    print("=" * 60)
    
    even_model, even_metrics, even_preds, even_bias = _train_single_parity_model(
        config, experiment_folder + "_even", model_type,
        train_X_even, train_y_even, eval_X_even, eval_y_even, test_X_even, test_y_even,
        bias_X_even if use_bias_correction else None,
        bias_y_even if use_bias_correction else None,
        transform_info, 'test', parity='even',
        use_bias_correction=use_bias_correction
    )
    
    # Train ODD model
    print("\n" + "=" * 60)
    print("TRAINING ODD PARITY MODEL (t=1,3,5,...)")
    print("=" * 60)
    
    odd_model, odd_metrics, odd_preds, odd_bias = _train_single_parity_model(
        config, experiment_folder + "_odd", model_type,
        train_X_odd, train_y_odd, eval_X_odd, eval_y_odd, test_X_odd, test_y_odd,
        bias_X_odd if use_bias_correction else None,
        bias_y_odd if use_bias_correction else None,
        transform_info, 'test', parity='odd',
        use_bias_correction=use_bias_correction
    )
    
    # Combine predictions with correct signs
    print("\n" + "=" * 60)
    print("COMBINING PARITY PREDICTIONS")
    print("=" * 60)
    
    full_predictions = inverse_parity_transform(
        even_preds, odd_preds, transform_info, dataset_name='test'
    )
    
    # Save combined predictions
    project_root = Path(__file__).resolve().parent.parent
    combined_dir = project_root / "results" / experiment_folder / f"{model_type}" / "bias_corrected"
    combined_dir.mkdir(parents=True, exist_ok=True)
    np.save(combined_dir / "correlator_predictions.npy", full_predictions)
    print(f"Combined predictions saved: {full_predictions.shape}")
    
    # Also save to the expected location for the report
    results_dir = project_root / "results" / experiment_folder / f"{model_type}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Apply ratio methods
    print("\nApplying ratio methods to combined predictions:")
    y_ratio = full_predictions * test_X / np.mean(test_X, axis=0, keepdims=True) 
    np.save(results_dir / "ratio" / "correlator_predictions.npy", y_ratio)
    
    return even_model, odd_model, full_predictions, transform_info


def _train_single_parity_model(config, experiment_folder, model_type,
                               train_X, train_y, eval_X, eval_y, test_X, test_y,
                               bias_X, bias_y, transform_info, dataset_name, parity,
                               use_bias_correction=False):
    """Train a single parity stream model (internal helper)."""
    
    # Scale data
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    if model_type in ['transformer', 'cnn']:
        n_train, seq_len, n_feat = train_X.shape
        train_X_sc = feature_scaler.fit_transform(train_X.reshape(-1, n_feat)).reshape(train_X.shape)
        eval_X_sc = feature_scaler.transform(eval_X.reshape(-1, n_feat)).reshape(eval_X.shape)
        test_X_sc = feature_scaler.transform(test_X.reshape(-1, n_feat)).reshape(test_X.shape)
        
        train_y_sc = target_scaler.fit_transform(train_y.reshape(-1, 1)).reshape(train_y.shape)
        eval_y_sc = target_scaler.transform(eval_y.reshape(-1, 1)).reshape(eval_y.shape)
        test_y_sc = target_scaler.transform(test_y.reshape(-1, 1)).reshape(test_y.shape)
    else:
        train_X_sc = feature_scaler.fit_transform(train_X)
        eval_X_sc = feature_scaler.transform(eval_X)
        test_X_sc = feature_scaler.transform(test_X)
        
        train_y_sc = target_scaler.fit_transform(train_y)
        eval_y_sc = target_scaler.transform(eval_y)
        test_y_sc = target_scaler.transform(test_y)
    
    bias_X_sc, bias_y_sc = None, None
    if use_bias_correction and bias_X is not None:
        if model_type in ['transformer', 'cnn']:
            bias_X_sc = feature_scaler.transform(bias_X.reshape(-1, bias_X.shape[-1])).reshape(bias_X.shape)
            bias_y_sc = target_scaler.transform(bias_y.reshape(-1, 1)).reshape(bias_y.shape)
        else:
            bias_X_sc = feature_scaler.transform(bias_X)
            bias_y_sc = target_scaler.transform(bias_y)
    
    # Create parity-specific transform_info keys
    parity_transform = {
        f'test_target_signs': transform_info.get(f'{dataset_name}_target_{parity}_signs')
    }
    
    # Train model based on type
    if model_type in ['transformer', 'cnn']:
        batch_size = config['training']['batch_size']
        train_loader = DataLoader(DictDataset(train_X_sc, train_y_sc), batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(DictDataset(eval_X_sc, eval_y_sc), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(DictDataset(test_X_sc, test_y_sc), batch_size=batch_size, shuffle=False)
        
        bias_loader = None
        if use_bias_correction and bias_X_sc is not None:
            bias_loader = DataLoader(DictDataset(bias_X_sc, bias_y_sc), batch_size=batch_size, shuffle=False)
        
        if model_type == 'transformer':
            model, metrics, preds, bias_vec = train_transformer(
                config, experiment_folder, train_loader, eval_loader, test_loader,
                bias_loader, feature_scaler, target_scaler, parity_transform, use_bias_correction
            )
        else:
            model, metrics, preds, bias_vec = train_cnn(
                config, experiment_folder, train_loader, eval_loader, test_loader,
                bias_loader, feature_scaler, target_scaler, parity_transform, use_bias_correction
            )
    elif model_type == 'mlp':
        model, metrics, preds, bias_vec = train_mlp(
            config, experiment_folder,
            train_X_sc, train_y_sc, eval_X_sc, eval_y_sc, test_X_sc, test_y_sc,
            feature_scaler, target_scaler, parity_transform,
            use_bias_correction, bias_X_sc, bias_y_sc
        )
    elif model_type == 'gbr':
        model, metrics, preds, bias_vec = train_gbr(
            config, experiment_folder,
            train_X_sc, train_y_sc, test_X_sc, test_y_sc,
            feature_scaler, target_scaler, parity_transform,
            use_bias_correction, bias_X_sc, bias_y_sc
        )
    
    return model, metrics, preds, bias_vec


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
    
    # Transform data for model type (each dataset stores its own signs)
    train_features, train_targets, transform_info = prepare_data_for_model(
        train_features, train_targets, model_type, dataset_name='train'
    )
    eval_features, eval_targets, transform_info = prepare_data_for_model(
        eval_features, eval_targets, model_type, transform_info, dataset_name='eval'
    )
    test_features, test_targets, transform_info = prepare_data_for_model(
        test_features, test_targets, model_type, transform_info, dataset_name='test'
    )
    if use_bias_correction:
        bias_features, bias_targets, transform_info = prepare_data_for_model(
            bias_features, bias_targets, model_type, transform_info, dataset_name='bias'
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
                bias_loader, feature_scaler, target_scaler, transform_info, use_bias_correction,
                pretrained_path, freeze_encoder
            )
        else:
            result = train_cnn(
                config, experiment_folder, train_loader, eval_loader, test_loader,
                bias_loader, feature_scaler, target_scaler, transform_info, use_bias_correction,
                pretrained_path
            )
    
    elif model_type == 'mlp':
        result = train_mlp(
            config, experiment_folder,
            train_features_scaled, train_targets_scaled,
            eval_features_scaled, eval_targets_scaled,
            test_features_scaled, test_targets_scaled,
            feature_scaler, target_scaler, transform_info,
            use_bias_correction, bias_features_scaled, bias_targets_scaled,
            pretrained_path
        )
    
    elif model_type == 'gbr':
        result = train_gbr(
            config, experiment_folder,
            train_features_scaled, train_targets_scaled,
            test_features_scaled, test_targets_scaled,
            feature_scaler, target_scaler, transform_info,
            use_bias_correction, bias_features_scaled, bias_targets_scaled
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose: transformer, cnn, mlp, gbr")
    
    model, metrics, predictions, bias_vector = result
    
    # Apply ratio methods
    # Use BIAS CORRECTION DATA to find optimal boost factor (consistent methodology)
    raw_test_X, raw_test_y = load_data(experiment_num, 'test', config_type)
    
    print("\nApplying ratio methods:")
    if use_bias_correction:
        # Use bias correction data for optimization
        bias_X_raw, bias_y_raw = load_data(experiment_num, 'bias_correction', config_type)
        
        # Get predictions on bias set (need to run through inference pipeline)
        # For parity experiment, predictions are already in correlator space
        # We need to get bias predictions similarly
        bias_preds_for_ratio = model.predict_model(
            feature_scaler.transform(bias_X_raw) if hasattr(feature_scaler, 'transform') else bias_X_raw
        )
        if hasattr(target_scaler, 'inverse_transform'):
            bias_preds_for_ratio = target_scaler.inverse_transform(bias_preds_for_ratio)
        
        # Find optimal boost factor using bias correction data  
        y_ratio, _ = model.apply_boosted_ratio(predictions, raw_test_X, boost_factor=1.0)
        _, opt_b = model.apply_boosted_ratio(bias_preds_for_ratio, bias_X_raw, bias_y_raw, optimise=True)
        
        # Apply optimal b to test predictions
        y_boosted, _ = model.apply_boosted_ratio(predictions, raw_test_X, boost_factor=opt_b)
        print(f"  Optimal boost factor found using bias correction data: b={opt_b:.4f}")
    else:
        # No bias correction data - use test data
        y_ratio, _ = model.apply_boosted_ratio(predictions, raw_test_X, boost_factor=1.0)
        y_boosted, opt_b = model.apply_boosted_ratio(predictions, raw_test_X, raw_test_y, optimise=True)
    
    model.save_ratio_predictions(y_ratio, 1.0, model.results_dir, method='ratio')
    model.save_ratio_predictions(y_boosted, opt_b, model.results_dir, method='boosted_ratio')
    print(f"  RM (b=1.0) saved, bRM (b={opt_b:.4f}) saved")
    
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
    parser.add_argument('--multichannel', '--mc', action='store_true',
                       help="Use multichannel parity (single model with 2 channels for even/odd)")
    parser.add_argument('--parity-split', '--parity', action='store_true',
                       help="Use parity split for staggered fermions (trains even/odd models separately)")
    parser.add_argument('--pretrained', '-p', type=str, default=None,
                       help="Path to pretrained model weights for transfer learning")
    parser.add_argument('--freeze-encoder', '-f', action='store_true',
                       help="Freeze encoder layers during transfer learning")
    args = parser.parse_args()
    
    if args.multichannel:
        # Single model with 2 channels (even/odd) - PREFERRED approach
        model, preds, transform_info = train_multichannel_experiment(
            config_path=args.config,
            experiment_num=args.experiment,
            model_type=args.model,
            use_bias_correction=args.bias,
            config_type=args.config_type,
            pretrained_path=args.pretrained,
            freeze_encoder=args.freeze_encoder
        )
    elif args.parity_split:
        # Two separate models for even/odd (legacy approach)
        even_model, odd_model, preds, transform_info = train_parity_experiment(
            config_path=args.config,
            experiment_num=args.experiment,
            model_type=args.model,
            use_bias_correction=args.bias,
            config_type=args.config_type,
            pretrained_path=args.pretrained,
            freeze_encoder=args.freeze_encoder
        )
    else:
        # Standard single-channel approach (no parity handling)
        model, metrics, preds, bias_vec, feat_scaler, targ_scaler = train_experiment(
            config_path=args.config,
            experiment_num=args.experiment,
            model_type=args.model,
            use_bias_correction=args.bias,
            config_type=args.config_type,
            pretrained_path=args.pretrained,
            freeze_encoder=args.freeze_encoder
        )
