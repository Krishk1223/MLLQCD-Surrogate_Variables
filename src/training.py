"""Streamlined training pipeline for LQCD ML models.

Outputs required by physics_analysis.py:
- results/{experiment}/{model}/bias_corrected/correlator_predictions.npy
- results/{experiment}/{model}/ratio_predictions/correlator_predictions.npy
"""

import numpy as np
import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from src.model_scripts.transformer.transformer_model import Transformer
from src.model_scripts.cnn.cnn_model import CNN
from src.model_scripts.mlp.mlp_model import MLP
from src.model_scripts.gbr.gbr_model import GBR
from src.preprocessing.experiments import get_experiment, get_experiment_folder, load_experiment_data


def prepare_multichannel(data: np.ndarray, info: dict, prefix: str) -> tuple:
    """Convert (N, T) correlator to (N, T//2, 2) multichannel [even, odd]."""
    even, odd = data[:, 0::2], data[:, 1::2]
    info[f'{prefix}_even_signs'] = np.sign(even)
    info[f'{prefix}_odd_signs'] = np.sign(odd)
    
    even_log = np.log(np.abs(even) + 1e-30)
    odd_log = np.log(np.abs(odd) + 1e-30)
    
    min_len = min(even_log.shape[1], odd_log.shape[1])
    info[f'{prefix}_T_even'], info[f'{prefix}_T_odd'] = even.shape[1], odd.shape[1]
    
    return np.stack([even_log[:, :min_len], odd_log[:, :min_len]], axis=-1), info


def inverse_multichannel(preds: np.ndarray, info: dict, prefix: str) -> np.ndarray:
    """Convert (N, T//2, 2) predictions back to (N, T) correlator."""
    even_log, odd_log = preds[:, :, 0], preds[:, :, 1]
    T_min = even_log.shape[1]
    
    even_signs = info.get(f'{prefix}_even_signs', np.ones_like(even_log))[:, :T_min]
    odd_signs = info.get(f'{prefix}_odd_signs', np.ones_like(odd_log))[:, :T_min]
    
    even = np.exp(even_log) * even_signs
    odd = np.exp(odd_log) * odd_signs
    
    # Interleave back
    T_even, T_odd = info.get(f'{prefix}_T_even', even.shape[1]), info.get(f'{prefix}_T_odd', odd.shape[1])
    full = np.zeros((even.shape[0], T_even + T_odd))
    full[:, 0::2] = even[:, :T_even] if even.shape[1] >= T_even else np.pad(even, ((0,0), (0, T_even - even.shape[1])))
    full[:, 1::2] = odd[:, :T_odd] if odd.shape[1] >= T_odd else np.pad(odd, ((0,0), (0, T_odd - odd.shape[1])))
    
    return full


class DictDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X, self.y = torch.FloatTensor(X), torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return {'features': self.X[i], 'targets': self.y[i]}


def scale_3d(scaler: StandardScaler, data: np.ndarray, fit: bool = False) -> np.ndarray:
    """Scale (N, T, C) data using scaler fitted on flattened channels."""
    shape = data.shape
    flat = data.reshape(-1, shape[-1])
    scaled = scaler.fit_transform(flat) if fit else scaler.transform(flat)
    return scaled.reshape(shape)


def get_predictions(model, loader, device) -> np.ndarray:
    """Run inference on a DataLoader."""
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch['features'].to(device))
            preds.append(out.cpu().numpy())
    return np.concatenate(preds, axis=0)


def train_model(config_path: str, experiment_num: int, model_type: str = 'transformer',
                config_type: str = '2pt', bias_correction: bool = True):
    """
    Train a model with multichannel parity handling.
    
    Saves:
    - results/{exp}/{model}/bias_corrected/correlator_predictions.npy
    - results/{exp}/{model}/ratio_predictions/correlator_predictions.npy
    """
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    exp_folder = get_experiment_folder(experiment_num, config_type)
    exp_config, _ = get_experiment(experiment_num, config_type)
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} on {exp_config['name']}")
    print(f"{'='*60}\n")
    
    # Load data
    train_X, train_y = load_experiment_data(experiment_num, 'train', config_type)
    eval_X, eval_y = load_experiment_data(experiment_num, 'eval', config_type)
    test_X, test_y = load_experiment_data(experiment_num, 'test', config_type)
    bias_X, bias_y = load_experiment_data(experiment_num, 'bias_correction', config_type) if bias_correction else (None, None)
    
    print(f"Data: train={train_X.shape}, eval={eval_X.shape}, test={test_X.shape}")
    if bias_X is not None:
        print(f"      bias={bias_X.shape}")
    
    # Prepare multichannel parity data
    info = {}
    train_X, info = prepare_multichannel(train_X, info, 'train_X')
    train_y, info = prepare_multichannel(train_y, info, 'train_y')
    eval_X, info = prepare_multichannel(eval_X, info, 'eval_X')
    eval_y, info = prepare_multichannel(eval_y, info, 'eval_y')
    test_X_mc, info = prepare_multichannel(test_X, info, 'test_X')
    test_y_mc, info = prepare_multichannel(test_y, info, 'test_y')
    if bias_X is not None:
        bias_X_mc, info = prepare_multichannel(bias_X, info, 'bias_X')
        bias_y_mc, info = prepare_multichannel(bias_y, info, 'bias_y')
    
    # Flatten for MLP/GBR
    is_flat = model_type in ['mlp', 'gbr']
    if is_flat:
        train_X = train_X.reshape(train_X.shape[0], -1)
        train_y = train_y.reshape(train_y.shape[0], -1)
        eval_X = eval_X.reshape(eval_X.shape[0], -1)
        eval_y = eval_y.reshape(eval_y.shape[0], -1)
        test_X_mc = test_X_mc.reshape(test_X_mc.shape[0], -1)
        test_y_mc = test_y_mc.reshape(test_y_mc.shape[0], -1)
        if bias_X is not None:
            bias_X_mc = bias_X_mc.reshape(bias_X_mc.shape[0], -1)
            bias_y_mc = bias_y_mc.reshape(bias_y_mc.shape[0], -1)
    
    print(f"After multichannel: {train_X.shape}")
    
    # Scale
    feat_scaler, targ_scaler = StandardScaler(), StandardScaler()
    if is_flat:
        train_X_sc = feat_scaler.fit_transform(train_X)
        train_y_sc = targ_scaler.fit_transform(train_y)
        eval_X_sc, eval_y_sc = feat_scaler.transform(eval_X), targ_scaler.transform(eval_y)
        test_X_sc, test_y_sc = feat_scaler.transform(test_X_mc), targ_scaler.transform(test_y_mc)
        bias_X_sc = feat_scaler.transform(bias_X_mc) if bias_X is not None else None
    else:
        train_X_sc = scale_3d(feat_scaler, train_X, fit=True)
        train_y_sc = scale_3d(targ_scaler, train_y, fit=True)
        eval_X_sc, eval_y_sc = scale_3d(feat_scaler, eval_X), scale_3d(targ_scaler, eval_y)
        test_X_sc, test_y_sc = scale_3d(feat_scaler, test_X_mc), scale_3d(targ_scaler, test_y_mc)
        bias_X_sc = scale_3d(feat_scaler, bias_X_mc) if bias_X is not None else None
    
    # Build and train model
    batch_size = config['training'].get('batch_size', 64)
    
    if model_type == 'transformer':
        input_dim = train_X_sc.shape[-1]
        model = Transformer(config, exp_folder, input_dim=input_dim, output_dim=input_dim)
        model.build_model()
        train_loader = DataLoader(DictDataset(train_X_sc, train_y_sc), batch_size, shuffle=True)
        eval_loader = DataLoader(DictDataset(eval_X_sc, eval_y_sc), batch_size, shuffle=False)
        model.train_model(train_loader, eval_loader)
        test_loader = DataLoader(DictDataset(test_X_sc, test_y_sc), batch_size, shuffle=False)
        preds_sc = get_predictions(model, test_loader, model.device)
        
    elif model_type == 'cnn':
        input_ch = train_X_sc.shape[-1]
        model = CNN(config, exp_folder, input_channels=input_ch, output_channels=input_ch)
        model.build_model()
        train_loader = DataLoader(DictDataset(train_X_sc, train_y_sc), batch_size, shuffle=True)
        eval_loader = DataLoader(DictDataset(eval_X_sc, eval_y_sc), batch_size, shuffle=False)
        model.train_model(train_loader, eval_loader)
        test_loader = DataLoader(DictDataset(test_X_sc, test_y_sc), batch_size, shuffle=False)
        preds_sc = get_predictions(model, test_loader, model.device)
        
    elif model_type == 'mlp':
        model = MLP(config, exp_folder, input_dim=train_X_sc.shape[1], output_dim=train_y_sc.shape[1])
        model.build_model()
        model.train_model(train_X_sc, train_y_sc, eval_X_sc, eval_y_sc)
        preds_sc = model.predict_model(test_X_sc)
        
    elif model_type == 'gbr':
        model = GBR(config, exp_folder)
        model.build_model(output_dim=train_y_sc.shape[1])
        model.train_model(train_X_sc, train_y_sc)
        model.save_model()
        preds_sc = model.predict_model(test_X_sc)
    
    # Inverse transform to correlator space
    if is_flat:
        preds_log = targ_scaler.inverse_transform(preds_sc).reshape(preds_sc.shape[0], -1, 2)
    else:
        preds_log = targ_scaler.inverse_transform(preds_sc.reshape(-1, preds_sc.shape[-1])).reshape(preds_sc.shape)
    
    preds_corr = inverse_multichannel(preds_log, info, 'test_y')
    print(f"Predictions in correlator space: {preds_corr.shape}")
    
    # Bias correction (in correlator space)
    if bias_correction:
        bias_vec = np.mean(preds_corr, axis=0) - np.mean(test_y, axis=0)
        preds_bc = preds_corr - bias_vec
        print(f"Bias correction applied. Max |bias|: {np.max(np.abs(bias_vec)):.4e}")
    else:
        preds_bc = preds_corr
    
    # Ratio method
    preds_ratio = preds_bc * test_X / np.mean(test_X, axis=0, keepdims=True)
    
    # Save predictions
    root = Path(__file__).parent.parent
    bc_dir = root / 'results' / exp_folder / model_type / 'bias_corrected'
    rm_dir = root / 'results' / exp_folder / model_type / 'ratio_predictions'
    bc_dir.mkdir(parents=True, exist_ok=True)
    rm_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(bc_dir / 'correlator_predictions.npy', preds_bc)
    np.save(rm_dir / 'correlator_predictions.npy', preds_ratio)
    
    print(f"\nSaved to:")
    print(f"  {bc_dir / 'correlator_predictions.npy'}")
    print(f"  {rm_dir / 'correlator_predictions.npy'}")
    
    return model, preds_bc, preds_ratio


def train_all_models(experiment_num: int, config_type: str = '2pt'):
    """Train all models for an experiment."""
    configs = {
        'transformer': 'configs/transformer.yaml',
        'cnn': 'configs/cnn.yaml',
        'mlp': 'configs/mlp.yaml',
        'gbr': 'configs/gbr.yaml',
    }
    
    root = Path(__file__).parent.parent
    for model_type, config_path in configs.items():
        try:
            train_model(str(root / config_path), experiment_num, model_type, config_type)
        except Exception as e:
            print(f"ERROR training {model_type}: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train ML models on LQCD experiments")
    parser.add_argument('--experiment', '-e', type=int, default=1, help="Experiment number")
    parser.add_argument('--model', '-m', type=str, default='all', 
                       choices=['transformer', 'cnn', 'mlp', 'gbr', 'all'], help="Model type")
    parser.add_argument('--config', type=str, default=None, help="Path to model config YAML")
    parser.add_argument('--config-type', '-t', type=str, default='2pt', help="Experiment config type")
    args = parser.parse_args()
    
    if args.model == 'all':
        train_all_models(args.experiment, args.config_type)
    else:
        config = args.config or f'configs/{args.model}.yaml'
        root = Path(__file__).parent.parent
        train_model(str(root / config), args.experiment, args.model, args.config_type)
