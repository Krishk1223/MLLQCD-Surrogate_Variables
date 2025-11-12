import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import pickle
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sys

def setup_logging(enable_logging=False):
    """Setup logging configuration"""
    if not enable_logging:
        logging.disable(logging.CRITICAL)
        return
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    log_path = project_root / "logs"
    log_path.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"ml_preprocessing_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"ML preprocessing logging started. Log file: {log_file}")

class MLPreprocessor2pt:
    def __init__(self, processed_data_path: Path, results_path: Path, sanity_report_path: Path = None, enable_logging=False):
        """
        Initialize 2pt correlator preprocessor
        
        Args:
            processed_data_path: Path to processed CSV files
            results_path: Path to save ML-ready data
            sanity_report_path: Optional path to sanity check report
            enable_logging: Whether to enable logging
        """
        self.processed_data_path = Path(processed_data_path)
        self.results_path = Path(results_path)
        self.sanity_report_path = sanity_report_path
        
        # Create output directories
        self.ml_data_path = self.results_path / "ml_processed"
        self.ml_data_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        setup_logging(enable_logging)
        
        # Data storage
        self.ensemble_data = {}
        self.processed_data = {}
        self.train_data = None
        self.bias_correction_data = None
        self.evaluation_data = None
        self.test_data = None
        self.normalization_params = {}
        
        # Processing parameters
        self.tau_max = 30  # Maximum time slice
        self.epsilon = 1e-15  # Epsilon for log transformation
        
        # Data split ratios
        self.train_size = 0.20  # 20% train data
        self.bias_correction_size = 0.15  # 15% bias correction data
        self.evaluation_size = 0.10  # 10% model evaluation data
        self.test_size = 0.55  # 55% unlabelled/test data   
        
        self.normalization_method = 'standard'
        
        logging.info(f"Initialized 2pt preprocessor with τ_max={self.tau_max}")
        logging.info(f"Processing data from: {self.processed_data_path}")
        logging.info(f"Output will be saved to: {self.ml_data_path}")

    def load_processed_data(self):
        """Load all CSV files from processed directory"""
        csv_files = list(self.processed_data_path.glob("*.csv"))
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.processed_data_path}")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                ensemble_name = csv_file.stem
                
                # Validate data structure
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if 'config_id' in numeric_cols:
                    numeric_cols = numeric_cols.drop('config_id')
                
                if len(numeric_cols) == 0:
                    logging.warning(f"No numeric columns found in {csv_file.name}, skipping")
                    continue
                
                self.ensemble_data[ensemble_name] = df
                logging.info(f"Loaded {ensemble_name}: {len(df)} configs, {len(numeric_cols)} time slices")
                
            except Exception as e:
                logging.error(f"Failed to load {csv_file.name}: {e}")
        
        return len(self.ensemble_data) > 0

    def apply_truncation(self, data_array):
        """
        Apply truncation rules:
        1. Cap at τ=30
        2. Truncate at first negative value per configuration
        """
        truncated_configs = []
        
        for config_idx, config_data in enumerate(data_array):
            # Rule 1: Cap at τ_max (already handled by column selection)
            truncated_config = config_data[:self.tau_max + 1].copy()
            
            # Rule 2: Find first negative value and truncate there
            negative_indices = np.where(truncated_config < 0)[0]
            
            if len(negative_indices) > 0:
                first_negative = negative_indices[0]
                if first_negative > 0:  # Keep at least one value
                    truncated_config = truncated_config[:first_negative]
                else:
                    # If first value is negative, flag for removal
                    logging.warning(f"Config {config_idx}: First value is negative, excluding config")
                    continue
            
            truncated_configs.append(truncated_config)
        
        if not truncated_configs:
            logging.error("No valid configurations after truncation")
            return None
        
        # Pad configurations to same length for consistency
        max_len = max(len(config) for config in truncated_configs)
        padded_configs = []
        
        for config in truncated_configs:
            if len(config) < max_len:
                # Pad with NaN (will be handled later)
                padded = np.full(max_len, np.nan)
                padded[:len(config)] = config
                padded_configs.append(padded)
            else:
                padded_configs.append(config)
        
        return np.array(padded_configs)

    def apply_log_scaling(self, data_array):
        """Apply log transformation with epsilon offset"""
        data_with_epsilon = data_array + self.epsilon
        
        mask = data_with_epsilon > 0
        log_data = np.full_like(data_array, np.nan)
        log_data[mask] = np.log(data_with_epsilon[mask])
        
        return log_data

    def apply_normalization(self, train_data, bias_correction_data=None, evaluation_data=None, test_data=None):
        """Apply normalization using training data parameters"""
        if self.normalization_method == 'standard':
            scaler = StandardScaler()
        elif self.normalization_method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")
        
        train_flat = train_data.reshape(-1, train_data.shape[-1])
        valid_mask = ~np.isnan(train_flat).any(axis=1)
        if not np.any(valid_mask):
            raise ValueError("No valid training samples for normalization")
        
        scaler.fit(train_flat[valid_mask])
        
        normalized_train = self._transform_with_scaler(scaler, train_data)
        normalized_bias_correction = self._transform_with_scaler(scaler, bias_correction_data) if bias_correction_data is not None else None
        normalized_evaluation = self._transform_with_scaler(scaler, evaluation_data) if evaluation_data is not None else None
        normalized_test = self._transform_with_scaler(scaler, test_data) if test_data is not None else None
        
        self.normalization_params = {
            'method': self.normalization_method,
            'scaler': scaler
        }
        
        return normalized_train, normalized_bias_correction, normalized_evaluation, normalized_test

    def _transform_with_scaler(self, scaler, data):
        """Helper function to transform data with scaler"""
        if data is None:
            return None
        
        original_shape = data.shape
        data_flat = data.reshape(-1, data.shape[-1])
        
        # Transform only valid (non-NaN) samples
        valid_mask = ~np.isnan(data_flat).any(axis=1)
        transformed_flat = np.full_like(data_flat, np.nan)
        
        if np.any(valid_mask):
            transformed_flat[valid_mask] = scaler.transform(data_flat[valid_mask])
        
        return transformed_flat.reshape(original_shape)

    def create_data_splits(self, data, labels=None, random_state=42):
        """Create train/bias_correction/evaluation/test splits"""
        
        # First split: separate test set (unlabelled/test data)
        if labels is not None:
            remaining_data, test_data, remaining_labels, test_labels = train_test_split(
                data, labels, test_size=self.test_size, random_state=random_state, stratify=labels
            )
        else:
            remaining_data, test_data = train_test_split(
                data, test_size=self.test_size, random_state=random_state
            )
            remaining_labels, test_labels = None, None
        
        # Calculate adjusted sizes for remaining splits
        total_remaining = 1 - self.test_size
        train_size_adj = self.train_size / total_remaining
        bias_correction_size_adj = self.bias_correction_size / total_remaining
        evaluation_size_adj = self.evaluation_size / total_remaining
        
        # Second split: separate training data
        if remaining_labels is not None:
            train_data, temp_data, train_labels, temp_labels = train_test_split(
                remaining_data, remaining_labels, test_size=(1-train_size_adj), 
                random_state=random_state, stratify=remaining_labels
            )
        else:
            train_data, temp_data = train_test_split(
                remaining_data, test_size=(1-train_size_adj), random_state=random_state
            )
            train_labels, temp_labels = None, None
        
        # Third split: separate bias correction and evaluation data
        bias_eval_total = bias_correction_size_adj + evaluation_size_adj
        bias_correction_ratio = bias_correction_size_adj / bias_eval_total
        
        if temp_labels is not None:
            bias_correction_data, evaluation_data, bias_correction_labels, evaluation_labels = train_test_split(
                temp_data, temp_labels, test_size=(1-bias_correction_ratio), 
                random_state=random_state, stratify=temp_labels
            )
        else:
            bias_correction_data, evaluation_data = train_test_split(
                temp_data, test_size=(1-bias_correction_ratio), random_state=random_state
            )
            bias_correction_labels, evaluation_labels = None, None
        
        return (train_data, bias_correction_data, evaluation_data, test_data), (train_labels, bias_correction_labels, evaluation_labels, test_labels)

    def save_ml_ready_data(self):
        """Save processed data and metadata"""
        # Save data arrays
        np.save(self.ml_data_path / 'train_data.npy', self.train_data)
        np.save(self.ml_data_path / 'bias_correction_data.npy', self.bias_correction_data)
        np.save(self.ml_data_path / 'evaluation_data.npy', self.evaluation_data)
        np.save(self.ml_data_path / 'test_data.npy', self.test_data)
        
        # Save metadata
        metadata = {
            'preprocessing_params': {
                'tau_max': self.tau_max,
                'epsilon': self.epsilon,
                'normalization_method': self.normalization_method,
                'train_size': self.train_size,
                'bias_correction_size': self.bias_correction_size,
                'evaluation_size': self.evaluation_size,
                'test_size': self.test_size
            },
            'data_shapes': {
                'train': self.train_data.shape,
                'bias_correction': self.bias_correction_data.shape,
                'evaluation': self.evaluation_data.shape,
                'test': self.test_data.shape
            },
            'ensemble_info': {name: df.shape for name, df in self.ensemble_data.items()}
        }
        
        with open(self.ml_data_path / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save normalization parameters
        with open(self.ml_data_path / 'normalization_params.pkl', 'wb') as f:
            pickle.dump(self.normalization_params, f)


    def integrate_sanity_report(self):
        """Integrate findings from sanity check report"""
        if self.sanity_report_path is None or not Path(self.sanity_report_path).exists():
            logging.info("No sanity check report provided, using default parameters")
            return
        
        logging.info(f"Integrating sanity check report: {self.sanity_report_path}")
        
        try:
            with open(self.sanity_report_path, 'r') as f:
                report_content = f.read()
            
            # Look for truncation suggestions in the report
            if "Consider truncating after τ=" in report_content:
                # Extract suggested truncation points (but cap at τ=30)
                import re
                truncation_matches = re.findall(r'τ=(\d+)', report_content)
                if truncation_matches:
                    suggested_tau = min(int(max(truncation_matches)), 30)
                    if suggested_tau < self.tau_max:
                        self.tau_max = suggested_tau
                        logging.info(f"Adjusted τ_max to {self.tau_max} based on sanity report")
            
            # Check for normalization recommendations
            if "log transformation" in report_content.lower():
                logging.info("Sanity check recommends log transformation - will be applied")
            
            if "robust scaling" in report_content.lower():
                self.normalization_method = 'standard'  # Use standard scaler as robust alternative
                logging.info("Using standard normalization based on sanity report recommendation")
                
        except Exception as e:
            logging.error(f"Failed to integrate sanity report: {e}")

    def run_full_preprocessing(self):
        """Execute complete preprocessing pipeline"""
        logging.info("="*60)
        logging.info("STARTING 2PT CORRELATOR ML PREPROCESSING PIPELINE")
        logging.info("="*60)
        
        start_time = time.time()
        
        try:
            if not self.load_processed_data():
                raise RuntimeError("Failed to load processed data")
            
            self.integrate_sanity_report()
            
            # Combine all ensemble data
            all_configs = []
            ensemble_labels = []
            
            for ensemble_name, df in self.ensemble_data.items():
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if 'config_id' in numeric_cols:
                    numeric_cols = numeric_cols.drop('config_id')
                
                # Limit to tau_max columns
                available_cols = min(len(numeric_cols), self.tau_max + 1)
                ensemble_data = df[numeric_cols[:available_cols]].values
                
                all_configs.append(ensemble_data)
                ensemble_labels.extend([ensemble_name] * len(ensemble_data))
            
            combined_data = np.vstack(all_configs)
            
            # Apply truncation
            truncated_data = self.apply_truncation(combined_data)
            if truncated_data is None:
                raise RuntimeError("Truncation failed - no valid configurations")
            
            log_scaled_data = self.apply_log_scaling(truncated_data)
            splits, _ = self.create_data_splits(log_scaled_data)
            train_data, bias_correction_data, evaluation_data, test_data = splits
            
            self.train_data, self.bias_correction_data, self.evaluation_data, self.test_data = self.apply_normalization(
                train_data, bias_correction_data, evaluation_data, test_data
            )
            
            self.save_ml_ready_data()
            
            return True
            
        except Exception as e:
            logging.error(f"Preprocessing failed: {e}")
            return False


class MLPreprocessor3pt:
    """Placeholder for 3pt correlator preprocessing (future implementation)"""
    
    def __init__(self, processed_data_path: Path, results_path: Path):
        self.processed_data_path = processed_data_path
        self.results_path = results_path
        logging.info("3pt preprocessor initialized (not yet implemented)")
    
    def run_full_preprocessing(self):
        """Future implementation for 3pt correlators"""
        logging.info("3pt preprocessing not yet implemented")
        return False


def main(enable_logging=False):
    """Main function - currently processes 2pt correlators only"""
    
    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    processed_data_path = project_root / "data" / "processed"
    results_path = project_root / "data" / "ml_processed"
    sanity_report_path = project_root / "results" / "sanity_check" / "sanity_check_report.txt"
    

    
    # Check if processed data exists
    if not processed_data_path.exists():
        print("Processed data directory not found!")
        print(f"Please run convert_data.py first to create {processed_data_path}")
        return False
    
    csv_files = list(processed_data_path.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in processed directory!")
        print("Please run convert_data.py first to process your raw data.")
        return False
    
    # Initialize and run 2pt preprocessor
    
    preprocessor_2pt = MLPreprocessor2pt(
        processed_data_path=processed_data_path,
        results_path=results_path,
        sanity_report_path=sanity_report_path if sanity_report_path.exists() else None,
        enable_logging=enable_logging
    )
    
    success = preprocessor_2pt.run_full_preprocessing()
    
    if success:
        print("2pt correlator preprocessing completed successfully!")
    else:
        print("2pt correlator preprocessing failed!")
        print("Check the logs for detailed error information.")
    
    return success


if __name__ == "__main__":
    # Check for logging flag
    enable_logs = False
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.lower() in ['--log', '--verbose', '-l', '-v']:
                enable_logs = True
                break
    
    main(enable_logging=enable_logs)