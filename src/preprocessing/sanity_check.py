import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import logging
import time
import sys
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def logging_setup(enable_logging=False, log_level=logging.INFO):
    if not enable_logging:
        logging.disable(logging.CRITICAL)
        return None
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    log_path = project_root / "logs"
    log_path.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = log_path / f"data_sanity_check_{timestamp}.log"
    
    logging.basicConfig(level=log_level,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ],
                        force=True)

    logger = logging.getLogger(__name__)
    logger.info(f"Sanity check logging enabled. Saving log file to {log_file}")
    return logger

class DataSanityChecker:
    def __init__(self, processed_path, results_path, enable_logging=False):
        self.processed_path = Path(processed_path)
        self.results_path = Path(results_path)
        self.results_path.mkdir(exist_ok=True)
        self.enable_logging = enable_logging
        self.logger = logging_setup(enable_logging)
        
        # Data storage
        self.csv_files = []
        self.data_dict = {}
        self.validation_results = {}
        self.snr_results = {}
        self.outliers = defaultdict(list)
        
        # Analysis parameters
        self.snr_threshold = 2.0  # Minimum acceptable SNR
        self.outlier_threshold = 3.0  # Standard deviations for outlier detection
        
    def log_info(self, message):
        if self.logger:
            self.logger.info(message)
        else:
            print(f"INFO: {message}")
    
    def log_warning(self, message):
        if self.logger:
            self.logger.warning(message)
        else:
            print(f"WARNING: {message}")
    
    def log_error(self, message):
        if self.logger:
            self.logger.error(message)
        else:
            print(f"ERROR: {message}")

    def discover_csv_files(self):
        """Find all CSV files in processed directory"""
        self.log_info(f"Discovering CSV files in {self.processed_path}")
        
        if not self.processed_path.exists():
            self.log_error(f"Processed data directory {self.processed_path} does not exist!")
            return False
        
        self.csv_files = list(self.processed_path.glob("*.csv"))
        
        if not self.csv_files:
            self.log_error(f"No CSV files found in {self.processed_path}")
            return False
        
        self.log_info(f"Found {len(self.csv_files)} CSV files")
        for file in self.csv_files[:5]:  # Log first 5 files
            self.log_info(f"  - {file.name}")
        if len(self.csv_files) > 5:
            self.log_info(f"  and {len(self.csv_files) - 5} more")
        
        return True

    def load_and_validate_data(self):
        """Load CSV files and validate data format & structure"""
        self.log_info("Loading and validating data format & structure...")
        
        validation_summary = {
            'total_files': len(self.csv_files),
            'loaded_successfully': 0,
            'failed_to_load': 0,
            'consistent_columns': True,
            'column_counts': [],
            'total_configs': 0,
            'data_types_valid': True
        }
        
        expected_columns = None
        
        for csv_file in self.csv_files:
            try:
                # Load CSV
                df = pd.read_csv(csv_file)
                label = csv_file.stem
                
                # Basic validation
                if df.empty:
                    self.log_warning(f"Empty file: {csv_file.name}")
                    continue
                
                # Check for required numeric columns
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) == 0:
                    self.log_error(f"No numeric columns found in {csv_file.name}")
                    validation_summary['failed_to_load'] += 1
                    continue
                
                # Store data
                self.data_dict[label] = df
                validation_summary['loaded_successfully'] += 1
                validation_summary['total_configs'] += len(df)
                
                # Check column consistency
                current_columns = len(df.columns)
                validation_summary['column_counts'].append(current_columns)
                
                if expected_columns is None:
                    expected_columns = current_columns
                elif expected_columns != current_columns:
                    validation_summary['consistent_columns'] = False
                    self.log_warning(f"Column count mismatch in {csv_file.name}: expected {expected_columns}, got {current_columns}")
                
                # Check for missing values, NaNs, infs
                missing_count = df.isnull().sum().sum()
                inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
                
                if missing_count > 0:
                    self.log_warning(f"Found {missing_count} missing values in {csv_file.name}")
                if inf_count > 0:
                    self.log_warning(f"Found {inf_count} infinite values in {csv_file.name}")
                
                self.log_info(f"Loaded {csv_file.name}: {len(df)} configs, {len(df.columns)} columns")
                
            except Exception as e:
                self.log_error(f"Failed to load {csv_file.name}: {e}")
                validation_summary['failed_to_load'] += 1
        
        self.validation_results = validation_summary
        self.log_info(f"Data loading complete: {validation_summary['loaded_successfully']}/{validation_summary['total_files']} files loaded successfully")
        
        return validation_summary['loaded_successfully'] > 0

    def detect_data_corruption(self):
        """Detect missing values, NaNs, and extreme outliers"""
        self.log_info("Detecting data corruption and outliers...")
        
        corruption_results = {
            'files_with_issues': [],
            'total_missing': 0,
            'total_infinite': 0,
            'total_outliers': 0,
            'outlier_configs': defaultdict(list)
        }
        
        for label, df in self.data_dict.items():
            # Get numeric columns (exclude config_id if present)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if 'config_id' in numeric_cols:
                numeric_cols = numeric_cols.drop('config_id')
            
            if len(numeric_cols) == 0:
                continue
            
            numeric_data = df[numeric_cols]
            
            # Check for missing/invalid data
            missing = numeric_data.isnull().sum().sum()
            infinite = np.isinf(numeric_data).sum().sum()
            
            if missing > 0 or infinite > 0:
                corruption_results['files_with_issues'].append(label)
                corruption_results['total_missing'] += missing
                corruption_results['total_infinite'] += infinite
            
            # Detect extreme outliers using IQR method
            for idx, row in numeric_data.iterrows():
                values = row.values
                Q1 = np.percentile(values, 25)
                Q3 = np.percentile(values, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = (values < lower_bound) | (values > upper_bound)
                outlier_count = np.sum(outliers)
                
                if outlier_count > len(values) * 0.1:  # More than 10% outliers
                    corruption_results['outlier_configs'][label].append(idx)
                    corruption_results['total_outliers'] += 1
        
        self.log_info(f"Corruption analysis complete:")
        self.log_info(f"  - Missing values: {corruption_results['total_missing']}")
        self.log_info(f"  - Infinite values: {corruption_results['total_infinite']}")
        self.log_info(f"  - Outlier configs: {corruption_results['total_outliers']}")
        
        return corruption_results

    def calculate_snr_analysis(self):
        """Calculate Signal-to-Noise Ratio for each time slice"""
        self.log_info("Calculating Signal-to-Noise Ratio analysis...")
        
        snr_results = {}
        
        for label, df in self.data_dict.items():
            # Get numeric columns (exclude config_id if present)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if 'config_id' in numeric_cols:
                numeric_cols = numeric_cols.drop('config_id')
            
            if len(numeric_cols) == 0:
                continue
            
            numeric_data = df[numeric_cols].values
            
            # Calculate ensemble mean and std for each time slice
            ensemble_mean = np.mean(numeric_data, axis=0)
            ensemble_std = np.std(numeric_data, axis=0)
            
            # Calculate SNR (avoid division by zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                snr = np.abs(ensemble_mean) / ensemble_std
                snr[ensemble_std == 0] = np.inf
            
            # Find noise-dominated regions
            low_snr_indices = np.where(snr < self.snr_threshold)[0]
            
            snr_results[label] = {
                'ensemble_mean': ensemble_mean,
                'ensemble_std': ensemble_std,
                'snr': snr,
                'low_snr_indices': low_snr_indices,
                'mean_snr': np.mean(snr[np.isfinite(snr)]),
                'time_slices': len(numeric_cols)
            }
            
            self.log_info(f"  {label}: Mean SNR = {snr_results[label]['mean_snr']:.2f}, Low SNR regions: {len(low_snr_indices)}")
        
        self.snr_results = snr_results
        return snr_results

    def generate_diagnostic_plots(self):
        """Generate comprehensive diagnostic plots"""
        self.log_info("Generating diagnostic plots...")
        
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Ensemble-averaged correlators with error bars
        self.plot_ensemble_averages()
        
        # 2. Config-specific heatmaps
        self.plot_config_heatmaps()
        
        # 3. SNR analysis plots
        self.plot_snr_analysis()
        
        # 4. Data distribution analysis
        self.plot_data_distributions()
        
        # 5. Outlier analysis
        self.plot_outlier_analysis()
        
        self.log_info("Diagnostic plots generated successfully")

    def plot_ensemble_averages(self):
        """Plot ensemble-averaged correlators with error bars"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        plot_count = 0
        for label, data in list(self.snr_results.items())[:4]:  # Plot first 4 ensembles
            if plot_count >= 4:
                break
            
            ax = axes[plot_count]
            
            time_slices = np.arange(len(data['ensemble_mean']))
            ensemble_mean = data['ensemble_mean']
            ensemble_std = data['ensemble_std']
            
            # Plot with error bars
            ax.errorbar(time_slices, ensemble_mean, yerr=ensemble_std, 
                       fmt='o-', capsize=3, capthick=1, markersize=4)
            ax.set_xlabel('Time Slice (τ)')
            ax.set_ylabel('Correlator Value')
            ax.set_title(f'Ensemble Average: {label}')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            
            plot_count += 1
        
        # Remove empty subplots
        for i in range(plot_count, 4):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'ensemble_averages.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_config_heatmaps(self):
        """Plot config-specific correlator heatmaps"""
        for label, df in list(self.data_dict.items())[:2]:  # Plot first 2 ensembles
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if 'config_id' in numeric_cols:
                numeric_cols = numeric_cols.drop('config_id')
            
            if len(numeric_cols) == 0:
                continue
            
            data = df[numeric_cols].values
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create heatmap
            im = ax.imshow(data, aspect='auto', cmap='viridis', interpolation='nearest')
            
            ax.set_xlabel('Time Slice (τ)')
            ax.set_ylabel('Configuration')
            ax.set_title(f'Correlator Heatmap: {label}')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Correlator Value')
            
            plt.tight_layout()
            plt.savefig(self.results_path / f'heatmap_{label}.png', dpi=300, bbox_inches='tight')
            plt.close()

    def plot_snr_analysis(self):
        """Plot SNR analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # SNR vs time slice
        ax1 = axes[0, 0]
        for label, data in list(self.snr_results.items())[:3]:
            time_slices = np.arange(len(data['snr']))
            ax1.plot(time_slices, data['snr'], 'o-', label=label, markersize=3)
        
        ax1.axhline(y=self.snr_threshold, color='red', linestyle='--', label=f'Threshold ({self.snr_threshold})')
        ax1.set_xlabel('Time Slice (τ)')
        ax1.set_ylabel('Signal-to-Noise Ratio')
        ax1.set_title('SNR vs Time Slice')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # SNR histogram
        ax2 = axes[0, 1]
        all_snr = []
        for data in self.snr_results.values():
            snr_finite = data['snr'][np.isfinite(data['snr'])]
            all_snr.extend(snr_finite)
        
        ax2.hist(all_snr, bins=50, alpha=0.7, density=True)
        ax2.axvline(x=self.snr_threshold, color='red', linestyle='--', label=f'Threshold ({self.snr_threshold})')
        ax2.set_xlabel('Signal-to-Noise Ratio')
        ax2.set_ylabel('Density')
        ax2.set_title('SNR Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Noise variance vs time slice
        ax3 = axes[1, 0]
        for label, data in list(self.snr_results.items())[:3]:
            time_slices = np.arange(len(data['ensemble_std']))
            ax3.plot(time_slices, data['ensemble_std']**2, 'o-', label=label, markersize=3)
        
        ax3.set_xlabel('Time Slice (τ)')
        ax3.set_ylabel('Noise Variance')
        ax3.set_title('Noise Variance vs Time Slice')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # Mean SNR by ensemble
        ax4 = axes[1, 1]
        labels = []
        mean_snrs = []
        for label, data in self.snr_results.items():
            labels.append(label[:10])  # Truncate long labels
            mean_snrs.append(data['mean_snr'])
        
        bars = ax4.bar(range(len(labels)), mean_snrs, alpha=0.7)
        ax4.axhline(y=self.snr_threshold, color='red', linestyle='--', label=f'Threshold ({self.snr_threshold})')
        ax4.set_xlabel('Ensemble')
        ax4.set_ylabel('Mean SNR')
        ax4.set_title('Mean SNR by Ensemble')
        ax4.set_xticks(range(len(labels)))
        ax4.set_xticklabels(labels, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'snr_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_data_distributions(self):
        """Plot data distribution analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Overall value distribution
        ax1 = axes[0, 0]
        all_values = []
        for df in self.data_dict.values():
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if 'config_id' in numeric_cols:
                numeric_cols = numeric_cols.drop('config_id')
            if len(numeric_cols) > 0:
                all_values.extend(df[numeric_cols].values.flatten())
        
        all_values = np.array(all_values)
        all_values = all_values[np.isfinite(all_values)]
        
        ax1.hist(all_values, bins=100, alpha=0.7, density=True)
        ax1.set_xlabel('Correlator Value')
        ax1.set_ylabel('Density')
        ax1.set_title('Overall Value Distribution')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Value range by time slice
        ax2 = axes[0, 1]
        for label, df in list(self.data_dict.items())[:3]:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if 'config_id' in numeric_cols:
                numeric_cols = numeric_cols.drop('config_id')
            
            if len(numeric_cols) > 0:
                data = df[numeric_cols].values
                means = np.mean(data, axis=0)
                stds = np.std(data, axis=0)
                time_slices = np.arange(len(means))
                
                ax2.fill_between(time_slices, means - stds, means + stds, alpha=0.3, label=f'{label} ±1σ')
                ax2.plot(time_slices, means, 'o-', label=f'{label} mean', markersize=3)
        
        ax2.set_xlabel('Time Slice (τ)')
        ax2.set_ylabel('Correlator Value')
        ax2.set_title('Value Range by Time Slice')
        ax2.legend()
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Config count by ensemble
        ax3 = axes[1, 0]
        labels = []
        config_counts = []
        for label, df in self.data_dict.items():
            labels.append(label[:10])
            config_counts.append(len(df))
        
        ax3.bar(range(len(labels)), config_counts, alpha=0.7)
        ax3.set_xlabel('Ensemble')
        ax3.set_ylabel('Number of Configurations')
        ax3.set_title('Configuration Count by Ensemble')
        ax3.set_xticks(range(len(labels)))
        ax3.set_xticklabels(labels, rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Time series length distribution
        ax4 = axes[1, 1]
        time_lengths = []
        for df in self.data_dict.values():
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if 'config_id' in numeric_cols:
                numeric_cols = numeric_cols.drop('config_id')
            time_lengths.append(len(numeric_cols))
        
        ax4.hist(time_lengths, bins=20, alpha=0.7)
        ax4.set_xlabel('Number of Time Slices')
        ax4.set_ylabel('Number of Ensembles')
        ax4.set_title('Time Series Length Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'data_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_outlier_analysis(self):
        """Plot outlier analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Outlier count by ensemble
        ax1 = axes[0]
        labels = []
        outlier_counts = []
        
        for label, df in self.data_dict.items():
            labels.append(label[:10])
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if 'config_id' in numeric_cols:
                numeric_cols = numeric_cols.drop('config_id')
            
            if len(numeric_cols) > 0:
                # Count outliers using Z-score method
                data = df[numeric_cols].values
                z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
                outliers = np.sum(z_scores > self.outlier_threshold)
                outlier_counts.append(outliers)
            else:
                outlier_counts.append(0)
        
        bars = ax1.bar(range(len(labels)), outlier_counts, alpha=0.7, color='red')
        ax1.set_xlabel('Ensemble')
        ax1.set_ylabel('Number of Outlier Values')
        ax1.set_title('Outlier Count by Ensemble')
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Outlier percentage distribution
        ax2 = axes[1]
        outlier_percentages = []
        for label, df in self.data_dict.items():
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if 'config_id' in numeric_cols:
                numeric_cols = numeric_cols.drop('config_id')
            
            if len(numeric_cols) > 0:
                data = df[numeric_cols].values
                total_values = data.size
                z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
                outliers = np.sum(z_scores > self.outlier_threshold)
                percentage = (outliers / total_values) * 100 if total_values > 0 else 0
                outlier_percentages.append(percentage)
        
        ax2.hist(outlier_percentages, bins=20, alpha=0.7, color='red')
        ax2.set_xlabel('Outlier Percentage (%)')
        ax2.set_ylabel('Number of Ensembles')
        ax2.set_title('Distribution of Outlier Percentages')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'outlier_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_ml_recommendations(self):
        """Generate ML preprocessing recommendations"""
        self.log_info("Generating ML preprocessing recommendations...")
        
        recommendations = {
            'data_quality': 'GOOD',
            'issues_found': [],
            'preprocessing_steps': [],
            'configs_to_exclude': [],
            'truncation_suggestions': [],
            'normalization_recommendations': []
        }
        
        # Analyze data quality
        total_configs = sum(len(df) for df in self.data_dict.values())
        
        if self.validation_results['failed_to_load'] > 0:
            recommendations['issues_found'].append(
                f"Failed to load {self.validation_results['failed_to_load']} files"
            )
            recommendations['data_quality'] = 'POOR'
        
        if not self.validation_results['consistent_columns']:
            recommendations['issues_found'].append("Inconsistent column counts across files")
            recommendations['data_quality'] = 'FAIR'
        
        # SNR-based recommendations
        low_snr_ensembles = []
        for label, data in self.snr_results.items():
            if data['mean_snr'] < self.snr_threshold:
                low_snr_ensembles.append(label)
                
            # Suggest truncation points
            snr = data['snr']
            valid_indices = np.where(snr >= self.snr_threshold)[0]
            if len(valid_indices) > 0:
                last_valid_idx = valid_indices[-1]
                if last_valid_idx < len(snr) - 1:
                    recommendations['truncation_suggestions'].append(
                        f"{label}: Consider truncating after τ={last_valid_idx}"
                    )
        
        if low_snr_ensembles:
            recommendations['issues_found'].append(
                f"Low SNR ensembles: {', '.join(low_snr_ensembles[:3])}"
            )
        
        # Value range analysis for normalization
        all_ranges = []
        for df in self.data_dict.values():
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if 'config_id' in numeric_cols:
                numeric_cols = numeric_cols.drop('config_id')
            
            if len(numeric_cols) > 0:
                data = df[numeric_cols].values
                data_range = np.max(data) - np.min(data)
                all_ranges.append(data_range)
        
        if all_ranges:
            max_range = max(all_ranges)
            min_range = min(all_ranges)
            
            if max_range / min_range > 100:
                recommendations['normalization_recommendations'].append(
                    "Large dynamic range detected - consider log transformation or robust scaling"
                )
            else:
                recommendations['normalization_recommendations'].append(
                    "Standard scaling or min-max normalization recommended"
                )
        
        # Basic preprocessing steps
        if recommendations['data_quality'] != 'GOOD':
            recommendations['preprocessing_steps'].extend([
                "Remove or interpolate missing values if detected",
                "Apply outlier detection and handling"
            ])
        
        if recommendations['data_quality'] == 'GOOD' and not recommendations['issues_found']:
            recommendations['preprocessing_steps'].insert(0, 
                "Data quality is excellent - minimal preprocessing required"
            )
        
        return recommendations

    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        self.log_info("Generating summary report...")
        
        # Get corruption results
        corruption_results = self.detect_data_corruption()
        
        # Get ML recommendations
        ml_recommendations = self.generate_ml_recommendations()
        
        # Create report
        report_lines = [
            "=" * 80,
            "LATTICE QCD DATA SANITY CHECK REPORT",
            "=" * 80,
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Processed directory: {self.processed_path}",
            "",
            "DATA OVERVIEW:",
            "-" * 40,
            f"Total CSV files found: {len(self.csv_files)}",
            f"Successfully loaded: {self.validation_results['loaded_successfully']}",
            f"Failed to load: {self.validation_results['failed_to_load']}",
            f"Total configurations: {self.validation_results['total_configs']}",
            f"Column consistency: {'YES' if self.validation_results['consistent_columns'] else 'NO'}",
            "",
            "DATA QUALITY ASSESSMENT:",
            "-" * 40,
            f"Missing values: {corruption_results['total_missing']}",
            f"Infinite values: {corruption_results['total_infinite']}",
            f"Outlier configurations: {corruption_results['total_outliers']}",
            f"Files with issues: {len(corruption_results['files_with_issues'])}",
            "",
            "SIGNAL-TO-NOISE ANALYSIS:",
            "-" * 40
        ]
        
        for label, data in self.snr_results.items():
            report_lines.extend([
                f"  {label}:",
                f"    Mean SNR: {data['mean_snr']:.2f}",
                f"    Time slices: {data['time_slices']}",
                f"    Low SNR regions: {len(data['low_snr_indices'])}",
            ])
        
        report_lines.extend([
            "",
            "ML PREPROCESSING RECOMMENDATIONS:",
            "-" * 40,
            f"Overall Data Quality: {ml_recommendations['data_quality']}",
            ""
        ])
        
        if ml_recommendations['issues_found']:
            report_lines.append("Issues Found:")
            for issue in ml_recommendations['issues_found']:
                report_lines.append(f"  {issue}")
            report_lines.append("")
        
        if ml_recommendations['truncation_suggestions']:
            report_lines.append("Truncation Suggestions:")
            for suggestion in ml_recommendations['truncation_suggestions']:
                report_lines.append(f"  {suggestion}")
            report_lines.append("")
        
        if ml_recommendations['normalization_recommendations']:
            report_lines.append("Normalization Recommendations:")
            for rec in ml_recommendations['normalization_recommendations']:
                report_lines.append(f"  {rec}")
            report_lines.append("")
        
        report_lines.append("General Preprocessing Steps:")
        for step in ml_recommendations['preprocessing_steps']:
            report_lines.append(f"  {step}")
        
        report_lines.extend([
            "",
            "GENERATED PLOTS:",
            "-" * 40,
            "  ensemble_averages.png",
            "  heatmap_*.png", 
            "  snr_analysis.png",
            "  data_distributions.png",
            "  outlier_analysis.png",
            "",
            "=" * 80
        ])
        
        # Save report
        report_text = "\n".join(report_lines)
        report_file = self.results_path / "sanity_check_report.txt"
        
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        # Also log the report
        for line in report_lines:
            self.log_info(line)
        
        self.log_info(f"Summary report saved to {report_file}")
        
        return report_text

    def run_full_analysis(self):
        """Run complete sanity check analysis"""
        self.log_info("Starting comprehensive data sanity check...")
        start_time = time.time()
        
        try:
            # Step 1: Discover files
            if not self.discover_csv_files():
                return False
            
            # Step 2: Load and validate data
            if not self.load_and_validate_data():
                return False
            
            # Step 3: Calculate SNR analysis
            self.calculate_snr_analysis()
            
            # Step 4: Generate diagnostic plots
            self.generate_diagnostic_plots()
            
            # Step 5: Generate summary report
            self.generate_summary_report()
            
            total_time = time.time() - start_time
            self.log_info(f"Sanity check completed successfully in {total_time:.2f} seconds")
            self.log_info(f"Results saved to: {self.results_path}")
            
            return True
            
        except Exception as e:
            self.log_error(f"Sanity check failed: {e}")
            return False

def main(enable_logging=False):
    """Main function"""
    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    processed_path = project_root / "data" / "processed"
    results_path = project_root / "results" / "sanity_check"
    
    # Create results directory
    results_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("LATTICE QCD DATA SANITY CHECKER")
    print("=" * 60)
    print(f"Processed data path: {processed_path}")
    print(f"Results will be saved to: {results_path}")
    print()
    
    # Initialize checker
    checker = DataSanityChecker(processed_path, results_path, enable_logging)
    
    # Run analysis
    success = checker.run_full_analysis()
    
    if success:
        print("Sanity check completed successfully!")
    else:
        print("Sanity check failed!")
        print("Check the error messages above for details.")
    
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