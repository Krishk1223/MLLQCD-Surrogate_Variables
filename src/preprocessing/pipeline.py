import sys
import argparse
import logging
from pathlib import Path
from time import time

try:
     from src.preprocessing.convert_data import convert_data as convert_data_main
     from src.preprocessing.sanity_check import DataSanityChecker
     from src.preprocessing.averaged_data import time_source_averaging
     from src.preprocessing.experiments import all_experiments_data_pipeline, single_experiment_data_pipeline
except ImportError:
    from convert_data import convert_data as convert_data_main
    from sanity_check import DataSanityChecker
    from averaged_data import time_source_averaging
    from experiments import all_experiments_data_pipeline, single_experiment_data_pipeline
class PreprocessingPipeline:
    """End-to-end preprocessing pipeline for MLLQCD."""
    
    def __init__(self, enable_logging=False, interactive=False):
        """Initialize the pipeline."""
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.enable_logging = enable_logging
        self.interactive = interactive
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration."""
        if self.enable_logging:
            log_dir = self.project_root / 'logs'
            log_dir.mkdir(exist_ok=True)
            
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_dir / 'pipeline.log'),
                    logging.StreamHandler()
                ]
            )
        else:
            logging.basicConfig(level=logging.WARNING)
        self.logger = logging.getLogger(__name__)
    
    def run_full_pipeline(self, skip_steps=None, force_overwrite=False):
        """
        Run the complete preprocessing pipeline.
        
        Args:
            skip_steps: List of steps to skip (e.g., ['convert', 'sanity_check'])
            force_overwrite: Force overwrite of existing data
        """
        skip_steps = skip_steps or []
        pipeline_steps = [
            ('convert', self.step_convert_data),
            ('sanity_check', self.step_sanity_check),
            ('averaging', self.step_averaging),
            ('experiments', self.step_experiments),
        ]
        
        self.logger.info("Starting preprocessing pipeline...")
        start_time = time()
        for step_name, step_func in pipeline_steps:
            if step_name in skip_steps:
                self.logger.info(f"Skipping: {step_name}")
                continue
            
            try:
                self.logger.info(f"Running: {step_name}")
                step_start = time()
                step_func()
                step_duration = time() - step_start
                self.logger.info(f"{step_name} completed in {step_duration:.2f}s")
                print(f"{step_name} completed in {step_duration:.2f}s")
            except Exception as e:
                self.logger.error(f"{step_name} failed: {str(e)}")
                raise
        
        total_duration = time() - start_time
        self.logger.info(f"Pipeline completed successfully in {total_duration:.2f}s")
        print(f"Pipeline completed successfully in {total_duration:.2f}s")
    
    def step_convert_data(self):
        """Preprocessing Step 1: Convert raw data to CSV format."""
        self.logger.info("Converting raw data to CSV format...")
        convert_data_main(enable_logging=self.enable_logging, interactive=self.interactive)
    
    def step_sanity_check(self):
        """Preprocessing Step 2: Run sanity checks on converted data."""
        self.logger.info("Running sanity checks on converted data...")
        processed_path = self.project_root / 'data' / 'processed' / 'unaveraged_data'
        results_path = self.project_root / 'results' / 'sanity_checks'
        sanity_checker = DataSanityChecker(processed_path=processed_path, results_path=results_path)
        try:
            sanity_checker.run_sanity_check_pipeline()
            self.logger.info("Sanity checks completed successfully.")
        except Exception as e:
            self.logger.error(f"Sanity check failed: {str(e)}")
            print("Sanity check failed. Please review the logs for details. Exiting pipeline.")
            sys.exit(1)
    
    def step_averaging(self):
        """Preprocessing Step 3 (for truth analysis): Create averaged data and jackknife errors."""
        self.logger.info("Creating averaged data and jackknife errors...")
        try:
            time_source_averaging()
            self.logger.info("Averaging completed successfully.")
        except Exception as e:
            self.logger.error(f"Averaging failed: {str(e)}")
            print("Averaging step failed. Please review the logs for details. Exiting pipeline.")
            sys.exit(1)
    
    def step_experiments(self, tau_max=None, strategy='remove'):
        """
        Step 4: Generate experiment data splits.
        
        Args:
            tau_max: Maximum tau cutoff (None uses all)
            strategy: Missing data strategy ('remove' for removing rows or 'warn' for warnung user)
        """
        self.logger.info("Generating experiment data splits...")
        start_time = time()
        try:
            all_experiments_data_pipeline(tau_max=tau_max, strategy=strategy)
            self.logger.info("Experiment data generation completed successfully.")
        except Exception as e:
            self.logger.error(f"Experiment data generation failed: {str(e)}")
            print("Experiment data generation failed. Please review the logs for details. Exiting pipeline.")
            sys.exit(1)
    
    def run_single_experiment(self, experiment_number, tau_max=None, strategy='remove'):
        """
        Run preprocessing and generate data for a single experiment.
        
        Args:
            experiment_number: Experiment ID to process
            tau_max: Maximum tau cutoff
            strategy: Missing data strategy
        """
        self.logger.info(f"Running pipeline for Experiment {experiment_number}...")
        start_time = time()
        
        try:
            self.step_convert_data()
            self.step_sanity_check()
            self.step_averaging()
            self.logger.info(f"Generating data for Experiment {experiment_number}...")
            single_experiment_data_pipeline(experiment_number, tau_max=tau_max, strategy=strategy)
            
            duration = time() - start_time
            self.logger.info(f"Experiment {experiment_number} completed in {duration:.2f}s")
        except Exception as e:
            self.logger.error(f"Experiment {experiment_number} failed: {str(e)}")
            raise


def main():
    """Main entry point for the preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="MLLQCD End-to-End Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python pipeline.py
  
  # Run pipeline with logging
  python pipeline.py --log
  
  # Skip sanity check and averaging
  python pipeline.py --skip sanity_check averaging
  
  # Run only single experiment
  python pipeline.py --experiment 1
  
  # Run with tau cutoff
  python pipeline.py --tau_max 30
        """
    )
    
    parser.add_argument('--logs', action='store_true', help="Enable detailed logging")
    parser.add_argument(
        '--skip',
        nargs='+',
        choices=['convert', 'sanity_check', 'averaging', 'experiments'],
        help="Steps to skip in the pipeline"
    )
    parser.add_argument(
        '--experiment',
        type=int,
        help="Run preprocessing for a specific experiment only"
    )
    parser.add_argument(
        '--tau_max',
        type=int,
        default=None,
        help="Maximum tau cutoff (default: None uses all tau)"
    )
    parser.add_argument(
        '--strategy',
        choices=['remove', 'warn'],
        default='remove',
        help="Missing data handling strategy (default: remove)"
    )
    parser.add_argument(
        '--i',
        action='store_true',
        help="Interactive mode for convert data step"
    )
    args = parser.parse_args()
    
    pipeline = PreprocessingPipeline(enable_logging=args.logs, interactive=args.i)
    
        
    if args.experiment is not None:
        pipeline.run_single_experiment(
            args.experiment,
            tau_max=args.tau_max,
            strategy=args.strategy
        )
    else:
        pipeline.run_full_pipeline(
            skip_steps=args.skip,
            force_overwrite=False
        )


if __name__ == "__main__":
    main()