"""
Experiment Runner for MLLQCD.

Runs the complete pipeline: preprocessing → training → evaluation → physics analysis.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocessing.pipeline import PreprocessingPipeline
from preprocessing.experiments import (
    load_config, get_experiment, get_experiment_folder, 
    generate_experiment, list_experiments
)
from training import train_model, train_all_models
from analysis.physics_analysis import TwoPointReport

MODELS = ['cnn', 'mlp', 'gbr', 'transformer']
CONFIG_PATHS = {
    'cnn': PROJECT_ROOT / 'configs' / 'cnn.yaml',
    'mlp': PROJECT_ROOT / 'configs' / 'mlp.yaml',
    'gbr': PROJECT_ROOT / 'configs' / 'gbr.yaml',
    'transformer': PROJECT_ROOT / 'configs' / 'transformer.yaml',
}


def run_preprocessing(config_type='2pt', force=False):
    """Run preprocessing pipeline if needed."""
    print("\n" + "="*60)
    print("STEP 1: PREPROCESSING")
    print("="*60 + "\n")
    
    processed_dir = PROJECT_ROOT / "data" / "processed" / "unaveraged_data"
    if processed_dir.exists() and list(processed_dir.glob("*.csv")) and not force:
        print("Processed data exists, skipping preprocessing.")
        print("Use --force to rerun preprocessing.\n")
        return True
    
    pipeline = PreprocessingPipeline(enable_logging=True)
    try:
        pipeline.run_full_pipeline(config_type=config_type)
        return True
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        return False


def run_experiment_data_generation(experiment_num, config_type='2pt', force=False):
    """Generate experiment data splits."""
    print("\n" + "="*60)
    print("STEP 2: EXPERIMENT DATA GENERATION")
    print("="*60 + "\n")
    
    exp, config = get_experiment(experiment_num, config_type)
    exp_path = PROJECT_ROOT / exp['output_path']
    
    if exp_path.exists() and (exp_path / "train_data_X.npy").exists() and not force:
        print(f"Experiment data exists at {exp_path}")
        print("Use --force to regenerate.\n")
        return exp['name'], exp_path
    
    try:
        generate_experiment(experiment_num, config_type)
        print(f"Generated data for: {exp['name']}")
        return exp['name'], exp_path
    except Exception as e:
        print(f"Experiment generation failed: {e}")
        return None, None


def run_training(experiment_num, config_type='2pt', models=None, use_bias_correction=True):
    """Train all models for an experiment."""
    print("\n" + "="*60)
    print("STEP 3: MODEL TRAINING")
    print("="*60 + "\n")
    
    models = models or MODELS
    results = {}
    
    for model_type in models:
        print("\n" + "-"*40)
        print(f"Training {model_type.upper()}")
        print("-"*40 + "\n")
        
        config_path = CONFIG_PATHS[model_type]
        if not config_path.exists():
            print(f"Config not found: {config_path}")
            continue
        
        try:
            model, preds_bc, preds_rm = train_model(
                config_path=str(config_path),
                experiment_num=experiment_num,
                model_type=model_type,
                config_type=config_type,
                bias_correction=use_bias_correction
            )
            results[model_type] = {'model': model, 'preds_bc': preds_bc, 'preds_rm': preds_rm}
            print(f"✓ {model_type.upper()} training complete")
        except Exception as e:
            print(f"✗ {model_type.upper()} training failed: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def run_ratio_methods(experiment_num, config_type='2pt', training_results=None):
    """Verify ratio methods were applied during training."""
    print("\n" + "="*60)
    print("STEP 4: RATIO METHODS (applied during training)")
    print("="*60 + "\n")
    
    experiment_folder = get_experiment_folder(experiment_num, config_type)
    
    for model_type in MODELS:
        ratio_path = PROJECT_ROOT / "results" / experiment_folder / model_type / "bias_corrected" / "ratio_predictions"
        
        if ratio_path.exists():
            print(f"✓ {model_type.upper()}: RM applied during training")
        else:
            print(f"✗ {model_type.upper()}: ratio predictions not found")


def run_physics_analysis(experiment_num, config_type='2pt', clean=False):
    """Run physics analysis and generate PDF."""
    print("\n" + "="*60)
    print("STEP 5: PHYSICS ANALYSIS")
    print("="*60 + "\n")
    
    exp, config = get_experiment(experiment_num, config_type)
    experiment_folder = get_experiment_folder(experiment_num, config_type)
    
    try:
        report = TwoPointReport(experiment_folder, project_root=PROJECT_ROOT)
        
        if not report.models_ready():
            print("Not all models have bias-corrected outputs.")
            print("Run training first.\n")
            return None
        
        pdf_path = report.run(save_pdf=True, clean=clean)
        return pdf_path
    except Exception as e:
        print(f"Physics analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_full_pipeline(experiment_num, config_type='2pt', models=None, 
                      use_bias_correction=True, skip_preprocessing=False,
                      skip_training=False, clean_plots=False, force=False):
    """Run the complete experiment pipeline."""
    start_time = datetime.now()
    
    print("\n" + "#"*60)
    print("#" + f"{'MLLQCD EXPERIMENT RUNNER':^58}" + "#")
    print("#"*60)
    print(f"\nExperiment: {experiment_num} ({config_type})")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not skip_preprocessing:
        if not run_preprocessing(config_type, force):
            print("Pipeline aborted: preprocessing failed")
            return
    
    exp_name, exp_path = run_experiment_data_generation(experiment_num, config_type, force)
    if exp_name is None:
        print("Pipeline aborted: experiment generation failed")
        return
    
    training_results = None
    if not skip_training:
        training_results = run_training(
            experiment_num, config_type, models, use_bias_correction
        )
        if not training_results:
            print("Pipeline aborted: no models trained successfully")
            return
    
    run_ratio_methods(experiment_num, config_type, training_results)
    pdf_path = run_physics_analysis(experiment_num, config_type, clean_plots)
    
    elapsed = datetime.now() - start_time
    print("\n" + "#"*60)
    print(f"Pipeline complete in {elapsed}")
    print("#"*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="MLLQCD Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Example usage:
        python experiment_runner.py --experiment 2
        python experiment_runner.py -e 2 --models cnn transformer
        python experiment_runner.py -e 2 --skip-preprocessing --skip-training
        python experiment_runner.py --list
        """
    )
    
    parser.add_argument('--experiment', '-e', type=int, help="Experiment number")
    parser.add_argument('--config-type', '-t', type=str, default='2pt', 
                       choices=['2pt', '3pt'], help="Experiment type")
    parser.add_argument('--models', '-m', nargs='+', 
                       choices=['cnn', 'mlp', 'gbr', 'transformer'],
                       help="Models to train (default: all)")
    parser.add_argument('--no-bias', action='store_true', help="Disable bias correction")
    parser.add_argument('--skip-preprocessing', action='store_true')
    parser.add_argument('--skip-training', action='store_true')
    parser.add_argument('--clean', action='store_true', help="Odd timeslices only in plots")
    parser.add_argument('--force', action='store_true', help="Force regenerate all data")
    parser.add_argument('--list', action='store_true', help="List experiments")
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable experiments:\n")
        list_experiments(args.config_type)
        return
    
    experiment_num = args.experiment
    if experiment_num is None:
        print("\nAvailable experiments:\n")
        list_experiments(args.config_type)
        print()
        try:
            experiment_num = int(input("Enter experiment number: "))
        except (ValueError, KeyboardInterrupt):
            print("\nAborted.")
            return
    
    run_full_pipeline(
        experiment_num=experiment_num,
        config_type=args.config_type,
        models=args.models,
        use_bias_correction=not args.no_bias,
        skip_preprocessing=args.skip_preprocessing,
        skip_training=args.skip_training,
        clean_plots=args.clean,
        force=args.force
    )



if __name__ == "__main__":
    main()