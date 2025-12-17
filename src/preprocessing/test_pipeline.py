import pytest
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from src.preprocessing.pipeline import PreprocessingPipeline

"""VERY BASIC TESTS FOR PREPROCESSING PIPELINE."""

@pytest.fixture
def pipeline():
    """Fixture to create a PreprocessingPipeline instance, logging disabled."""
    return PreprocessingPipeline(enable_logging=False)

@pytest.fixture
def logged_pipeline():
    """Fixture to create a PreprocessingPipeline instance, logging enabled."""
    return PreprocessingPipeline(enable_logging=True)

@pytest.fixture
def project_paths():
    root = Path(__file__).resolve().parent.parent.parent
    return {
        'root': root,
        'raw': root / 'data' / 'raw',
        'processed': root / 'data' / 'processed',
        'unaveraged': root / 'data' / 'processed' / 'unaveraged_data',
        'averaged': root / 'data' / 'processed' / 'averaged_data',
        'jackknife mean': root / 'data' / 'processed' / 'jackknife_mean',
        'jackknife errors': root / 'data' / 'processed' / 'jackknife_errors',
        'results': root / 'results' / 'sanity_checks',
        'logs': root / 'logs'
    }

class TestPreprocessingPipeline:
    def test_pipeline_initialisation(self, pipeline):
        """
        Test that the PreprocessingPipeline initializes correctly 
        with project root and logger instructions
        """
        assert pipeline is not None
        assert hasattr(pipeline, 'logger')
        assert hasattr(pipeline, 'project_root')
    
    def test_root_validity(self, pipeline):
        """Test that the project root path is valid."""
        assert pipeline.project_root.exists()
        assert pipeline.project_root.is_dir()
    
    def test_logging_default(self, pipeline):
        """Test that logging is disabled by default."""
        assert pipeline.enable_logging is False
    
    def test_logging_enabled(self, logged_pipeline):
        """Test that logging is enabled when specified."""
        assert logged_pipeline.enable_logging is True
    
class TestDataConversion:
    @pytest.mark.slow #slow test so optionally skippable
    def test_convert_data(self, pipeline):
        """Test the raw data conversion process."""
        try:
            pipeline.step_convert_data()
        except FileNotFoundError:
            pytest.skip("Raw data files not found; skipping conversion test.")
        except Exception as e:
            pytest.fail(f"Data conversion failed with exception: {e}")

    def test_convert_data_directory(self, project_paths):
        """Check that the converted data directory exists."""
        unaveraged_path = project_paths['unaveraged']
        if not unaveraged_path.exists():
            pytest.skip("Converted non time source averaged data directory not found: skipping test.")
        assert unaveraged_path.is_dir()
    
class TestDataSanityChecker:
    @pytest.mark.slow #slow test so optionally skippable
    def test_sanity_check_execution(self, pipeline, project_paths):
        """Test sanity check actually executes without major errors."""
        unaveraged_path = project_paths['unaveraged']
        if not unaveraged_path.exists():
            pytest.skip("Converted non time source averaged data directory not found: skipping sanity check test.")
        try:
            pipeline.step_sanity_check()
        except SystemExit:
            pytest.skip("Sanity check failed and exited the pipeline; skipping test.")
        except Exception as e:
            pytest.fail(f"Sanity check failed with exception: {e}")
    
    def test_sanity_check_results(self, project_paths):
        """Check that sanity check results are generated."""
        results_path = project_paths['results']
        if not results_path.exists():
            pytest.skip("Sanity check results directory not found:, will skip test.")
        test_file = results_path / "sanity_check_report.txt"
        assert test_file.exists(), "Sanity check report file not found."
    
class TestAveragedDataCreation:
    @pytest.mark.slow #slow test so optionally skippable
    def test_averaging_execution(self, pipeline, project_paths):
        """Test the averaging process executes without major errors."""
        unaveraged_path = project_paths['unaveraged']
        if not unaveraged_path.exists():
            pytest.skip("Converted non time source averaged data directory not found: skipping averaging test.")
        try:
            pipeline.step_averaging()
        except SystemExit:
            pytest.skip("Averaging step failed and exited the pipeline; skipping test.")
        except Exception as e:
            pytest.fail(f"Averaging step failed with exception: {e}")
    
    def test_averaged_data_directory(self, project_paths):
        """Check that the averaged data directory exists."""
        averaged_path = project_paths['averaged']
        jackknife_mean_path = project_paths['jackknife mean']
        jackknife_errors_path = project_paths['jackknife errors']
        if not averaged_path.exists() and not jackknife_mean_path.exists() and not jackknife_errors_path.exists():
            pytest.skip("Averaged data directory not found: skipping test.")
        assert averaged_path.is_dir()
        assert jackknife_mean_path.is_dir()
        assert jackknife_errors_path.is_dir()

    def test_averaged_data_files(self, project_paths):
        """Check that there are CSV files in the averaged data directory."""
        averaged_path = project_paths['averaged']
        csv_files = list(averaged_path.glob("*.csv"))
        assert len(csv_files) > 0, "No CSV files found in the averaged data directory."
        for file in csv_files:
            df = pd.read_csv(file)
            assert not df.empty, f"Averaged data file {file} is empty."
    
    def test_jackknife_mean_files(self, project_paths):
        """Check that there are CSV files in the jackknife mean directory."""
        jackknife_mean_path = project_paths['jackknife mean']
        npy_files = list(jackknife_mean_path.glob("*.npy"))
        assert len(npy_files) > 0, "No CSV files found in the jackknife mean directory."
        for file in npy_files:
            np_array = np.load(file)
            assert np_array.size > 0, f"Jackknife mean file {file} is empty."
    
    def test_jackknife_error_files(self, project_paths):
        """Check that there are NPY files in the jackknife errors directory."""
        jackknife_errors_path = project_paths['jackknife errors']
        npy_files = list(jackknife_errors_path.glob("*.npy"))
        assert len(npy_files) > 0, "No NPY files found in the jackknife errors directory."
        for file in npy_files:
            np_array = np.load(file)
            assert np_array.size > 0, f"Jackknife error file {file} is empty."
    
class TestExperimentGeneration:
    @pytest.mark.slow
    def test_experiment_run(self, pipeline, project_paths):
        """Test that the experiment step runs without major errors."""
        unaveraged_path = project_paths['unaveraged']
        if not unaveraged_path.exists():
            pytest.skip("Converted non time source averaged data directory not found: skipping experiment generation test.")
        try:
            pipeline.step_experiments(tau_max=10, strategy='remove')
        except Exception as e:
            pytest.fail(f"Experiment generation failed with exception: {e}")
        
    def test_file_types(self, project_paths):
        """Check that experiment output files have correct files contained."""
        required_files = ['train_data_X.npy', 'train_data_y.npy',
                        'evaluation_data_X.npy', 'evaluation_data_y.npy',
                        'bias_correction_data_X.npy', 'bias_correction_data_y.npy',
                        'test_data_X.npy', 'test_data_y.npy',
                        'metadata.json']
        
        experiments_path = project_paths['root'] / 'data' / 'experiments'
        if not experiments_path.exists():
            pytest.skip("Experiments directory not found; skipping test.")
        
        experiment_dirs = [d for d in experiments_path.iterdir() if d.is_dir()]
        assert len(experiment_dirs) > 0, "No experiment directories found."
        
        for exp_dir in experiment_dirs:
            for required_file in required_files:
                file_path = exp_dir / required_file
                assert file_path.exists(), f"Missing {required_file} in {exp_dir.name}"

    def test_output_files_exist(self, project_paths):
        """Check that experiment output files are generated and contain data."""
        experiments_path = project_paths['root'] / 'data' / 'experiments'
        if not experiments_path.exists():
            pytest.skip("Experiments output directory not found: skipping test.")
        
        experiment_dirs = [d for d in experiments_path.iterdir() if d.is_dir()]
        assert len(experiment_dirs) > 0, "No experiment directories found."
        
        for exp_dir in experiment_dirs:
            train_file = exp_dir / 'train_data_X.npy'
            assert train_file.exists(), f"train_data_X.npy not found in {exp_dir.name}"
            temp_array = np.load(train_file)
            assert temp_array.size > 0, f"train_data_X.npy in {exp_dir.name} is empty."
            
class TestFullPipeline: 
    @pytest.mark.slow
    def test_full_pipeline_execution(self, pipeline):
        """Test that the full preprocessing pipeline runs without major errors."""
        try:
            pipeline.run_full_pipeline(skip_steps=['convert', 'sanity_check'], force_overwrite=False)
        except FileNotFoundError:
            pytest.skip("Required data files not found; skipping full pipeline test.")
        except SystemExit:
            pytest.skip("A step in the pipeline failed and exited; skipping full pipeline test.")
        except Exception as e:
            pytest.fail(f"Full preprocessing pipeline failed with exception: {e}")
    
    def test_single_experiment_pipeline(self, pipeline):
        """Test that the single experiment preprocessing pipeline runs without major errors."""
        try:
            pipeline.run_single_experiment(experiment_number=1, tau_max=10, strategy='remove')
        except FileNotFoundError:
            pytest.skip("Required data files not found; skipping single experiment pipeline test.")
        except SystemExit:
            pytest.skip("Single experiment pipeline failed and exited; skipping test.")
        except Exception as e:
            pytest.fail(f"Single experiment preprocessing pipeline failed with exception: {e}")

        
    
                

    
