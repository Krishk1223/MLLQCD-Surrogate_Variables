import numpy as np
import pandas as pd 
from pathlib import Path

"""
PREPROCESSING STEP 3: Averages time sources in processed CSV files and saves the
                      averaged data as csvs and the errors as npy files.
                      Useful for truth analysis. 
                      Model training will use unaveraged data.
"""

CONFIG = {
    "time_sources": 4,
    "delete_originals": False
}

def time_source_averaging():
    #Path setup:
    project_root = Path(__file__).resolve().parent.parent.parent #moves to the project root directory which is MLLQCD
    input_path = project_root / "data" / "processed" / "unaveraged_data"
    output_path = project_root / "data" / "processed"
    time_sources = CONFIG["time_sources"]
    delete_originals = CONFIG["delete_originals"]
    average_time_sources(input_path, output_path, pattern="*.csv", time_sources=time_sources, delete_originals=delete_originals)

def average_time_sources(input_path: Path, output_path: Path, pattern="*.csv", time_sources=4, delete_originals=False):
    """Averages time sources in processed CSV files and saves the averaged data as csvs and the errors as npy files."""
    #Path setup for averaging:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path {input_path} does not exist")
    
    csv_files = list(input_path.glob(pattern))
    if not csv_files:
        print(f"No CSV files found in {input_path}")
        return
    print(f"Averaging time sources for {len(csv_files)} CSV files from {input_path}:")

    output_path.mkdir(exist_ok=True, parents=True)
    subfolders = ["averaged_data", "jackknife_mean", "jackknife_errors", 'jackknife_samples']
    for subfolder in subfolders:
        (output_path / subfolder).mkdir(exist_ok=True, parents=True)
    
    processed_files = []  # Track successfully processed files
    
    #CSV processing and averaging:
    for csv_file in csv_files:
        try:
            ensemble_name = csv_file.stem
            df = pd.read_csv(csv_file)
            if df.empty:
                print(f"Skipping empty file: {csv_file}")
                continue
            
            #Getting rid of config_id and τ_0 columns if they exist as well as header if it exists:
            if 'config_id' in df.columns:
                df = df.drop(columns=['config_id'], errors='ignore')
            
            if 'τ_0' in df.columns:
                df = df.drop(columns=['τ_0'], errors='ignore')
            
            first_row = df.iloc[0]
            if first_row.astype(str).str.contains('τ_').any():
                df = df.iloc[1:].reset_index(drop=True)
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()  # Fixed to include all numeric types
            if len(numeric_cols) == 0:
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                numeric_cols = df.columns.tolist()

            data = df[numeric_cols].values.astype(np.float64)

            #averaging over time sources:
            n_rows = len(data)  # Remove the incorrect +1
            n_configs = n_rows // time_sources

            if n_rows % time_sources != 0:
                print(f"Warning: Number of rows {n_rows} is not a multiple of time_sources {time_sources} in file {csv_file}. Some data may be ignored.")
                data = data[:n_configs * time_sources] #trim excess rows
            
            reshaped_data = data.reshape(n_configs, time_sources, -1) #3d array
            averaged_data = np.mean(reshaped_data, axis=1) # naive mean over time sources

            #jackknife stats:
            jackknife_mean, jackknife_error, jackknife_samples = jackknife_stats(averaged_data) #stats over time source averaged configs
            
            #CSV for averaged data:
            averaged_df = pd.DataFrame(averaged_data, columns=numeric_cols)
            averaged_csv_path = output_path / "averaged_data" / f"{ensemble_name}_averaged.csv"
            averaged_df.to_csv(averaged_csv_path, index=False)

            #jackknife mean and error npy files:
            jackknife_mean_path = output_path / "jackknife_mean" / f"{ensemble_name}_jackknife_mean.npy"
            jackknife_error_path = output_path / "jackknife_errors" / f"{ensemble_name}_jackknife_error.npy"
            jackknife_samples_path = output_path / "jackknife_samples" / f"{ensemble_name}_jackknife_samples.npy"
            np.save(jackknife_mean_path, jackknife_mean)
            np.save(jackknife_error_path, jackknife_error)
            np.save(jackknife_samples_path, jackknife_samples)  # Saving jackknife samples as well

            print(f"Processed {ensemble_name}: {n_configs} configs averaged over {time_sources} time sources.")
            
            # Mark file as successfully processed
            processed_files.append(csv_file)
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            print(f"Skipping deletion of {csv_file} due to processing error")
            continue
    
    # Delete original files after successful processing
    if delete_originals and processed_files:
        print(f"\nDeleting {len(processed_files)} original CSV files...")
        for csv_file in processed_files:
            try:
                csv_file.unlink()  # Delete the file
                print(f"  Deleted: {csv_file.name}")
            except Exception as e:
                print(f"  Failed to delete {csv_file.name}: {e}")
        
        print(f"Cleanup complete: {len(processed_files)} files deleted")
    elif not delete_originals:
        print("Original files preserved (delete_originals=False)")

def jackknife_stats(data: np.ndarray):
    """Computes jackknife averages and errors for a given amount of time sources.
       Data shape: (n_configs x n_time_columns) E.g for a data input with 96 correlator 
       values and 4 time sources per config. data shape will be (96, 4)
       Uses numpy vectorised operations to make it faster in case of large datasets."""
    N = data.shape[0]
    
    total_sum = np.sum(data, axis=0) #
    jackknife_samples = (total_sum-data)/(N-1)
    
    jackknife_mean = np.mean(jackknife_samples, axis=0)
    jackknife_difference = jackknife_samples-jackknife_mean
    jackknife_variance = ((N-1)/N)*np.sum(jackknife_difference**2, axis=0)
    jackknife_error = np.sqrt(jackknife_variance)
    
    return jackknife_mean, jackknife_error, jackknife_samples

if __name__ == "__main__":
    time_source_averaging()


    
