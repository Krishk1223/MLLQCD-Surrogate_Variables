import csv
import os
from pathlib import Path
import logging 
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import time
import sys
import numpy as np

def logging_setup(enable_logging=False, log_level=logging.INFO):
    if not enable_logging:
        logging.disable(logging.CRITICAL)
        return None
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    log_path = project_root / "logs"
    log_path.mkdir(exist_ok=True)

    #timestamp for log file:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = log_path / f"data_conversion_{timestamp}.log"
    
    #Log setup
    logging.basicConfig(level=log_level,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[
                            logging.FileHandler(log_file), #log file
                            logging.StreamHandler() #console output
                        ],
                        force=True)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging enabled. Saving log file to {log_file}")
    return logger

def main(enable_logging=False):

    #Log setup:
    logger = logging_setup(enable_logging=enable_logging)

    #Print standardized important log messages if regardless of availability of logger:
    def log_info(message):
        if logger:
            logger.info(f"INFO: {message}")
        else:
            print(f"INFO: {message}")
    
    def log_error(message):
        if logger:
            logger.error(message)
        else:
            print(f"ERROR: {message}")
    
    def log_warning(message):
        if logger:
            logger.warning(message)
        else:
            print(f"WARNING: {message}")

    #Path Setup:
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    download_path = project_root / "data"
    raw_path = os.path.join(download_path,"raw")
    processed_path = os.path.join(download_path,"processed") 

    for path in [raw_path, processed_path]:
        if not os.path.exists(path):
            os.makedirs(path)
            log_info(f"Directory {path} created for data")
    
    #Data file checks:
    raw_files = os.listdir(raw_path)
    if raw_files == []:
        log_warning(f"No files found in {raw_path}. Please add relevant data files and rerun the script.")
        askfile = input("Would you like to provide a filepath for a data file now? (y/n): ")
        if askfile.lower() == "y" or askfile.lower() == "yes":
            try:
                u_path = input("Please enter the full filepath for the data file which you would like to convert:")
                file_ext = file_allowed(u_path)
                if file_ext:
                    header_present = ask_header()
                    buffer_choice = ask_buffer()
                    tau_index = ask_tau()
                    data_to_csv_multithreaded(u_path, processed_path, file_ext, header=header_present, enable_logging=enable_logging, buffer=buffer_choice, zero_start=tau_index)
                else:
                    raise ValueError("File format not supported.")
            except Exception as e:
                log_error(f"Error finding file {e}")

    else:
        has_header = ask_header() #assume all files have same header status
        buffer_choice = ask_buffer()
        tau_index = ask_tau()
        for file in raw_files:
            file_ext = os.path.splitext(file)[1]
            if not file_allowed(os.path.join(raw_path, file)):
                log_error("File format not supported please use one of the following formats: .gpl, .txt")
                try:
                    user_path = input("Please enter the filepath for the data file which you would like to convert:")
                    file_ext = file_allowed(user_path)
                    if file_ext:
                        data_to_csv_multithreaded(user_path, processed_path, file_ext, header=has_header, enable_logging=enable_logging, buffer=buffer_choice, zero_start=tau_index)
                    else:
                        raise ValueError("File format not supported.")
                except Exception as e:
                    log_error(f"Error finding file {e}")
            else:
                input_file_path = os.path.join(raw_path, file)
                log_info(f"Processing file: {input_file_path}")
                data_to_csv_multithreaded(input_file_path, processed_path, file_ext, header=has_header, enable_logging=enable_logging, buffer=buffer_choice, zero_start=tau_index)

def ask_header():
    while True:
        response = input("Does the data file need a header row? (y/n): ")
        if response.lower() == 'y' or response.lower() == 'yes':
            return True
        elif response.lower() == 'n' or response.lower() == 'no':
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

def ask_buffer():
    while True:
        buffer_request = input("Would you like to use a larger buffer size for processing? (y/n): ")
        if buffer_request.lower() == 'y' or buffer_request.lower() == 'yes':
            return True
        elif buffer_request.lower() == 'n' or buffer_request.lower() == 'no':
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

def ask_tau():
    while True:
        tau_request = input("Start τ indexing from 0 or 1? (0/1, default = 1): ")
        if tau_request == '0':
            return True
        elif tau_request == '1' or tau_request == '':
            return False
        else:
            print("Invalid input. Please enter '0' or '1'.")

def optimal_settings(enable_logging=False, large_buffer=False):
    cpu_count = multiprocessing.cpu_count()
    optimal_workers = max(2, min(cpu_count -1, 8))

    #buffer sizes based on core count and if the user wants larger buffer
    if large_buffer:
        if cpu_count >= 8:
            buffer_size = 128*1024
        elif cpu_count >=4 and cpu_count <8:
            buffer_size = 64*1024
        else:
            buffer_size = 32*1024
    else:
        if cpu_count >=8:
            buffer_size = 64*1024
        elif cpu_count >= 4 and cpu_count < 8:
            buffer_size = 32*1024
        else:
            buffer_size = 16*1024

    if enable_logging:
        logging.info(f"Detected CPU cores: {cpu_count}")
        logging.info(f"Optimal multithreaded settings: Workers: {optimal_workers}, Buffer Size: {buffer_size} bytes")
    
    return optimal_workers, buffer_size

def file_allowed(input_path):
    accepted_extensions = [".gpl", ".txt"]
    file_extension = os.path.splitext(input_path)[1]
    if file_extension not in accepted_extensions:
        return False
    else:
        return file_extension
     
def csv_writer(label, configurations, output_path, buffer_size, header=False, enable_logging=False, zero_start=False):
    try:
        # Safe filename
        filename = label.replace(".ll", "").replace("/", "_").replace(".", "_").replace(" ", "_")
        output_file = os.path.join(output_path, f"{filename}.csv")

        if configurations:
            data_matrix = np.vstack(configurations) #vertically stacking arrays for efficiency
            if header:
                if zero_start:
                    header_row = ["config_id"] + [f"τ_{i}" for i in range(data_matrix.shape[1])]
                else:
                    header_row = ["config_id"] + [f"τ_{i}" for i in range(1, data_matrix.shape[1] + 1)]
                
                with open(output_file,'w') as file:
                    file.write(','.join(header_row) + '\n') #manual header write
                            
                config_ids = np.arange(1, len(data_matrix)+1).reshape(-1,1)
                id_data = np.hstack((config_ids, data_matrix)) #horizontal stacking
                with open(output_file,'a') as file: #append config id and data to header written file
                    np.savetxt(file, id_data, delimiter=",", fmt="%.16g")  
            else:
                np.savetxt(output_file, data_matrix, delimiter=",", fmt="%.16g")
        else:
            return "ERROR"
        if enable_logging:
            logging.info(f"Data for label {label} saved to {output_file} of dimensions {data_matrix.shape[0]} rows x {data_matrix.shape[1]} columns.")
        return "SUCCESS"
    except Exception as e:
        if enable_logging:
            logging.error(f"Error in writing CSV for label {label}: {e}")
        return "ERROR"

def data_to_csv_multithreaded(input_path, output_path, file_extension, header=False, enable_logging=False, buffer=False, zero_start=False):
    accepted_extensions = [".gpl", ".txt"]
    if file_extension not in accepted_extensions:
        raise ValueError("File format not supported for conversion.")

    #Worker and buffer settings
    optimal_workers, buffer_size = optimal_settings(enable_logging=enable_logging, large_buffer=buffer)

    dict_list = defaultdict(list) #using defaultdict for easier appending of non existent keys
    total_lines = 0

    if enable_logging:
        logging.info(f"Starting data parsing for file: {input_path}")

    start_time = time.time()
    skip_header = False
    try:
        with open(input_path, 'r', buffering=buffer_size) as infile:
            for line_count, line in enumerate(infile,1):
                line = line.strip()
                total_lines = line_count

                if not line or line.startswith('#'):
                    continue

                if header and not skip_header:
                    skip_header = True
                    # Check if line has numerical data
                    columns = line.split()
                    if len(columns) > 1:
                        try:
                            float(columns[1])  # Attempt to convert to float if data exists
                        except ValueError:
                            continue  #skip header
                    else:
                        continue  #skip header

                columns = line.split()
                if not columns:
                    continue

                labels = columns[0]
                if len(columns) > 1:
                    #converts data to float
                    try:
                        if zero_start: 
                            data_row = np.array(columns[1:], dtype=np.float64)
                        else:
                            data_row = np.array(columns[2:], dtype=np.float64)
                        dict_list[labels].append(data_row)
                    except ValueError:
                        if enable_logging:
                            logging.warning(f"Non-numeric data found in line {line_count} so could not convert data to float for label {labels}")
                        continue    
    except Exception as e:
        if enable_logging:
            logging.error(f"Error processing file {e}")
        else:
            print(f"ERROR: Error processing file {e}")
        return

    read_time = time.time() - start_time

    if not dict_list:
        if enable_logging:
            logging.warning(f"No data found in file {input_path} after parsing.")
        return
    if enable_logging:
        logging.info(f"Read {total_lines} lines from {input_path} in {read_time:.2f} seconds.")
        logging.info(f"Starting CSV writing with {optimal_workers} threads.")

    successful_writes = 0
    failed_writes = 0
    write_start_time = time.time()

    with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
        future_to_label = {
            executor.submit(csv_writer, label, configurations, output_path, buffer_size, header, enable_logging, zero_start): label
            for label, configurations in dict_list.items()
        }
        for future in as_completed(future_to_label):
            label = future_to_label[future]
            try:
                result = future.result()
                if result == "SUCCESS":
                    successful_writes += 1
                else:
                    failed_writes += 1
            except Exception as e:
                failed_writes += 1
                if enable_logging:
                    logging.error(f"Error occurred for label {label}: {e}")
    write_time = time.time() - write_start_time
    total_time_taken = time.time() - start_time
    
    #SUMMARY:
    summary = ["="*40 + " SUMMARY " + "="*40,
               f"Total labels processed: {len(dict_list)}",
               f"Successful CSV writes: {successful_writes}",
               f"Failed CSV writes: {failed_writes}",
               f"Time taken for reading data: {read_time:.2f} seconds",
               f"Time taken for writing CSVs: {write_time:.2f} seconds",
               f"Total time taken in processing: {total_time_taken:.2f} seconds",
               "="*90]
    
    for line in summary:
        if enable_logging:
            logging.info(line)
        else:
            print(line)

if __name__ == "__main__":
    enable_logs = False 
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.lower() in ['--log', '--verbose', '-l', '-v']:
                enable_logs = True
                break

    main(enable_logging=enable_logs)