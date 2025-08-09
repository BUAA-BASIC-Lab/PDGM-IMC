import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # for one_hot
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor


def process_file(args):
    """
    Read a single CSV and return numpy arrays for x, w, error.

    Args:
        args (tuple): (file_path, max_rows_per_file)

    Returns:
        tuple: (x_data, w_data, error_data)
    """
    file, max_rows_per_file = args
    df = pd.read_csv(file, nrows=max_rows_per_file)

    x_data = df['x'].values

    w_cols = [col for col in df.columns if col.startswith('w_')]
    w_data = df[w_cols].values

    error_cols = [col for col in df.columns if col.startswith('error_')]
    error_data = df[error_cols].values

    return x_data, w_data, error_data


def load_readback_data_multidie(
    base_dir,
    dtype=torch.float32,
    block_size=(16, 16),
    max_rows_per_file=4096,
    num_workers=4,
    flatten=True
):
    """
    Load readback data from multiple dies under `base_dir/die_*` with multiprocessing.
    Die IDs are encoded to one-hot vectors and returned as `die_tensor`.

    Args:
        base_dir (str): Root directory containing subfolders like 'die_0', 'die_1', ...
        dtype (torch.dtype): Tensor dtype.
        block_size (tuple): (H, W). H is used when expanding x in flatten mode.
        max_rows_per_file (int): Max rows to read per CSV.
        num_workers (int): Processes used by multiprocessing Pool.
        flatten (bool): If True, return flat shapes.
                        If False, return 4D tensors.

    Returns:
        x_tensor (Tensor)
        w_tensor (Tensor)
        error_tensor (Tensor)
        die_tensor (Tensor): one-hot, shape (N, n_dies)
    """
    # Sanity checks
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Directory {base_dir} not found.")

    # Find all die_* subfolders
    die_dirs = [d for d in glob.glob(os.path.join(base_dir, "die_*")) if os.path.isdir(d)]
    if not die_dirs:
        raise FileNotFoundError(f"No 'die_*' subfolders found in {base_dir}.")

    # Map die_id -> contiguous index [0, n_dies)
    die_ids = sorted([int(os.path.basename(d).split('_')[1]) for d in die_dirs])
    n_dies = len(die_ids)
    die_id_to_idx = {die_id: idx for idx, die_id in enumerate(die_ids)}

    H, W = block_size

    all_x, all_w, all_error, all_die_ids = [], [], [], []
    for die_dir in die_dirs:
        die_id = int(os.path.basename(die_dir).split('_')[1])
        file_pattern = os.path.join(die_dir, "**", "readback_error_w=*_x=*.csv")
        files = sorted(glob.glob(file_pattern, recursive=True))

        if not files:
            print(f"Warning: no CSV files matched in {die_dir}.")
            continue

        # Load all CSVs in this die with multiprocessing
        with Pool(processes=num_workers) as pool:
            results = pool.map(process_file, [(f, max_rows_per_file) for f in files])

        # Accumulate arrays and record die index per row
        for x_data, w_data, error_data in results:
            all_x.append(x_data)
            all_w.append(w_data)
            all_error.append(error_data)
            all_die_ids.append(np.full(x_data.shape[0], die_id_to_idx[die_id], dtype=np.int64))

    print(f"Loaded {len(die_dirs)} die folders from {base_dir}, up to {max_rows_per_file} rows per file.")

    # Concatenate along the sample dimension
    x = np.concatenate(all_x)                          
    w = np.concatenate(all_w, axis=0)                 
    error = np.concatenate(all_error, axis=0)         
    die_ids_array = np.concatenate(all_die_ids)       

    N = len(x)

    # One-hot encode die indices -> (N, n_dies)
    die_tensor = F.one_hot(torch.tensor(die_ids_array, dtype=torch.long), num_classes=n_dies).to(dtype)

    if flatten:
        x_expanded = np.tile(x[:, np.newaxis], (1, H))

        x_tensor = torch.tensor(x_expanded, dtype=dtype)
        w_tensor = torch.tensor(w, dtype=dtype)
        error_tensor = torch.tensor(error, dtype=dtype)
    else:
        x_expanded = np.expand_dims(x, 1).repeat(H, axis=1)   
        x_expanded = np.expand_dims(x_expanded, 2).repeat(W, axis=2)  
        x_tensor = torch.tensor(x_expanded, dtype=dtype).unsqueeze(1) 

        w_reshaped = w.reshape(N, 1, H, W)
        error_reshaped = error.reshape(N, 1, H, W)
        w_tensor = torch.tensor(w_reshaped, dtype=dtype)
        error_tensor = torch.tensor(error_reshaped, dtype=dtype)

    # Quick summaries
    print(f"Total samples: {N}")
    print(f"x_tensor shape: {tuple(x_tensor.shape)}")
    print(f"w_tensor shape: {tuple(w_tensor.shape)}")
    print(f"error_tensor shape: {tuple(error_tensor.shape)}")
    print(f"die_tensor shape: {tuple(die_tensor.shape)}")

    return x_tensor, w_tensor, error_tensor, die_tensor

def load_infer_tensors_from_directory(directory, flatten=False, device='cpu', num_workers=4):
    """
    Load 'y' and 'error' tensors from all CSV files in the specified directory.

    Args:
        directory (str): Path to the directory containing CSV files.
        flatten (bool): If True, flatten the loaded tensors to 1D.
        device (str): Target device to store tensors ('cpu' or 'cuda').
        num_workers (int): Number of parallel threads for loading.

    Returns:
        tuple:
            y_tensor (torch.Tensor): Concatenated tensor of all 'y' values from CSV files.
            error_tensor (torch.Tensor): Concatenated tensor of all 'error' values from CSV files.

    Notes:
        - Assumes each CSV file contains data compatible with load_csv_to_tensor().
        - Uses ThreadPoolExecutor for parallel I/O to speed up loading.
    """
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]

    y_tensors = []
    error_tensors = []

    # Parallel load each CSV file
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(load_csv_to_tensor, file_path, flatten, device) for file_path in file_paths]
        for future in futures:
            y_tensor, error_tensor = future.result()
            y_tensors.append(y_tensor)
            error_tensors.append(error_tensor)

    # Concatenate all tensors along the first dimension
    y_tensor = torch.cat(y_tensors, dim=0)
    error_tensor = torch.cat(error_tensors, dim=0)

    return y_tensor, error_tensor

def load_csv_to_tensor(file_path, flatten=False, device='cpu'):
    """
    Load a single CSV file into 'y' and 'error' tensors.

    Args:
        file_path (str): Path to the CSV file.
        flatten (bool): If True, flatten y/error arrays to 1D and limit zero-y rows to 100.
                        If False, preserve array shape and limit zero-y rows to 1000.
        device (str): Device to store the tensors ('cpu' or 'cuda').

    Returns:
        tuple:
            y_tensor (torch.Tensor): Loaded 'y' values as float32 tensor.
            error_tensor (torch.Tensor): Loaded 'error' values as float32 tensor.

    Notes:
        - 'y' columns are identified by prefix "y".
        - 'error' columns are identified by prefix "error".
        - Zero-y rows are downsampled to avoid imbalance.
    """
    df = pd.read_csv(file_path)

    # Identify y and error columns
    y_columns = sorted([col for col in df.columns if col.startswith('y')])
    error_columns = sorted([col for col in df.columns if col.startswith('error')])

    y_data = df[y_columns].values
    error_data = df[error_columns].values

    if flatten:
        # Flatten to shape (N, 1)
        y_data_flat = y_data.reshape(-1, 1)
        error_data_flat = error_data.reshape(-1, 1)

        # Mask for rows where y == 0
        mask_all_zero = (y_data_flat[:, 0] == 0)
        non_zero_y_data = y_data_flat[~mask_all_zero]
        non_zero_error_data = error_data_flat[~mask_all_zero]

        zero_y_data = y_data_flat[mask_all_zero]
        zero_error_data = error_data_flat[mask_all_zero]

        # Limit zero-y rows to 100
        if len(zero_y_data) > 100:
            zero_y_data = zero_y_data[:100]
            zero_error_data = zero_error_data[:100]

        # Combine and convert to tensors
        y_data_final = np.vstack([non_zero_y_data, zero_y_data])
        error_data_final = np.vstack([non_zero_error_data, zero_error_data])

    else:
        # Mask for rows where all y == 0
        mask_all_zero = (y_data == 0).all(axis=1)
        non_zero_y_data = y_data[~mask_all_zero]
        non_zero_error_data = error_data[~mask_all_zero]

        zero_y_data = y_data[mask_all_zero]
        zero_error_data = error_data[mask_all_zero]

        # Limit zero-y rows to 1000
        if len(zero_y_data) > 1000:
            zero_y_data = zero_y_data[:1000]
            zero_error_data = zero_error_data[:1000]

        # Combine and convert to tensors
        y_data_final = np.vstack([non_zero_y_data, zero_y_data])
        error_data_final = np.vstack([non_zero_error_data, zero_error_data])

    y_tensor = torch.tensor(y_data_final, dtype=torch.float32, device=device)
    error_tensor = torch.tensor(error_data_final, dtype=torch.float32, device=device)

    return y_tensor, error_tensor


def load_infer_tensors_from_directory(directory, flatten=False, device='cpu', num_workers=4):
    """
    Load and combine 'y' and 'error' tensors from all CSV files in a directory.

    Args:
        directory (str): Directory containing CSV files.
        flatten (bool): Passed to load_csv_to_tensor() to control flattening behavior.
        device (str): Device to store the tensors ('cpu' or 'cuda').
        num_workers (int): Number of threads for parallel file loading.

    Returns:
        tuple:
            y_tensor (torch.Tensor): Concatenated 'y' tensor from all files.
            error_tensor (torch.Tensor): Concatenated 'error' tensor from all files.

    Notes:
        - Uses ThreadPoolExecutor for parallel I/O.
        - File names must end with '.csv'.
    """
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]

    y_tensors = []
    error_tensors = []

    # Parallel CSV loading
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(load_csv_to_tensor, file_path, flatten, device) for file_path in file_paths]
        for future in futures:
            y_tensor, error_tensor = future.result()
            y_tensors.append(y_tensor)
            error_tensors.append(error_tensor)

    # Concatenate all tensors along the first dimension
    y_tensor = torch.cat(y_tensors, dim=0)
    error_tensor = torch.cat(error_tensors, dim=0)

    return y_tensor, error_tensor
