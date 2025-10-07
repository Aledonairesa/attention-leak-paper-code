import os
from collections import Counter
import pandas as pd
import math
import random
import pickle
import argparse
import ast

from utils.preprocessing import *

def parse_args():
    parser = argparse.ArgumentParser(description="Generate a customized dataset")

    # Add arguments
    parser.add_argument('--datasets_path', type=str, required=True,
                        help="Path to the folder with the available datasets.")
    parser.add_argument('--num_samples', type=int, required=True,
                        help="The total number of samples in the final dataset.")
    parser.add_argument('--num_webs', type=int, required=True,
                        help="The number of different webs in the final dataset.")
    parser.add_argument('--num_datasets_per_web', type=int, required=True,
                        help="The number of different datasets per web available in the final dataset.")
    parser.add_argument('--levels', type=str, required=True,
                        help="List of levels (integers) as a string. Example: '[1, 2, 3]'")
    parser.add_argument('--level_weights', type=str, required=True,
                        help="List of level weights (floats) as a string. Example: '[0.4, 0.3, 0.3]'")
    parser.add_argument('--time_between_tasks', type=float, required=True,
                        help="The time in seconds between the beginning of the datasets used to generate negative samples.")
    parser.add_argument('--seed', type=int, required=True,
                        help="The seed to generate pairs of datasets to generate negative samples.")
    parser.add_argument('--output_folder', type=str, required=True,
                        help="Folder to save the output .pkl file.")

    # Parse the arguments
    return parser.parse_args()

def homogeneous_selection(elements, N):
    """
    Selects N elements evenly spaced from the given list.

    Parameters:
    elements (list): The list from which elements will be selected.
    N (int): The number of elements to select.

    Returns:
    list: A list of N elements selected homogeneously from the input list.
    """
    
    length = len(elements)
    if N > length:
        raise ValueError("N cannot be greater than the number of elements in the list.")
    
    # Calculate step size
    step = length / N
    
    # Select elements at regular intervals
    selected_elements = [elements[int(i * step)] for i in range(N)]
    
    return selected_elements
    
def deranged_shuffle(n, seed=None):
    """
    Generates two lists of integers from 0 to n-1, where the second list is a random 
    permutation (shuffle) of integers such that no element in the second list is in the 
    same position as its original index in the first list (a derangement).
    
    Parameters:
        n (int): The size of the lists (from 0 to n-1).
        
    Returns:
        tuple: A tuple containing two lists - list1 (ordered list from 0 to n-1) and 
               list2 (deranged shuffled list with no element at the same index).
    """
    # Set the seed for reproducibility, if provided
    if seed is not None:
        random.seed(seed)
        
    # Initialize two lists: list1 and list2
    list1 = list(range(n))
    list2 = list(range(n))
    
    # Using the Fisher-Yates algorithm to shuffle list2 until we get a derangement
    while True:
        random.shuffle(list2)
        
        # Check if any element in list2 matches the same position in list1
        if all(list1[i] != list2[i] for i in range(n)):
            break  # Break if no elements match their original positions
    
    return list1, list2

def distribute_integer(N, weights):
    """
    Distribute an integer value N proportionally based on a list of weights 
    using the Largest Remainder Method.

    Args:
        N (int): The total integer value to distribute.
        weights (list of float): A list of non-negative weights that define 
                                 the proportions for the distribution.

    Returns:
        list of int: A list of integers representing the final allocation 
                     of N based on the weights.
    """
    n_items = len(weights)
    
    # Multiply each weight by N to get the exact proportional shares
    n_i_float = [N * w for w in weights]
    
    # Take the floor of each share to get the initial integer allocations
    n_i_int = [int(math.floor(x)) for x in n_i_float]
    total_int = sum(n_i_int)
    
    # Calculate how many units are left to distribute
    D = N - total_int
    
    # Compute the remainders for each allocation
    remainders = [x - int_x for x, int_x in zip(n_i_float, n_i_int)]
    
    # Sort indices based on the descending order of remainders
    indices = sorted(range(n_items), key=lambda i: -remainders[i])
    
    # Initialize extra units to zero
    extra_units = [0] * n_items
    
    # Distribute the remaining units to the items with the largest remainders
    for i in range(D):
        idx = indices[i]
        extra_units[idx] += 1
    
    # Calculate the final allocations
    n_i_final = [n_i_int[i] + extra_units[i] for i in range(n_items)]
    
    return n_i_final

def choose_datasets(datasets_list, num_webs, num_datasets_per_web):
    """
    Homogeneously selects datasets from all available datasets.
    The homogeneous selection is applied at the web level, and
    at the datasets per web level.

    Parameters:
    - datasets_list (list): a list of strings of all the
                          available datasets.
    - num_webs (int): the number of different webs in the final
                    dataset.
    - num_datasets_per_web (int): the number of different datasets
                                per web available in the final
                                dataset.

    Returns:
    - list: a list of the selected datasets.
    """
    
    # Extract the number of webs and how many datasets per web there are
    datasets_list_webs = [name.split("-")[0] for name in datasets_list]
    webs_counts = Counter(datasets_list_webs)
    webs_list = list(webs_counts.keys())
    
    # Homogeneously select "num_webs" webs from the available webs
    selected_webs = homogeneous_selection(webs_list, num_webs)
    
    # For each web, homogeneously select "num_datasets_per_web" datasets
    selected_datasets = []
    for web in selected_webs:
        same_web_list = [dataset for dataset in datasets_list if web in dataset]
        selected_web_datasets = homogeneous_selection(same_web_list, num_datasets_per_web)
        selected_datasets.extend(selected_web_datasets)
    
    return selected_datasets

def preprocess_dataframe(df):
    """
    Preprocesses the data of a given dataset, performing multiple
    steps encapsulated in functions.
    
    Parameters:
    - df (pd.DataFrame): the dataframe to be processed
    
    Returns:
    - pd.DataFrame: the preprocessed dataframe
    """
    
    df = remove_frame_number(df)
    # df = remove_nan_ip_proto(df)
    df = merge_ports(df)
    df = process_frame_time_column(df)
    # df = add_endpoint_columns(df)
    df = filter_hosts(df)
    df = unify_hosts(df)
    df = replace_nans_in_tcp(df)
    df = leave_only_tcp_and_delete_ipproto(df)
    df = merge_ips_and_create_send_column(df, "172.17.")
    
    return df

def generate_dataset(datasets_path, num_samples, num_webs, num_datasets_per_web, 
                     levels, level_weights, time_between_tasks, seed, output_folder):
    """
    Generates a customized dataset given a list of paths to raw datasets.
    The webs are chosen homogeneously, the datasets from each web group are
    chosen homogeneosuly, the samples from each dataset are chosen homogeneously
    the positive and negative samples are as close as possible to 50-50%.
    
    Parameters:
    - datasets_path (list of str): a list of strings of all the available datasets.
    - num_samples (int): the total number of samples in the final dataset.
    - num_webs (int): the number of different webs in the final dataset.
    - num_datasets_per_web (int): the number of different datasets per web available
                                  in the final dataset.
    - levels (list of int): a list with the specific levels. A level is the number of
                            previous samples to compare the new frame to.
    - level_weights (list of float): the weights to customize the distribution of levels.
    - time_between_tasks (float): the time in seconds between the begginning of the datasets
                                  that are used to generate the "negative" samples.
    - seed (int): the seed to generate the pairs of datasets to generate the "negative" samples.
    - output_folder (str): folder to save the output .pkl file.
    
    Returns:
    - list of tuples of dictionaries and booleans: the dictionaries are the samples, the 
                        tuples are pairs of dictionaries and booleans, booleans indicating
                        whether the sample is a positive or negative instance.
    """
    
    # List of parameters to perform checks on
    params_to_check = [num_samples, num_webs, num_datasets_per_web]
    params_to_check.extend(levels)
    
    # Check if parameters are positive
    if not(all(parameter > 0 for parameter in params_to_check)):
        print("Error: num_samples, num_webs, num_datasets_per_web and levels should be positive.")
        return
    
    # Check if levels are integers
    if not(all(isinstance(parameter, int) for parameter in params_to_check)):
        print("Error: num_samples, num_webs, num_datasets_per_web and levels should be integers.")
        return
    
    # Check if weights are well formatted
    if (sum(level_weights) != 1) or (not all(weight > 0 for weight in level_weights)):
        print("Error: the weights should be positive and sum up to 1.")
        return
    
    # Check minimum number of samples is satisfied
    min_samples = num_webs*num_datasets_per_web*2*len(levels)
    if num_samples < min_samples:
        print(f"Error: the minimum number of samples for this configuration is {min_samples}.")
        return
    
    # Check there is at least one sample per level
    num_samples_per_dataset_per_level = num_samples/(num_webs*num_datasets_per_web*len(levels)*2)
    normalized_weights = [level_weight*len(levels) for level_weight in level_weights]
    approx_samples_per_weight = [norm_weight*num_samples_per_dataset_per_level for norm_weight in normalized_weights]
    if not all(x >= 1 for x in approx_samples_per_weight):
        print('Error: there will be some levels with 0 samples with this configuration.')
        return
    
    # Extract a list of all the available datasets
    datasets_list = os.listdir(datasets_path)
    
    # Homogeneously choose datasets from all the available datasets
    chosen_datasets = choose_datasets(datasets_list, num_webs, num_datasets_per_web)
    
    # Read the chosen datasets
    dataframes = []
    for dataset in chosen_datasets:
        path_to_csv = os.path.join(datasets_path, dataset, "data.csv")
        df = pd.read_csv(path_to_csv)
        dataframes.append(df)
    
    # Preprocess the chosen datasets
    preprocessed_dataframes = []
    for df in dataframes:
        df = preprocess_dataframe(df)
        preprocessed_dataframes.append(df)
    
    # Calculate the number of good and bad samples per dataset (as close as possible to 50-50%)
    num_datasets = len(chosen_datasets)
    num_samples_per_dataset = num_samples/num_datasets
    good_num_samples_per_dataset = int(num_samples_per_dataset/2)
    bad_num_samples_per_dataset = int(num_samples_per_dataset/2 + num_samples_per_dataset%2)
    
    # Calculate the weighted number of samples per level, per dataset
    good_weighted_num_samples_per_dataset = distribute_integer(good_num_samples_per_dataset, level_weights)
    bad_weighted_num_samples_per_dataset = distribute_integer(bad_num_samples_per_dataset, level_weights)
    
    # Save the samples as a list of tuples, each tuple a sample.
    # Each tuple contains a dictionary with a sampled df, and whether
    # or not the sample is positive or negative as a boolean.
    final_samples = []
    
    # Generate the good samples
    for level, samples in zip(levels, good_weighted_num_samples_per_dataset):
        for df in preprocessed_dataframes:
            df_len = df.shape[0]
            sampling_positions = homogeneous_selection(list(range(0, df_len-level)), samples)
            for position in sampling_positions:
                sample = df.iloc[position : position+level+1] # +1 to add the "new" frame
                final_samples.append((sample.to_dict(), True))
    
    
    # Generate a (deranged) permutation of the datasets, so pairs of non equal datasets
    # can be made to incorporate a frame from a different dataset
    idx_datasets_1, idx_datasets_2 = deranged_shuffle(num_datasets, seed)
    
    # Generate the bad samples
    for level, samples in zip(levels, bad_weighted_num_samples_per_dataset):
        for idx_df_1, idx_df_2 in zip(idx_datasets_1, idx_datasets_2):
            # Get the corresponding paired dataframes from the dataframes list
            df1, df2 = preprocessed_dataframes[idx_df_1], preprocessed_dataframes[idx_df_2]
            # Adjust second dataframe time to simulate the second task has been generated close to the first one
            df2['frame.time'] = df2['frame.time'] - df2['frame.time'].iloc[0] + df1['frame.time'].iloc[0] + time_between_tasks
            # Decide which dataframe ends sooner
            last_time_df1 = df1['frame.time'].iloc[-1]
            last_time_df2 = df2['frame.time'].iloc[-1]
            if last_time_df1 < last_time_df2:
                df_low, df_high = df1, df2
            else:
                df_low, df_high = df2, df1
            # For the dataframe that ends sooner, sample sets of rows (just like with good examples)
            df_low_len = df_low.shape[0]
            sampling_positions = homogeneous_selection(list(range(0, df_low_len-level)), samples)
            for position in sampling_positions:
                sample = df_low.iloc[position : position+level]
                # Append the "bad" row, corresponding to the first row in the other dataframe whose
                # frame.time is higher than the last frame.time of the sample
                sample_last_time = sample['frame.time'].iloc[-1]
                first_matching_row_df_high = df_high[df_high['frame.time'] > sample_last_time].iloc[0]
                sample_extended = pd.concat([sample, first_matching_row_df_high.to_frame().transpose()]).reset_index(drop=True)
                final_samples.append((sample_extended.to_dict(), False))
    
    # Save the list of samples to a .pkl file
    file_name = f'dataset_{num_samples}_{num_webs}_{num_datasets_per_web}_{levels}_{level_weights}_{time_between_tasks}_{seed}.pkl'
    with open(os.path.join(output_folder, file_name), 'wb') as f:
        pickle.dump(final_samples, f)
    
    return final_samples

def main():

    # Get command-line arguments
    args = parse_args()
    
    # Call main funtion
    generate_dataset(args.datasets_path,
                     int(args.num_samples),
                     int(args.num_webs),
                     int(args.num_datasets_per_web), 
                     ast.literal_eval(args.levels),
                     ast.literal_eval(args.level_weights),
                     float(args.time_between_tasks),
                     int(args.seed),
                     args.output_folder)

if __name__ == "__main__":
    main()