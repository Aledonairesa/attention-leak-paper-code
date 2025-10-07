import os
import pickle
import pandas as pd
from tqdm import tqdm
import argparse

from utils.extract_features_utils import FeatureExtractor

def parse_args():
    parser = argparse.ArgumentParser(description="Extract features from dataset")

    # Add arguments
    parser.add_argument('--dataset_path', type=str, required=True,
                        help="Path to the dataset to extract features from.")
    parser.add_argument('--output_folder', type=str, required=True,
                        help="Folder to save the output .csv file.")
    parser.add_argument('--show_progress', type=str, default="True",
                        help="Whether or not to show a progress bar (True/False)")
    parser.add_argument('--asn', type=str, default="True",
                        help="Whether or not to extract ASN-related features (True/False)")

    return parser.parse_args()

def extract_features(dataset, show_progress=True, asn=True):
    """
    Parameters:
    - dataset (list of tuples): the dataset with all the samples.
    - show_progress (bool): whether or not to display a progress bar.
    - asn (bool): whether or not to extract ASN-related features.
    
    Returns:
    - full_feature_dataset (pd.DataFrame): the final dataframe.
    """
    
    new_rows_list = []
    iterator = tqdm(dataset, desc="Processing samples", unit="sample") if show_progress else dataset

    for sample in iterator:
        data = sample[0]
        sample_type = sample[1]
        
        FE = FeatureExtractor(data)

        # ---Time---
        # FE.extract_time_to_last_frame()

        # ---Send/Receive---
        FE.extract_send_in_last_send()
        FE.extract_send_fraction_in_send_50()
        # FE.extract_send_fraction_in_send_ALL()
        FE.extract_current_send()

        # ---ASN---
        if asn:
            FE.extract_country_in_country()
            FE.extract_country_in_last_country()
            FE.extract_country_fraction_in_country_50()
            # FE.extract_country_fraction_in_country_ALL()
            FE.extract_desc_in_desc()
            FE.extract_desc_in_last_desc()
            FE.extract_desc_fraction_in_desc_50()
            # FE.extract_desc_fraction_in_desc_ALL()
            FE.extract_network_in_network()
            FE.extract_network_in_last_network()
            FE.extract_network_fraction_in_network_50()
            # FE.extract_network_fraction_in_network_ALL()

        # ---IP---
        FE.extract_ip_in_ip()
        FE.extract_ip_in_last_ip()

        # ---Source IP---
        # FE.extract_ipsrc_in_ipsrc()
        # FE.extract_ipsrc_in_ipdst()
        # FE.extract_ipsrc_in_last_ipsrc()
        # FE.extract_ipsrc_in_last_ipdst()

        # ---Destination IP---
        # FE.extract_ipdst_in_ipdst()
        # FE.extract_ipdst_in_ipsrc()
        # FE.extract_ipdst_in_last_ipdst()
        # FE.extract_ipdst_in_last_ipsrc()

        # ---IP Protocol---
        # FE.extract_ipproto_in_ipproto()
        # FE.extract_ipproto_in_last_ipproto()
        # FE.extract_ipproto_fraction_in_ipproto()
        # FE.extract_current_ipproto_features()

        # --- TCP flags ---
        FE.extract_tcpflag_in_last50_tcpflag()
        FE.extract_tcpflag_fraction_in_tcpflag_50()
        # FE.extract_tcpflag_fraction_in_tcpflag_ALL()
        FE.extract_current_tcpflag_features_2()
        FE.extract_previous_tcpflag_features_2()

        # --- Frame length ---
        FE.extract_diff_lenframe_to_last()
        FE.extract_diff_lenframe_to_last_mean_50()
        # FE.extract_diff_lenframe_to_last_mean_ALL()
        FE.extract_lenframe_last_variance_50()
        # FE.extract_lenframe_last_variance_ALL()
        FE.extract_current_framelen_features()

        # --- Source port ---
        FE.extract_portsrc_in_portsrc()
        FE.extract_portsrc_in_portdst()
        FE.extract_portsrc_in_last_portsrc()
        FE.extract_portsrc_in_last_portdst()

        # --- Destination port ---
        FE.extract_portdst_in_portdst()
        FE.extract_portdst_in_portsrc()
        FE.extract_portdst_in_last_portdst()
        FE.extract_portdst_in_last_portsrc()

        # --- Source endpoint ---
        # FE.extract_endpointsrc_in_endpointsrc()
        # FE.extract_endpointsrc_in_endpointdst()
        # FE.extract_endpointsrc_in_last_endpointsrc()
        # FE.extract_endpointsrc_in_last_endpointdst()

        # --- Destination endpoint ---
        # FE.extract_endpointdst_in_endpointdst()
        # FE.extract_endpointdst_in_endpointsrc()
        # FE.extract_endpointdst_in_last_endpointdst()
        # FE.extract_endpointdst_in_last_endpointsrc()

        # --- Public/Private IPs ---
        # FE.extract_current_last_private_features()

        # ---Other---
        FE.extract_length_previous()

        new_row = FE.get_new_row()
        new_row['target'] = 1 if sample_type else 0
        new_rows_list.append(new_row)
    
    return pd.DataFrame(new_rows_list)

def main():
    args = parse_args()
    
    with open(args.dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    show_progress = args.show_progress == "True"
    asn = args.asn == "True"

    feature_df = extract_features(dataset, show_progress=show_progress, asn=asn)

    # Ensure output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    base = os.path.splitext(os.path.basename(args.dataset_path))[0]
    output_csv_file = os.path.join(args.output_folder, f"{base}.csv")
    feature_df.to_csv(output_csv_file, index=False)

if __name__ == "__main__":
    main()
