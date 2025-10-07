import pandas as pd
import numpy as np
from dateutil import parser
import pytz
from ipwhois import IPWhois

# REMOVING frame.number COLUMN
"""
This column was artificially created in the data collection process. Therefore it 
won't be present in a real scenario and should be deleted since it can't be used.
"""
def remove_frame_number(df):
    """
    Removes the 'frame.number' column from the DataFrame if it exists.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to modify.
    
    Returns:
    pd.DataFrame: The DataFrame without the 'frame.number' column, if it was present.
    """
    if 'frame.number' in df.columns:
        return df.drop(columns=['frame.number'])
    else:
        raise ValueError("Column 'frame.number' does not exist in the DataFrame.")
        
 
# REMOVING ROWS WITH NaN VALUES IN ip.proto COLUMN
"""
Rows without this column don't make sense, as all frames must be sent under a protocol. 
This means these rows are incorrect or corrupt in some way, thus potentially adding 
noise to to the dataset.
"""
def remove_nan_ip_proto(df):
   """
   Remove rows from a DataFrame where the 'ip.proto' column contains NaN values.

   Args:
       dataframe (pd.DataFrame): The input DataFrame.

   Returns:
       pd.DataFrame: A DataFrame with rows containing NaN in 'ip.proto' removed.
                     If 'ip.proto' column doesn't exist, returns the original DataFrame.
   """
   if 'ip.proto' in df.columns:
       cleaned_dataframe = df.dropna(subset=['ip.proto'])
       return cleaned_dataframe
   else:
       raise ValueError("Column 'ip.proto' does not exist in the dataset.")
 

# MERGING TCP AND UDP SOURCE AND DESTINATION PORT COLUMNS INTO A SINGLE srcport AND dstport COLUMNS
"""
The information of the protocol is already present in the `ip.proto` column. Merging 
the column avoids unnecessary NaN values.
"""
def merge_ports(df):
    """
    Merges specific port columns into single columns. Uses .combine_first() to fill
    the NaN values in one column with the values from the other column.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the columns to merge.

    Returns:
    pd.DataFrame: A DataFrame with new columns 'port.src' and 'port.dst' containing the merged values.
    """
    df = df.copy()
    
    # Check for missing columns
    required_columns = ['tcp.srcport', 'udp.srcport', 'tcp.dstport', 'udp.dstport']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    # Return the original DataFrame if any columns are missing
    if missing_columns:
        print(f"Missing columns: {', '.join(missing_columns)}. The function requires all these columns to proceed.")
        return df
    
    # Merge 'tcp.srcport' and 'udp.srcport' into 'srcport'
    df['port.src'] = df['tcp.srcport'].combine_first(df['udp.srcport']).astype(str)
    
    # Merge 'tcp.dstport' and 'udp.dstport' into 'dstport'
    df['port.dst'] = df['tcp.dstport'].combine_first(df['udp.dstport']).astype(str)
    
    # Delete the original columns after merging
    df.drop(columns=['tcp.srcport', 'udp.srcport', 'tcp.dstport', 'udp.dstport'], inplace=True)
    
    return df


# TREATING THE frame.time COLUMN TO CONVERT IT INTO FLOAT (UNIX TIMESTAMP)
"""
This is a necessary step if arithmetic is to be done with the `frame.time` column, which originally 
is a string.
"""
def process_frame_time_column(df):
    """
    Converts the 'frame.time' column in a DataFrame from datetime strings with timezone abbreviations
    to Unix timestamps. It handles errors and returns a new DataFrame with the processed column.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the 'frame.time' column to process.
    
    Returns:
    pd.DataFrame: A new DataFrame with the 'frame.time' column converted to Unix timestamps.
    """
    # Mapping from timezone abbreviations to pytz timezones
    timezone_mapping = {
        'CEST': 'Europe/Berlin',
        'HORA ESTÁNDAR ROMANCE': 'CET'
    }
    
    def datetime_to_unix_timestamp(date_str):
        try:
            # Split the datetime string and the timezone abbreviation
            for i, char in enumerate(reversed(date_str)):
                if char.isdigit():
                    datetime_str, tz_abbr = date_str[:-i], date_str[-i+1:] # +1 for extra space
                    break
            
            # Parse the datetime string without timezone
            naive_dt = parser.parse(datetime_str)
            
            # Get the corresponding pytz timezone
            timezone_str = timezone_mapping.get(tz_abbr.upper())
            if not timezone_str:
                raise ValueError(f"Timezone abbreviation '{tz_abbr}' not found in mapping.")
            
            # Create a timezone object
            timezone = pytz.timezone(timezone_str)
            
            # Create a timezone-aware datetime object
            aware_dt = timezone.localize(naive_dt)
            
            # Convert to UTC
            utc_dt = aware_dt.astimezone(pytz.UTC)
            
            # Convert to Unix timestamp (including fractional seconds)
            unix_timestamp = utc_dt.timestamp()
            
            return unix_timestamp
        except Exception as e:
            print(f"Error processing '{date_str}': {e}")
            return np.nan
    
    # Create a copy of the DataFrame to avoid modifying the original
    new_df = df.copy()
    
    # Apply the conversion to the 'frame.time' column
    new_df['frame.time'] = new_df['frame.time'].apply(datetime_to_unix_timestamp)
    
    return new_df
 
 
# ADD COLUMNS TO PROVIDE ENDPOINT INFORMATION, COMBINING IPs AND PORTS
"""
Endpoints are an important concept in computer networking, so directly specifiying them might 
prove beneficial.
"""
def add_endpoint_columns(df):
   """
   Adds 'endpoint.src' and 'endpoint.dst' columns to the DataFrame by combining IP
   addresses and port numbers from 'ip.src' and 'port.src' for source, and 'ip.dst' 
   and 'port.dst' for destination.

   Parameters:
   df (pandas.DataFrame): DataFrame with columns 'ip.src', 'ip.dst', 'port.src', and 'port.dst'

   Returns:
   pandas.DataFrame: DataFrame with added 'endpoint.src' and 'endpoint.dst' columns

   """
   # Check for missing columns
   required_columns = {'ip.src', 'ip.dst', 'port.src', 'port.dst'}
   missing_columns = required_columns - set(df.columns)
   
   if missing_columns:
       raise ValueError(f"DataFrame is missing required columns: {', '.join(missing_columns)}")
   
   # Ensure ports are treated as strings for concatenation
   df['endpoint.src'] = df['ip.src'] + ':' + df['port.src'].astype(str)
   df['endpoint.dst'] = df['ip.dst'] + ':' + df['port.dst'].astype(str)
   
   return df


# KEEPING ONLY IP ADDRESSES FROM THE HOST THAT GENERATED THE DATA
"""
This is done to ensure no information from other hosts has been included in the dataset while 
generating the data.
"""
def filter_hosts(df):
    """
    Filters rows in a DataFrame based on IP address occurrences.
    This is done to ensure no information from other hosts has been
    included in the dataset while generating the data.

    This function first identifies the most frequent IP address in the range 
    '172.17.X.X' across the 'ip.src' and 'ip.dst' columns combined. It then 
    removes all rows that contain IP addresses within this range, except for 
    those containing the most frequent IP address. 

    Parameters:
    df (pandas.DataFrame): A DataFrame containing 'ip.src' and 'ip.dst',
                           with IP addresses as strings.

    Returns:
    pandas.DataFrame: A filtered DataFrame with rows containing IPs in the 
                      specified range removed, except for the most frequent IP.
    """
    # Filter rows with IPs in the range 172.17.X.X in both columns
    ip_range = '172.17.'
    mask = df['ip.src'].str.startswith(ip_range) | df['ip.dst'].str.startswith(ip_range)
    df_filtered = df[mask]

    # Count occurrences of each IP in the range in both 'ip.src' and 'ip.dst' columns
    ip_counts = pd.concat([df_filtered['ip.src'], df_filtered['ip.dst']]).value_counts()
    
    # Filter only the IPs in the specified range
    ip_counts = ip_counts[ip_counts.index.str.startswith(ip_range)]
    
    # Find the most frequent IP in the range
    if not ip_counts.empty:
        most_frequent_ip = ip_counts.idxmax()
    
        # Filter the original DataFrame, keeping only rows with the most frequent IP in the specified range
        final_df = df[~mask | (df['ip.src'] == most_frequent_ip) | (df['ip.dst'] == most_frequent_ip)]
    else:
        # If there are no IPs in the range, return the original dataframe
        final_df = df

    return final_df


# SIMULATE THE HOST IS THE SAME FOR ALL DATASETS (WITH IP 172.17.0.3)
"""
This is done so that the final dataset makes sense when we combine the sub datasets, which are 
potentially from different hosts.
"""
def unify_hosts(df):
    """
    Replaces IP addresses in the 'ip.src' and 'ip.dst' columns of a DataFrame.
    Any IP addresses that match the pattern '172.17.X.X' are replaced with '172.17.0.3'.
    
    Parameters:
    df (pd.DataFrame): A DataFrame containing 'ip.src' and 'ip.dst' columns with IP addresses as strings.
    
    Returns:
    pd.DataFrame: A DataFrame with specified IP addresses replaced.
    """
    # IP pattern to look for
    ip_pattern = '172.17.'
    
    # Replace in both 'ip.src' and 'ip.dst' columns
    df['ip.src'] = df['ip.src'].apply(lambda x: '172.17.0.3' if ip_pattern in str(x) else x)
    df['ip.dst'] = df['ip.dst'].apply(lambda x: '172.17.0.3' if ip_pattern in str(x) else x)
    
    return df


# REPLACE ACTUAL NAN VALUES FOR "nan" STRINGS IN `tcp.flags.str`
"""
This way NaNs can be used as information in the feature extraction process.
"""
def replace_nans_in_tcp(df):
    """
    Replaces all NaN values in the 'tcp.flags.str' column of the given dataframe with the string 'nan'.
    
    Parameters:
    df (pandas.DataFrame): The dataframe containing the 'tcp.flags.str' column.
    
    Returns:
    pandas.DataFrame: The dataframe with NaN values in the 'tcp.flags.str' column replaced by 'nan'.
    """
    # Check if the 'tcp.flags.str' column exists
    if 'tcp.flags.str' in df.columns:
        # Replace NaN values with the string 'nan'
        df['tcp.flags.str'] = df['tcp.flags.str'].fillna('nan')
    else:
        print("The column 'tcp.flags.str' does not exist in the dataframe.")
    return df

# LEAVING ONLY TCP FRAMES
"""
Almost all frames are TCP. Leaving the small minorities can confuse the model
and make the final system perform worse.
"""
def leave_only_tcp_and_delete_ipproto(df):
    """
    This function filters a DataFrame to include only rows where the 'ip.proto' column equals 6.0,
    then deletes the 'ip.proto' column from the resulting DataFrame.
    """
    filtered_df = df[df['ip.proto'] == 6.0].copy()  # Filter rows and create a copy to avoid modifying original
    filtered_df.drop(columns=['ip.proto'], inplace=True)  # Remove 'ip.proto' column
    return filtered_df

# MERGING IP.SRC AND IP.DST, IGNORING HOST IP, THEN CREATING SEND/RECEIVE BINARY COLUMN
"""
The host IP doesn't add any important information other than its position in either 
ip.src or ip.dst which is actually whether the information is being sent or received.
"""
def merge_ips_and_create_send_column(df, start_str):
    """
    Processes the given dataframe by performing the following operations:
    1. Converts all values that start with `start_str` to NaN in columns 'ip.src' and 'ip.dst'.
    2. Creates a new column 'send', which is 1 if the NaN is in the 'ip.src' column, and 0 if in the 'ip.dst' column.
    3. Merges 'ip.src' and 'ip.dst' into a single column, filling NaNs in one column with the value from the other.
    4. Deletes the original 'ip.src' and 'ip.dst' columns.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        start_str (str): The string to match at the start of values for conversion to NaN.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    # Convert values starting with start_str to NaN in 'ip.src' and 'ip.dst'
    df['ip.src'] = df['ip.src'].apply(lambda x: np.nan if isinstance(x, str) and x.startswith(start_str) else x)
    df['ip.dst'] = df['ip.dst'].apply(lambda x: np.nan if isinstance(x, str) and x.startswith(start_str) else x)
    
    # Create the 'send' column
    df['send'] = df.apply(lambda row: 1 if pd.isna(row['ip.src']) else (0 if pd.isna(row['ip.dst']) else np.nan), axis=1)
    
    # Merge 'ip.src' and 'ip.dst' into a single column
    df['ip'] = df['ip.src'].combine_first(df['ip.dst'])
    
    # Delete the 'ip.src' and 'ip.dst' columns
    df.drop(columns=['ip.src', 'ip.dst'], inplace=True)
    
    return df

# ADDS INFO FROM THE ASN OF THE IP USING IPWHOIS
def add_asn_info(df):
    """
    Updates a Pandas DataFrame by adding 'asn_country', 'asn_description', and 'network_name' columns
    for rows where 'ip.proto' equals 6, based on the 'ip' column. Uses a lookup table
    to optimize repeated lookups.

    Args:
        df (pd.DataFrame): The input DataFrame containing 'ip' and 'ip.proto' columns.

    Returns:
        pd.DataFrame: The updated DataFrame with 'asn_country', 'asn_description', and 'network_name' columns.
    """
    # Create a lookup table to cache ASN information
    asn_info_lookup = {}

    # Initialize new columns with default values
    df["asn_country"] = "Unknown"
    df["asn_description"] = "Unknown"
    df["network_name"] = "Unknown"

    # Filter rows where 'ip.proto' equals 6
    for index, row in df.iterrows():
        if row.get("ip.proto") == 6:
            ip = row["ip"]
            if ip not in asn_info_lookup:
                try:
                    # Query ASN information
                    obj = IPWhois(ip)
                    res = obj.lookup_rdap()
                    country = res.get("asn_country_code", "Unknown")
                    asn_description = res.get("asn_description", "Unknown")
                    network_name = res.get("network", {}).get("name", "Unknown")

                    # Process ASN description: split by ',' and then by '-'
                    if asn_description and "," in asn_description:
                        asn_description = asn_description.split(",")[0]
                    if asn_description and "-" in asn_description:
                        asn_description = asn_description.split("-")[0]

                    # Process Network name: split by '-'
                    if network_name and "-" in network_name:
                        network_name = network_name.split("-")[0]
                except Exception as e:
                    country = "Unknown"
                    asn_description = "Unknown"
                    network_name = "Unknown"  # Handle potential lookup errors

                # Cache the results
                asn_info_lookup[ip] = (country, asn_description, network_name)
            else:
                # Use cached result
                country, asn_description, network_name = asn_info_lookup[ip]

            # Update the respective columns in the DataFrame
            df.at[index, "asn_country"] = str(country)
            df.at[index, "asn_description"] = str(asn_description)
            df.at[index, "network_name"] = str(network_name)

    return df
