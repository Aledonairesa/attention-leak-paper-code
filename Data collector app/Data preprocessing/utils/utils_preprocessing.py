import pandas as pd
from io import StringIO
import re
import numpy as np
import pytz
import csv

# ==========================================================================
# FRAMES
# ==========================================================================


def read_frames_csv(frames_path: str) -> pd.DataFrame:

    EXPECTED_COLUMN_NAMES = [
    'frame.number','frame.time','ip.src','ip.dst','ip.proto',
    'tcp.srcport','tcp.dstport','udp.srcport','udp.dstport',
    'tcp.flags.str','frame.len','dns.qry.name','dns.a'
    ]
    
    raw_lines_list = [] # To store lines read from file (list of strings)
    
    # 1. Read file and prepare lines (applying the "skip last line" logic)
    try:
        with open(frames_path, "r", encoding="utf-8", errors="replace") as f:
            raw_lines_list = f.readlines()
        
        if not raw_lines_list: # File is empty
            # print(f"Info: File '{frames_path}' is empty.")
            return pd.DataFrame(columns=EXPECTED_COLUMN_NAMES)

        # Determine lines to be parsed: skip the last line if file has more than one line.
        # This retains the original script's behavior.
        if len(raw_lines_list) > 1:
            lines_for_parser_list = raw_lines_list[:-1]
        else: # File has 0 or 1 line, parse what's there (no "last line" to skip).
            lines_for_parser_list = raw_lines_list
        
        if not lines_for_parser_list: 
            # This could happen if the file had 0 lines initially.
            # print(f"Info: No lines to parse from '{frames_path}' after initial processing.")
            return pd.DataFrame(columns=EXPECTED_COLUMN_NAMES)

        # For pandas, join lines into a single string. For csv.reader, use the list.
        csv_string_for_pandas = "".join(lines_for_parser_list)
        if not csv_string_for_pandas.strip(): # All lines were whitespace or empty
            # print(f"Info: Content of '{frames_path}' is empty or whitespace after processing.")
            return pd.DataFrame(columns=EXPECTED_COLUMN_NAMES)

    except FileNotFoundError:
        print(f"Error: File not found at '{frames_path}'.")
        return pd.DataFrame(columns=EXPECTED_COLUMN_NAMES)
    except Exception as e:
        print(f"Error during file reading or initial line processing for '{frames_path}': {e}")
        return pd.DataFrame(columns=EXPECTED_COLUMN_NAMES)

    # 2. Attempt parsing with pandas 'python' engine
    try:
        frames_df = pd.read_csv(
            StringIO(csv_string_for_pandas),
            quotechar='"',          # Explicitly define the quote character
            engine='python',        # Use the more flexible Python parsing engine
            skipinitialspace=True   # Handles cases like "field1, field2"
        )
        # Convert 'frame.number' to numeric if it exists
        if 'frame.number' in frames_df.columns:
            frames_df['frame.number'] = pd.to_numeric(frames_df['frame.number'], errors='coerce')
        # print(f"Info: Successfully parsed '{frames_path}' using pandas 'python' engine.")
        return frames_df
    except pd.errors.EmptyDataError: 
        # This can occur if csv_string_for_pandas was effectively empty (e.g., only a header line that read_csv might treat as data if no other lines)
        # print(f"Info: Pandas 'python' engine reported EmptyDataError for '{frames_path}'. Trying to extract header.")
        if lines_for_parser_list and lines_for_parser_list[0].strip(): # Check if there was any content (e.g. a header)
            try:
                header_line_content = StringIO(lines_for_parser_list[0])
                # Read just the header, no data rows
                header_df = pd.read_csv(header_line_content, nrows=0, engine='python', quotechar='"', skipinitialspace=True)
                return header_df # Returns empty DataFrame with columns from header
            except Exception:
                # print(f"Warning: Could not extract header from possibly header-only file '{frames_path}'.")
                pass # Fall through to return default empty DataFrame
        return pd.DataFrame(columns=EXPECTED_COLUMN_NAMES)
    except pd.errors.ParserError as e_pandas:
        print(f"Warning: Pandas 'python' engine failed for '{frames_path}': {e_pandas}. Falling back to csv.reader.")
    except Exception as e_general_pandas: # Catch other potential pandas errors
        print(f"Warning: An unexpected error occurred with pandas 'python' engine for '{frames_path}': {e_general_pandas}. Falling back to csv.reader.")

    # 3. Fallback: Use Python's 'csv' module for robust line-by-line parsing
    # print(f"Info: Attempting fallback parsing for '{frames_path}' with csv.reader.")
    parsed_header = []
    data_rows = []
    try:
        # Use the list of strings `lines_for_parser_list` directly
        csv_module_reader = csv.reader(lines_for_parser_list, quotechar='"', skipinitialspace=True)
        
        parsed_header = next(csv_module_reader) # First line from the list is header
        num_header_cols = len(parsed_header)

        for i, row_fields in enumerate(csv_module_reader): # Process remaining lines as data
            if len(row_fields) == num_header_cols:
                data_rows.append(row_fields)
            else:
                # Handle rows with an unexpected number of fields by adjusting them
                # print(f"Warning: Line {i+2} in '{frames_path}' (data row {i+1}) has {len(row_fields)} fields, expected {num_header_cols}. Adjusting row.")
                adjusted_row = row_fields[:num_header_cols] # Truncate if too long
                adjusted_row.extend([''] * (num_header_cols - len(adjusted_row))) # Pad with empty strings if too short
                data_rows.append(adjusted_row)
        
        if not parsed_header and not data_rows: # Should ideally not happen if initial checks passed
            # print(f"Info: CSV fallback parsing yielded no header or data for '{frames_path}'.")
            return pd.DataFrame(columns=EXPECTED_COLUMN_NAMES)

        frames_df = pd.DataFrame(data_rows, columns=parsed_header if parsed_header else None)
        
        if 'frame.number' in frames_df.columns: # Ensure 'frame.number' is numeric
            frames_df['frame.number'] = pd.to_numeric(frames_df['frame.number'], errors='coerce')
        # print(f"Info: Fallback parsing with csv.reader for '{frames_path}' completed.")
        return frames_df

    except StopIteration: # No lines for csv_module_reader (e.g., lines_for_parser_list was empty or only one line which was header)
        if parsed_header: # Only header was successfully read by csv.reader
            # print(f"Info: CSV fallback for '{frames_path}' found only a header.")
            return pd.DataFrame(columns=parsed_header)
        # print(f"Info: CSV fallback for '{frames_path}' found no data (StopIteration).")
        return pd.DataFrame(columns=EXPECTED_COLUMN_NAMES)
    except Exception as e_csv_fallback:
        print(f"Error: Fallback parsing with csv.reader for '{frames_path}' also failed: {e_csv_fallback}")
        # Return empty DF with best-effort columns (parsed_header if available, else default)
        final_columns = parsed_header if parsed_header else EXPECTED_COLUMN_NAMES
        return pd.DataFrame(columns=final_columns)

def process_frame_time_column(frames_df):
    # Extract the datetime text and the full timezone name in two columns
    frames_df['frame.time'] = frames_df['frame.time'].astype(str)
    parts = frames_df['frame.time'].str.extract(
        r'^(.*?)(?:\s+)([A-Za-zÀ-ÿ\s]+)$', expand=True
    )
    parts.columns = ['datetime_str', 'tz_abbr']
    parts['tz_key'] = parts['tz_abbr'].str.strip().str.upper()

    # Map each tz_key to its UTC offset in seconds
    offset_map = {
        'CEST': 2 * 3600,
        'HORA DE VERANO ROMANCE': 2 * 3600,
        'HORA ESTÁNDAR ROMANCE': 1 * 3600,
        'ROMANCE STANDARD TIME': 1 * 3600
    }
    parts['offset_s'] = parts['tz_key'].map(offset_map)

    # Parse all datetimes at once
    dt = pd.to_datetime(parts['datetime_str'], format=None, exact=False)

    # Convert to POSIX seconds, then subtract the offset
    unix_secs = dt.astype('int64') / 1e9 - parts['offset_s']

    # Return a new DataFrame (or assign back)
    out = frames_df.copy()
    out['frame.time'] = unix_secs

    return out

def infer_host_ip(df):
    all_ips = pd.concat([df['ip.src'], df['ip.dst']]).dropna().astype(str)
    values, counts = np.unique(all_ips.values, return_counts=True)
    return values[np.argmax(counts)]

def add_send_column(df, reference_column, condition_value, new_column_name):
    df[new_column_name] = (df[reference_column] == condition_value).astype(int)
    return df

def create_merged_ips_column(df, start_str):
    # Create masks for values to ignore (i.e., those starting with start_str)
    src_mask = df['ip.src'].astype(str).str.startswith(start_str)
    dst_mask = df['ip.dst'].astype(str).str.startswith(start_str)

    # Replace those values with NaN
    src_clean = df['ip.src'].where(~src_mask)
    dst_clean = df['ip.dst'].where(~dst_mask)

    # Merge columns, preferring non-NaN from 'ip.src'
    df['ip'] = src_clean.combine_first(dst_clean)

    return df

def delete_rows_with_nan_ips(dataframe):
    return dataframe.dropna(subset=['ip.src', 'ip.dst'])

def delete_multicast(df):
    """Removes rows where 'ip.src' or 'ip.dst' is in 224.0.0.0/24 local multicast range."""
    # Build mask using vectorized string matching
    src_multicast = df['ip.src'].astype(str).str.startswith('224.0.0.')
    dst_multicast = df['ip.dst'].astype(str).str.startswith('224.0.0.')

    # Keep rows that are NOT local multicast in src or dst
    return df[~(src_multicast | dst_multicast)]

def merge_ports(df):
    """
    Merges TCP and UDP source/destination port columns into 'port.src' and 'port.dst'.
    """
    port_cols = {
        'port.src': ('tcp.srcport', 'udp.srcport'),
        'port.dst': ('tcp.dstport', 'udp.dstport')
    }
    
    if not all(col in df.columns for cols in port_cols.values() for col in cols):
        missing = [col for cols in port_cols.values() for col in cols if col not in df.columns]
        print(f"Missing columns: {', '.join(missing)}. The function requires all these columns.")
        return df

    df = df.copy()

    for out_col, (tcp_col, udp_col) in port_cols.items():
        df[out_col] = df[tcp_col].combine_first(df[udp_col]).astype(str)

    df.drop(columns=[col for cols in port_cols.values() for col in cols], inplace=True)
    
    return df

def infer_frames_tz(frames_df):
    """
    Inspect the original 'frame.time' strings and return a suitable
    pytz zone name ('Europe/Madrid' for CET/CEST, or fallback 'UTC').
    """
    # Extract the textual suffix
    frames_df['frame.time'] = frames_df['frame.time'].astype(str)
    tz_abbr = frames_df['frame.time'].str.extract(
        r'^\s*.*?\s+([A-Za-zÀ-ÿ\s]+)$', expand=False
    ).str.strip().str.upper()

    if tz_abbr.isin({'CEST', 'HORA DE VERANO ROMANCE'}).any():
        return 'Europe/Madrid'            # summer (UTC+2 / UTC+1 in winter)
    if tz_abbr.isin({'HORA ESTÁNDAR ROMANCE', 'ROMANCE STANDARD TIME'}).any():
        return 'Europe/Madrid'            # winter label
    return 'UTC'                          # safe default

# === Frames preprocessing pipeline function ===
def read_and_preprocess_frames(frames_path):
    # Read the input CSV file and load it into a DataFrame
    frames_df = read_frames_csv(frames_path)
    print(f"- Loaded {len(frames_df)} rows from CSV.")

    # Convert the 'frame.time' column to Unix timestamps (UTC)
    frames_df = process_frame_time_column(frames_df)
    print("- 'frame.time' column converted to Unix timestamps.")

    # Infer the host IP by identifying the most frequent IP address
    inferred_host_ip = infer_host_ip(frames_df)
    print(f"- Inferred host IP: {inferred_host_ip}")

    # Add a 'send' column: 1 if the row corresponds to an outgoing packet, 0 otherwise
    frames_df = add_send_column(frames_df, "ip.src", inferred_host_ip, "send")
    print("- 'send' column added based on host IP.")

    # Create a unified 'ip' column with the relevant external IP
    frames_df = create_merged_ips_column(frames_df, inferred_host_ip)
    print("- 'ip' column created by merging source and destination IPs.")

    # Remove rows with missing source or destination IP addresses
    before_drop = len(frames_df)
    frames_df = delete_rows_with_nan_ips(frames_df)
    print(f"- Removed {before_drop - len(frames_df)} rows with missing IPs.")

    # Remove local multicast traffic (224.0.0.0/24)
    before_filter = len(frames_df)
    frames_df = delete_multicast(frames_df)
    print(f"- Removed {before_filter - len(frames_df)} local multicast rows.")

    # Merge TCP/UDP port columns into unified 'port.src' and 'port.dst' columns
    frames_df = merge_ports(frames_df)
    print("- Merged TCP/UDP port columns into 'port.src' and 'port.dst'.")

    return frames_df


# ==========================================================================
# TIMESTAMPS
# ==========================================================================

def read_timestamps_csv(timestamps_path):
    pattern = re.compile(r'\[(.*?)\].*?Window Focus Changed:\s*(.*)')
    invisible_chars = re.compile(r'[\u200b\u200c\u200d\uFEFF]')
    data = []

    with open(timestamps_path, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                time = match.group(1)
                message = invisible_chars.sub('', match.group(2))  # Remove invisible characters
                data.append({'time': time, 'message': message})

    return pd.DataFrame(data)

def delete_empty_rows(df):
    return df[df['message']!=''].reset_index(drop=True)

def convert_timestamp_time_to_unix(df, column_name="time", timezone="CET"):
    tz = pytz.timezone(timezone)
    
    # Fix the time format by replacing the last colon with a dot
    df = df.copy()
    df[column_name] = df[column_name].str.replace(r':(\d{1,6})$', r'.\1', regex=True)
    
    # Parse with pd.to_datetime (flexible)
    dt = pd.to_datetime(df[column_name], errors='coerce')
    
    # Localize and convert to UTC
    dt_local = dt.dt.tz_localize(tz, ambiguous='NaT', nonexistent='NaT')
    df[column_name] = dt_local.dt.tz_convert("UTC").astype('int64') / 1e9
    
    return df

def remove_spurious_tasks(dataframe, message_column):
    substrings_to_remove = [
        "Programa.exe", "Programa de recollida de dades", "Conmutación de tareas",
        "Trace Tracker App", "Asistente de acoplamiento", "Auxiliar d'acoblament",
        "UnlockingWindow", "Buscar", "Búsqueda", "Cerca", "Notificación nueva",
        "ancho medio", "Column Width", "Save As", "Guardar como",
        "Lista de accesos directos", "Sin título y"
    ]

    pattern = '|'.join(re.escape(s) for s in substrings_to_remove)
    filtered_df = dataframe[~dataframe[message_column].str.contains(pattern, case=False, na=False)]

    return filtered_df.reset_index(drop=True)

def remove_consecutive_small_differences(df, column, threshold=0.2):
    """
    Removes consecutive rows where the absolute difference in a numerical column
    is smaller than the specified threshold.
    """
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"Column '{column}' must be numeric.")

    diffs = df[column].diff().abs()
    diffs.iloc[0] = np.inf  # Always keep the first row

    return df[diffs >= threshold].reset_index(drop=True)

def remove_consecutive_duplicates(df, column="message"):
    """
    Removes consecutive duplicate values in the specified column of a DataFrame.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")
    
    mask = df[column] != df[column].shift()
    return df[mask].reset_index(drop=True)

# === Timestamps preprocessing pipeline function ===
def read_and_preprocess_timestamps(timestamps_path, timezone="CET", time_diff_threshold=0.2):
    # Read the raw timestamp file and extract time/message pairs
    df = read_timestamps_csv(timestamps_path)
    print(f"- Loaded {len(df)} timestamp rows from file.")

    # Remove rows with empty message strings
    df = delete_empty_rows(df)
    print(f"- Removed empty message rows. Remaining: {len(df)} rows.")

    # Convert 'time' column to Unix timestamps in UTC
    df = convert_timestamp_time_to_unix(df, column_name="time", timezone=timezone)
    print("- 'time' column converted to Unix timestamps (UTC).")

    # Remove messages known to be spurious
    df = remove_spurious_tasks(df, message_column="message")
    print(f"- Removed spurious task rows. Remaining: {len(df)} rows.")

    # Remove consecutive rows with very small time differences
    df = remove_consecutive_small_differences(df, column="time", threshold=time_diff_threshold)
    print(f"- Removed consecutive rows with < {time_diff_threshold}s time difference. Remaining: {len(df)} rows.")

    # Remove consecutive rows with identical messages
    df = remove_consecutive_duplicates(df, column="message")
    print(f"- Removed consecutive duplicate messages. Final row count: {len(df)}")

    return df