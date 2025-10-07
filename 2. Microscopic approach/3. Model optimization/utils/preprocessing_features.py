def normalize_columns(df, columns):
    """
    Normalizes specified columns of a dataframe using max absolute scaling
    with a predifined maximum found empirically.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    columns (list): List of column names to normalize.

    Returns:
    pd.DataFrame: A dataframe with normalized columns.
    """
    # Create a copy of the dataframe to avoid modifying the original one
    df_normalized = df.copy()
    
    # Predefined maximum absolute value found empirically
    max_abs_value = 10000
    max_abs_value_var = 10**7
    
    # Normalize each specified column using Max Absolute Scaling
    for col in columns:
        if col in df_normalized.columns:
            if col=="lenframe_last_variance_50":
                df_normalized[col] = df_normalized[col] / max_abs_value_var
            else:
                df_normalized[col] = df_normalized[col] / max_abs_value
    
    return df_normalized