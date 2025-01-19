import pandas as pd
import numpy as np


def is_bool(col):
    valid_vals = set(col.dropna().unique())
    valid_bool_vals = {True, False, 'true', 'false', "true ", "false ", "True ", "False ", 0, 1}
    return valid_vals.issubset(valid_bool_vals)


def convert_to_bool(col):
    return col.apply(lambda x: True if x in ['true', 1, True] else (False if x in ['false', 0, False] else x))


def is_num(col):
    try:
        pd.to_numeric(col.dropna())
        return True
    except ValueError:
        return False


def is_time(col):
    try:

        pd.to_datetime(col, errors='raise', format='%Y-%m-%d')  # Forcer un format spécifique de date
        return True
    except Exception:
        return False


def get_types_col(df):


    df = df.replace(
        to_replace=r'^\s*(true|True)\s*$', 
        value=True, 
        regex=True
    ).replace(
        to_replace=r'^\s*(false|False)\s*$', 
        value=False, 
        regex=True
    )
    df.drop("C1", axis=1, inplace=True)
    bools = [col for col in df.columns if is_bool(df[col])]
    nums = [col for col in df.columns if is_num(df[col]) and col not in bools]
    times = [col for col in df.columns if is_time(df[col])]

    quals = list(set(df.columns) - set(nums) - set(bools) - set(times))

    for col in bools:
        df[col] = convert_to_bool(df[col])

    for col in times:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    for col in quals:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) == 2:

            df[col] = df[col].map({unique_vals[0]: True, unique_vals[1]: False})
            bools.append(col)
            quals.remove(col)

    nums.remove("COD_INSEE")
    nums.remove("COD_IRIS")
    bools.remove("TARGET")

    df["COD_INSEE"] = df["COD_INSEE"].fillna(-1).astype(int).astype(str).replace("-1", float("nan"))
    df["COD_IRIS"] = df["COD_IRIS"].fillna(-1).astype(int).astype(str).replace("-1", float("nan"))

    df.replace({col: {'nan': np.nan} for col in df.columns}, inplace=True)

    return df, (bools, quals, nums, times)
