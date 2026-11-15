"""Chronological splits with strict date boundaries."""

import pandas as pd


def chronological_split(
    frame: pd.DataFrame,
    date_column: str = "_date",
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train + validation fractions must be less than 1.")
    if date_column not in frame:
        raise ValueError(f"Missing date column: {date_column}")
    out = frame.copy()
    out[date_column] = pd.to_datetime(out[date_column], errors="coerce")
    out = out.loc[out[date_column].notna()].sort_values(date_column, kind="stable")
    unique_dates = pd.Index(out[date_column].drop_duplicates().sort_values())
    if len(unique_dates) < 3:
        raise ValueError("At least three distinct match dates are required.")
    train_end = max(1, min(len(unique_dates) - 2, int(len(unique_dates) * train_fraction)))
    val_end = max(train_end + 1, min(len(unique_dates) - 1, int(len(unique_dates) * (train_fraction + validation_fraction))))
    train_cut = unique_dates[train_end - 1]
    val_cut = unique_dates[val_end - 1]
    train = out.loc[out[date_column] <= train_cut].copy()
    validation = out.loc[(out[date_column] > train_cut) & (out[date_column] <= val_cut)].copy()
    test = out.loc[out[date_column] > val_cut].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Chronological split produced an empty partition.")
    return train, validation, test
