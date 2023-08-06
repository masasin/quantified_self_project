from __future__ import annotations

import enum
import re
import string
from pathlib import Path
from typing import Iterable

import pandas as pd
from pydantic import BaseModel

data_root = Path("data")
data_raw = data_root / "raw/MetaMotion"
data_interim = data_root / "interim"

# %% Read data from files
Label = enum.StrEnum("Label", "bench dead ohp row squat rest")
Category = enum.StrEnum("Category", "heavy medium light sitting standing")
Sensor = enum.StrEnum("Sensor", "accelerometer gyroscope")


def number_part(s: str) -> str:
    return re.search(r"[\d.]+", s).group()


class ExerciseDetails(BaseModel):
    participant: str
    label: Label
    category: Category
    set_number: int
    sensor: Sensor
    frequency: float

    @classmethod
    def from_filename(cls, filename: str) -> ExerciseDetails:
        set_data, _, date, _, device, frequency, _ = filename.split("_")

        match set_data.split("-"):
            case [participant, label, category, *_]:
                category, set_number = cls._extract_category_and_set_number(category)
            case [participant, label, category]:
                category, set_number = cls._extract_category_and_set_number(category)
            case _:
                raise ValueError(f"Invalid filename: {filename}")
        return cls(
            participant=participant,
            label=Label[label],
            category=Category[category],
            set_number=set_number,
            sensor=Sensor[device.lower()],
            frequency=float(number_part(frequency)),
        )

    @staticmethod
    def _extract_category_and_set_number(category):
        try:
            set_number = int(number_part(category))
        except AttributeError:
            set_number = 1
        category = category.rstrip(string.digits)
        return category, set_number

    def to_pandas(self) -> dict:
        return {
            "participant": self.participant,
            "label": self.label.value,
            "category": self.category.value,
            "set_number": self.set_number,
        }


def read_data_from_files(files: Iterable[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_acc = pd.DataFrame()
    df_gyro = pd.DataFrame()
    id_acc = id_gyro = 0

    for file in files:
        if "Accelerometer" in file.stem:
            id_acc += 1
            df_acc = pd.concat(
                [
                    df_acc,
                    pd.read_csv(file).assign(
                        **ExerciseDetails.from_filename(file.stem).to_pandas(),
                        set_id=id_acc,
                    ),
                ],
            )
        elif "Gyroscope" in file.stem:
            id_gyro += 1
            df_gyro = pd.concat(
                [
                    df_gyro,
                    pd.read_csv(file).assign(
                        **ExerciseDetails.from_filename(file.stem).to_pandas(),
                        set_id=id_gyro,
                    ),
                ]
            )
        else:
            raise ValueError(f"Invalid filename: {file.stem}")

    df_acc.rename(
        columns={
            "epoch (ms)": "time",
            "x-axis (g)": "ax",
            "y-axis (g)": "ay",
            "z-axis (g)": "az",
        },
        inplace=True,
    )

    df_gyro.rename(
        columns={
            "epoch (ms)": "time",
            "x-axis (deg/s)": "wx",
            "y-axis (deg/s)": "wy",
            "z-axis (deg/s)": "wz",
        },
        inplace=True,
    )

    df_acc.index = pd.to_datetime(df_acc.time, unit="ms")
    df_gyro.index = pd.to_datetime(df_gyro.time, unit="ms")
    df_acc.drop(columns=["time", "time (01:00)", "elapsed (s)"], inplace=True)
    df_gyro.drop(columns=["time", "time (01:00)", "elapsed (s)"], inplace=True)

    df_acc.participant = df_acc.participant.astype("category")
    df_acc.label = df_acc.label.astype("category")
    df_acc.category = df_acc.category.astype("category")
    df_gyro.participant = df_gyro.participant.astype("category")
    df_gyro.label = df_gyro.label.astype("category")
    df_gyro.category = df_gyro.category.astype("category")

    return df_acc, df_gyro


# %% Merging datasets
def merge_datasets(df_acc: pd.DataFrame, df_gyro: pd.DataFrame) -> pd.DataFrame:
    df_merged = pd.merge(
        df_acc,
        df_gyro,
        how="outer",
        left_index=True,
        right_index=True,
        suffixes=("_acc", "_gyro"),
    )

    for col in [col for col in df_merged.columns if col.endswith("_acc")]:
        df_merged[col] = df_merged[col].combine_first(
            df_merged[col.replace("_acc", "_gyro")]
        )
        df_merged.drop(columns=[col.replace("_acc", "_gyro")], inplace=True)
        df_merged.rename(columns={col: col.replace("_acc", "")}, inplace=True)

    df_merged = df_merged[
        [
            "ax",
            "ay",
            "az",
            "wx",
            "wy",
            "wz",
            "set_id",
            "participant",
            "label",
            "category",
            "set_number",
        ]
    ]
    df_merged.set_id = df_merged.set_id.astype("int")
    df_merged.set_number = df_merged.set_number.astype("int")

    return df_merged


# %% Resample data (frequency conversion)
# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz
def resample_data(df_merged: pd.DataFrame, period="200ms") -> pd.DataFrame:
    sampling = {
        "ax": "mean",
        "ay": "mean",
        "az": "mean",
        "wx": "mean",
        "wy": "mean",
        "wz": "mean",
        "set_id": "last",
        "participant": "last",
        "label": "last",
        "category": "last",
        "set_number": "last",
    }
    df_by_day = [df for _, df in df_merged.groupby(pd.Grouper(freq="D"))]
    df_resampled = pd.concat(
        df.resample(period).apply(sampling).dropna() for df in df_by_day
    )
    return df_resampled


# %% Main
if __name__ == "__main__":
    df_acc, df_gyro = read_data_from_files(data_raw.glob("*.csv"))
    df_merged = merge_datasets(df_acc, df_gyro)
    df_resampled = resample_data(df_merged)
    df_resampled.to_pickle(data_interim / "01_data_processed.pkl")
