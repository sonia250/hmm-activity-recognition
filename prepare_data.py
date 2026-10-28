"""
prepare_data.py
Turns Sensor Logger zip exports into clean, labelled CSV files.
Long recordings are trimmed and then cut into short chunks (files).

Input : raw_zips/<activity>_<split>_<N>.zip
        activity = standing | walking | jumping | still
        split    = train | test
        e.g. walking_train_1.zip, walking_train_2.zip, walking_test_1.zip
Output: data/train/<activity>_train_<N>_<chunk>.csv
        data/test/<activity>_test_<N>_<chunk>.csv

Each output file has: time_s, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
(time_s starts at 0, uniform 50 Hz grid)

Run:  python prepare_data.py
"""
import glob
import io
import os
import re
import zipfile

import numpy as np
import pandas as pd

RAW_DIR = "raw_zips"
OUT_DIR = "data"
TARGET_HZ = 50          # common sampling rate for every recording
TRIM_SECONDS = 1.0      # remove first/last second (getting into position)
CHUNK_SECONDS = 8.0     # length of each output file (rubric: 5-10 s)
MIN_CHUNK_SECONDS = 5.0 # drop a leftover piece shorter than this


def read_sensor(zf, name, prefix):
    """Read one sensor CSV from the zip and rename x/y/z with a prefix."""
    with zf.open(name) as f:
        df = pd.read_csv(io.BytesIO(f.read()))
    df = df[["seconds_elapsed", "x", "y", "z"]].copy()
    df.columns = ["time_s", f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]
    return df


def process_zip(path):
    """Merge accelerometer + gyroscope on a uniform grid and trim the ends."""
    with zipfile.ZipFile(path) as zf:
        acc = read_sensor(zf, "Accelerometer.csv", "acc")
        gyr = read_sensor(zf, "Gyroscope.csv", "gyro")

    start = max(acc["time_s"].iloc[0], gyr["time_s"].iloc[0]) + TRIM_SECONDS
    end = min(acc["time_s"].iloc[-1], gyr["time_s"].iloc[-1]) - TRIM_SECONDS
    grid = np.arange(start, end, 1.0 / TARGET_HZ)

    out = pd.DataFrame({"time_s": grid - grid[0]})
    for df in (acc, gyr):
        for col in df.columns[1:]:
            out[col] = np.interp(grid, df["time_s"].values, df[col].values)
    return out


def split_chunks(df):
    """Cut a recording into consecutive chunks of CHUNK_SECONDS."""
    n = int(CHUNK_SECONDS * TARGET_HZ)
    min_n = int(MIN_CHUNK_SECONDS * TARGET_HZ)
    chunks = []
    for i in range(0, len(df), n):
        piece = df.iloc[i:i + n].copy()
        if len(piece) < min_n:
            break
        piece["time_s"] = piece["time_s"] - piece["time_s"].iloc[0]
        chunks.append(piece)
    return chunks


def main():
    paths = sorted(glob.glob(os.path.join(RAW_DIR, "*.zip")))
    if not paths:
        print(f"No zip files found in '{RAW_DIR}/'")
        return

    for sub in ("train", "test"):
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

    rows = []
    for path in paths:
        base = os.path.splitext(os.path.basename(path))[0]
        m = re.match(r"^(standing|walking|jumping|still)_(train|test)_(\d+)$", base)
        if not m:
            print(f"SKIPPED (bad name, use e.g. walking_train_1.zip): {base}.zip")
            continue
        activity, split, num = m.group(1), m.group(2), int(m.group(3))
        chunks = split_chunks(process_zip(path))
        for k, piece in enumerate(chunks, start=1):
            name = f"{activity}_{split}_{num}_{k:02d}.csv"
            piece.to_csv(os.path.join(OUT_DIR, split, name), index=False)
            rows.append((activity, split, len(piece) / TARGET_HZ))
        print(f"{base:22s} -> {len(chunks)} file(s)")

    s = pd.DataFrame(rows, columns=["activity", "split", "seconds"])
    print("\nNumber of files per activity and split:")
    print(s.pivot_table(index="activity", columns="split",
                        values="seconds", aggfunc="count").fillna(0).astype(int))
    print("\nTotal clean seconds per activity and split:")
    print(s.pivot_table(index="activity", columns="split",
                        values="seconds", aggfunc="sum").round(1))


if __name__ == "__main__":
    main()
