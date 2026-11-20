import pandas as pd
import numpy as np
from pathlib import Path
from pypdf import PdfWriter

master = Path("D:/nightglow_files/master_cnn.csv")
df = pd.read_csv(master)

#Quality filtering
df = df[
    (df["pearson_r_muv"] >= 0.55)
    & (~df["type"].isin(["NONE", "NON-DET"]))
    & (df["color_flag_muv"] == "green")
    & (df["count"] >= 6)
]

# SEP event filtering
strong_uvd_min = 0.18
strong_alt_max = 112

support_uvd_min = 0.10
support_alt_max = 130

alt_min = 50
uvd_snr_min = 3

df = df.copy()

quality_mask = (
    (df["altitude"] >= alt_min)
    & (df["CO2p_snr"] >= uvd_snr_min)
)

df["sep_strong"] = (
    quality_mask
    & (df["CO2p_radiance"] >= strong_uvd_min)
    & (df["altitude"] < strong_alt_max)
)

df["sep_support"] = (
    quality_mask
    & (df["CO2p_radiance"] >= support_uvd_min)
    & (df["altitude"] < support_alt_max)
)

# Count distinct scans meeting either level.
qualifying_counts = (
    df.loc[df["sep_strong"] | df["sep_support"]]
    .groupby("orbit_number")["scan"]
    .nunique()
)

# Route 1: one strong scan plus a second qualifying scan.
strong_orbits = qualifying_counts[
    qualifying_counts >= 2
].index.intersection(
    df.loc[df["sep_strong"], "orbit_number"].unique()
)

# Route 2: at least three supporting scans.
low_support_counts = (
    df.loc[df["sep_support"]]
    .groupby("orbit_number")["scan"]
    .nunique()
)

support_orbits = low_support_counts[
    low_support_counts >= 3
].index

sep_orbits = strong_orbits.union(support_orbits)

df["sep_orbit_match"] = df["orbit_number"].isin(sep_orbits)
df["is_sep"] = df["sep_orbit_match"]
df = df.loc[df["is_sep"]].copy()

df.to_csv("D:/nightglow_files/sep_test.csv", index=False)

# Merge quicklooks for the unique orbits surviving both filters.
pdf_root = Path("D:/nightglow_files/quicklooks_out_cnn")
output_pdf = Path("D:/nightglow_files/sep_test_ql.pdf")
orbits = sorted(df["orbit_number"].dropna().astype(int).unique())
pdf_files = []
missing_orbits = []

for orbit in orbits:
    # For example, orbit 8001 lives in 08000/orbit08001.pdf.
    block_start = (orbit // 100) * 100
    pdf_file = pdf_root / f"{block_start:05d}" / f"orbit{orbit:05d}.pdf"
    if pdf_file.is_file():
        pdf_files.append(pdf_file)
    else:
        missing_orbits.append(orbit)
        print(f"Missing PDF: {pdf_file}")

if pdf_files:
    writer = PdfWriter()
    try:
        for pdf_file in pdf_files:
            writer.append(str(pdf_file))
        with output_pdf.open("wb") as file:
            writer.write(file)
    finally:
        writer.close()
    print(f"Merged {len(pdf_files)} orbit PDFs into {output_pdf}")
else:
    print("No matching orbit PDFs found; no merged PDF written.")

if missing_orbits:
    print(f"Missing PDFs for {len(missing_orbits)} of {len(orbits)} orbits.")
