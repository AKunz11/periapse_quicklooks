from pathlib import Path
from pypdf import PdfWriter
import pandas as pd
import re
import numpy as np

orb_start = 8000
orb_end = 9000

# Load merged comparison CSV
df = pd.read_csv(
    f"D:/ml_project/model_compare/{orb_start:05d}_{orb_end:05d}_merged.csv"
)

# Keep only CNN-only detections that are green or yellow
df_new = df[
    (df["method_match"] == "cnn_only") | (df["method_match"] == "both")
    & (df["color_flag_muv_cnn"].isin(["green", "yellow"]))
].copy()

df_old = df[
    (df["method_match"] == "old_only") | (df["method_match"] == "both")
    & (df["color_flag_muv_old"].isin(["green", "yellow"]))
].copy()

# Unique CNN-only orbit numbers
unique_orbits = set(
    df_old["orbit_number_old"]
    .dropna()
    .astype(int)
    .unique()
)

print(f"CNN-only rows: {len(df_new)}")
print(f"CNN-only unique orbits: {len(unique_orbits)}")
print("Orbits:", sorted(unique_orbits))
print("=" * 80)


def get_orbit_number(pdf_path):
    """Extract a standalone five-digit orbit number from the filename."""
    match = re.search(r"(?<!\d)(\d{5})(?!\d)", pdf_path.stem)
    return int(match.group(1)) if match else None


# Keep track ONLY of block PDFs created during this run
created_block_pdfs = []

for num in np.arange(orb_start, orb_end, 100):

    orb_block = int(num)

    input_folder = Path(
        f"D:/nightglow/quicklooks_out/{orb_block:05d}"
    )

    output_pdf = (
        input_folder
        / f"combined_quicklooks_{orb_block:05d}.pdf"
    )

    # Find orbit PDFs belonging to CNN-only detections
    pdf_files = sorted(
        pdf
        for pdf in input_folder.glob("*.pdf")
        if pdf.name != output_pdf.name
        and get_orbit_number(pdf) in unique_orbits
    )

    if not pdf_files:
        print(
            f"No CNN-only PDFs found in block "
            f"{orb_block:05d}"
        )
        continue

    writer = PdfWriter()

    for pdf_file in pdf_files:
        print(f"Adding: {pdf_file.name}")
        writer.append(str(pdf_file))

    with output_pdf.open("wb") as file:
        writer.write(file)

    writer.close()

    # IMPORTANT:
    # Only add PDFs actually created during THIS run
    created_block_pdfs.append(output_pdf)

    print(
        f"\nCombined {len(pdf_files)} PDFs into:"
    )
    print(output_pdf)
    print("=" * 80 + "\n")


# ============================================================
# Final merge
# ============================================================

base_folder = Path(
    "D:/nightglow/quicklooks_out"
)

final_output_pdf = (
    base_folder
    / f"combined_quicklooks_"
      f"{orb_start:05d}_{orb_end:05d}_old.pdf"
)

if not created_block_pdfs:

    print(
        "No block PDFs were created. "
        "Skipping final merge."
    )

else:

    writer = PdfWriter()

    for pdf_file in created_block_pdfs:
        print(f"Adding block: {pdf_file}")
        writer.append(str(pdf_file))

    with final_output_pdf.open("wb") as file:
        writer.write(file)

    writer.close()

    print(
        f"\nCombined {len(created_block_pdfs)} "
        f"block PDFs into:"
    )
    print(final_output_pdf)
    print("=" * 80)