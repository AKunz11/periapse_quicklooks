from pathlib import Path
from pypdf import PdfWriter
import pandas as pd
import re
import numpy as np

orb_start = 15200
orb_end = 15300
method = "new" #old or new
unique = False #only in the new/old method, NO overlap
reject = False #reject by eye stuff

# By eye rejections
rejects_old=[8098.08 ,8145.11, 8212.01, 8281.08, 8323.06, 8362.01, 8577.02, 8594.08, 8607.08,
             8621.07, 8632.06, 8635.11, 8681.04, 8681.09, 8700.03, 8998.01]

rejects_cnn=[8384.03, 8856.12, 8986.02]

# Load merged comparison CSV
df = pd.read_csv(
    f"D:/ml_project/model_compare/{orb_start:05d}_{orb_end:05d}_merged.csv"
)



if method == "new":

    # Choose which method_match values to keep
    if unique:
        method_types = ["cnn_only"]
    else:
        method_types = ["cnn_only", "both"]

    selected_df = df[
        df["method_match"].isin(method_types)
        & df["color_flag_muv_cnn"].isin(["green", "yellow"])
    ].copy()

    # Remove by-eye rejected detections
    if reject:
        selected_df = selected_df[
            ~selected_df["nick_number"].isin(rejects_cnn)
        ]

    orbit_column = "orbit_number_cnn"


elif method == "old":

    # Choose which method_match values to keep
    if unique:
        method_types = ["old_only"]
    else:
        method_types = ["old_only", "both"]

    selected_df = df[
        df["method_match"].isin(method_types)
        & df["color_flag_muv_old"].isin(["green", "yellow"])
    ].copy()

    # Remove by-eye rejected detections
    if reject:
        selected_df = selected_df[
            ~selected_df["nick_number"].isin(rejects_old)
        ]

    orbit_column = "orbit_number_old"


else:
    raise ValueError("method must be 'new' or 'old'")


# Get orbit numbers to include in PDF
unique_orbits = set(
    selected_df[orbit_column]
    .dropna()
    .astype(int)
    .unique()
)


print(f"Method: {method}")
print(f"Unique only: {unique}")
print(f"Detection rows: {len(selected_df)}")
print(f"Unique orbits: {len(unique_orbits)}")
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

    if method == "new":
        input_folder = Path(
            f"D:/nightglow/quicklooks_out_cnn/{orb_block:05d}"
        )

        output_pdf = (
            input_folder
            / f"combined_quicklooks_{orb_block:05d}.pdf"
        )
    else:
        input_folder = Path(
            f"D:/nightglow/quicklooks_out/{orb_block:05d}"
        )

        output_pdf = (
            input_folder
            / f"combined_quicklooks_{orb_block:05d}.pdf"
        )

    # Find orbit PDFs belonging to detections
    pdf_files = sorted(
        pdf
        for pdf in input_folder.glob("*.pdf")
        if pdf.name != output_pdf.name
        and get_orbit_number(pdf) in unique_orbits
    )

    if not pdf_files:
        print(
            f"No PDFs found in block "
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

suffix = "_unique" if unique else ""

if method == "new":
    base_folder = Path(
        "D:/nightglow/quicklooks_out_cnn"
    )

    final_output_pdf = (
        base_folder
        / f"combined_quicklooks_"
        f"{orb_start:05d}_{orb_end:05d}_new{suffix}.pdf"
    )

else:
    base_folder = Path(
        "D:/nightglow/quicklooks_out"
    )

    final_output_pdf = (
        base_folder
        / f"combined_quicklooks_"
        f"{orb_start:05d}_{orb_end:05d}_old{suffix}.pdf"
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