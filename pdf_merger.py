from pathlib import Path
from pypdf import PdfWriter
import glob
import pandas as pd
import re
import numpy as np

orb_start=8000
orb_end=9000
csv_folder = Path("D:/nightglow/tables_cnn")
csv_files = sorted(csv_folder.glob("*_detections_short_cnn.csv"))

df = pd.concat(
    (pd.read_csv(f) for f in csv_files),
    ignore_index=True
)

df = df[(df["orbit_number"]>=orb_start) & (df["orbit_number"]<orb_end) & (df["color_flag_muv"]=="green")]

unique_orbits = set(df["orbit_number"].dropna().astype(int))

def get_orbit_number(pdf_path):
    """Extract a standalone five-digit orbit number from the filename."""
    match = re.search(r"(?<!\d)(\d{5})(?!\d)", pdf_path.stem)

    return int(match.group(1)) if match else None

for num in np.arange(orb_start,orb_end,100):
    orb_block=num
    input_folder = Path(f"D:/nightglow/quicklooks_out_cnn/{orb_block:05d}")
    output_pdf = input_folder / f"combined_quicklooks_{orb_block:05d}.pdf"

    pdf_files = sorted(pdf for pdf in input_folder.glob("*.pdf") if pdf.name != output_pdf.name and get_orbit_number(pdf) in unique_orbits)

    if not pdf_files:
        print(f"No PDFs matching the orbit list were found in {orb_block:05d}")
        continue

    writer = PdfWriter()

    for pdf_file in pdf_files:
        print(f"Adding: {pdf_file.name}")
        writer.append(str(pdf_file))

    with output_pdf.open("wb") as file:
        writer.write(file)

    writer.close()

    print(f"\nCombined {len(pdf_files)} PDFs into:")
    print(output_pdf)
    print('='*80 + '\n')

base_folder = Path("D:/nightglow/quicklooks_out_cnn")
final_output_pdf = base_folder / f"combined_quicklooks_{orb_start:05d}_{orb_end:05d}.pdf"

combined_pdfs = sorted(
    base_folder.glob("*/combined_quicklooks_*.pdf"),
    key=lambda pdf: get_orbit_number(pdf) or -1
)

if not combined_pdfs:
    print("No combined PDFs were found. Skipping final merge.")

else:
    writer = PdfWriter()

    for pdf_file in combined_pdfs:
        print(f"Adding block: {pdf_file}")
        writer.append(str(pdf_file))

    with final_output_pdf.open("wb") as file:
        writer.write(file)

    writer.close()

    print(f"\nCombined {len(combined_pdfs)} block PDFs into:")
    print(final_output_pdf)
    print('='*80)