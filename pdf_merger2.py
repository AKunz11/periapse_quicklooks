from pathlib import Path
from pypdf import PdfWriter
import pandas as pd
import re


# ============================================================
# SETTINGS
# ============================================================

orbit_start = 23000
orbit_end = 23100
drive_letter = "E"  # Change this to "D" if the files are on the D drive

# Folder containing the detection CSV
csv_folder = Path(f"{drive_letter}:/nightglow/tables_cnn")

# Folder containing the orbit PDFs
pdf_folder = Path(
    f"{drive_letter}:/nightglow/quicklooks_out_cnn/{orbit_start:05d}"
)

# Input detection CSV
csv_file = (
    csv_folder
    / f"{orbit_start:05d}_{orbit_end:05d}_detections_short_cnn.csv"
)

# Output PDF
output_pdf = (
    pdf_folder
    / f"{orbit_start:05d}_{orbit_end:05d}_detections.pdf"
)


# ============================================================
# LOAD DETECTIONS
# ============================================================

try:
    df = pd.read_csv(csv_file)

except FileNotFoundError:
    print(f"CSV file not found: {csv_file}\nMaybe no detections in this orbit range?")
    exit(1)

# Get every unique orbit containing at least one detection
detection_orbits = set(
    df["orbit_number"]
    .dropna()
    .astype(int)
    .unique()
)

print(f"Detection rows: {len(df)}")
print(f"Detection orbits: {len(detection_orbits)}")
print("Orbits:", sorted(detection_orbits))
print("=" * 80)


# ============================================================
# EXTRACT ORBIT NUMBER FROM PDF NAME
# ============================================================

def get_orbit_number(pdf_path):
    """
    Extract a standalone five-digit orbit number from a PDF filename.

    Example:
        mvn_iuv_quicklook_orbit15234.pdf
        -> 15234
    """

    match = re.search(
        r"(?<!\d)(\d{5})(?!\d)",
        pdf_path.stem
    )

    return int(match.group(1)) if match else None


# ============================================================
# FIND MATCHING PDFs
# ============================================================

pdf_files = sorted(
    pdf
    for pdf in pdf_folder.glob("*.pdf")
    if pdf.name != output_pdf.name
    and get_orbit_number(pdf) in detection_orbits
)


print(f"\nFound {len(pdf_files)} matching PDFs.")

for pdf in pdf_files:
    print(f"  {get_orbit_number(pdf)}: {pdf.name}")


# ============================================================
# MERGE
# ============================================================

if not pdf_files:

    print("\nNo matching PDFs found. Nothing to merge.")

else:

    writer = PdfWriter()

    for pdf_file in pdf_files:
        print(f"Adding: {pdf_file.name}")
        writer.append(str(pdf_file))

    with output_pdf.open("wb") as file:
        writer.write(file)

    writer.close()

    print("\n" + "=" * 80)
    print(f"Merged {len(pdf_files)} orbit PDFs.")
    print(f"Output: {output_pdf}")