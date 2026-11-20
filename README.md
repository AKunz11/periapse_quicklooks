```text
/*==========================================================*
 *
 *   ███╗   ███╗ █████╗ ██╗   ██╗███████╗███╗   ██╗
 *   ████╗ ████║██╔══██╗██║   ██║██╔════╝████╗  ██║
 *   ██╔████╔██║███████║██║   ██║█████╗  ██╔██╗ ██║
 *   ██║╚██╔╝██║██╔══██║╚██╗ ██╔╝██╔══╝  ██║╚██╗██║
 *   ██║ ╚═╝ ██║██║  ██║ ╚████╔╝ ███████╗██║ ╚████║
 *   ╚═╝     ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚══════╝╚═╝  ╚═══╝
 *
 *                I U V S  /  P E R I A P S E
 *                  Q U I C K L O O K S
 *
 *==========================================================*/
```

# MAVEN/IUVS Periapse Quicklooks

Generate periapse limb-scan quicklook PDFs and candidate aurora tables from MAVEN/IUVS Level 1B and custom Level 1C data. The main workflow combines spectral and geometry diagnostics with a convolutional neural network (CNN) and fit-quality filters to reject likely cosmic-ray contamination.

This repository also includes tools for downloading MAVEN SPICE kernels, reviewing individual multiple-linear-regression (MLR) fits in a browser, exporting CNN cosmic-ray predictions, and merging quicklooks for detected orbits.

## Repository contents

| Path | Purpose |
| --- | --- |
| `limb_aurora_ql_CNN_ssd_updated_v4.py` | Main batch processor. Creates orbit quicklook PDFs, detection CSVs, and per-orbit spectrum archives. |
| [`side_segments/aurora_ql_side_segment.py`](side_segments/aurora_ql_side_segment.py) | Version of the original quicklook code adapted for inlimb and outlimb observations. |
| [`sep_events/sep_events_test.py`](sep_events/sep_events_test.py) | Filters aurora detections for candidate solar energetic particle (SEP) events. |
| `cosmic_ray_model.keras` | Trained CNN used to score integrations for cosmic-ray contamination. |
| `download_spice.py` | Command-line installer for the generic and MAVEN SPICE kernels used by the quicklook workflow. |
| `mlr_app.py` | Local Dash application for inspecting an orbit/scan/integration and its MLR components. |
| `cr_labeling.py` | Standalone batch inference script that writes one CNN prediction per integration. |
| `pdf_merger2.py` | Merges quicklook PDFs for detected orbits listed in a detection CSV. |
| `PyUVS/` | Local IUVS analysis utilities plus calibration, template, and map assets. |
| `requirements.txt` | Pinned Python environment used by the project. |

The `updated_v4` script is the current main entry point. Older quicklook versions are intentionally excluded by `.gitignore` and should be treated as historical working copies.

## Requirements

- Python 3 with `venv` support
- MAVEN/IUVS periapse Level 1B MUV FITS files
- Custom periapse Level 1C FITS files whose `emission_features` table includes the `NO band` feature (the default L1C product is not sufficient)
- MAVEN and generic SPICE kernels covering the requested orbit range
- Enough local storage for the science data, kernels, and generated PDFs

The scripts currently use Windows-style paths, but their `pathlib`-based settings can be changed for another platform.

## Installation

From the repository root, create and activate a virtual environment, then install the pinned dependencies.

For your sake, use a virtual environment (VENV) to easily keep track of libraries.

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run all commands from the repository root. The main script and the MLR viewer load `cosmic_ray_model.keras` and files under `PyUVS/ancillary/` using relative paths.

## Expected data layout

Both L1B and L1C inputs are grouped into zero-padded 100-orbit directories. For example, orbit 18742 is read from the `18700` directories.

```text
E:/nightglow/
├── b/
│   └── 18700/
│       └── *periapse*orbit18742-muv*.fits.gz
└── c/
    └── 18700/
        └── *periapse*orbit18742*.fits.gz
```

The main script uses the most recent matching L1C file and matches each L1C scan to the L1B file with the nearest start time.

## Download SPICE kernels

(NOTICE) I've never actually tested this, so not sure if it works or is necessary.
At the bare minimum, I know you need `naif0012.tls`, `pck00011.tpc`, `de440s.bsp`, `mar097.bsp`

The downloader treats `--orbit-start` as inclusive and `--orbit-end` as exclusive:

```powershell
python download_spice.py --orbit-start 18700 --orbit-end 18800 --output spice
```

It creates the structure expected by `PyUVS.spice.load_iuvs_spice`:

```text
spice/
├── generic_kernels/
│   ├── generic.tm
│   └── ...
└── mvn/
    ├── mvn.tm
    └── ...
```

Existing non-empty files are skipped. Use `--force` to redownload them or `--all-ck` to download every MAVEN CK instead of selecting kernels around the requested coverage. The script performs structural checks and a SpiceyPy load test when it finishes.

## Configure the main workflow

Edit the configuration block near the top of `limb_aurora_ql_CNN_ssd_updated_v4.py`:

```python
CR_MODEL_FILE = Path("cosmic_ray_model.keras")
CR_THRESHOLD = 0.75
CR_REFERENCE_ORBIT = 8212

RUN_START = 18700
RUN_END = 18800       # exclusive
BLOCK_SIZE = 100

BASE_L1C = Path("D:/nightglow/c_limb")
BASE_L1B = Path("D:/nightglow/b_limb")
TABLE_OUTPUT_DIR = Path("D:/nightglow/tables_cnn_limb") # output for the csv files
QUICKLOOK_OUTPUT_DIR = Path("D:/nightglow/quicklooks_out_cnn_limb") # output for quicklook pdfs
SPECTRA_OUTPUT_DIR = Path("D:/nightglow/spectra_out_limb") # output for spectra .npz files
SPICE_DIR='D:/periapse_quicklooks/spice' # where u put the spice kernels
LIMB_TYPE="outlimb" # periapse, inlimb, outlimb (apoapse is unsupported currently)
```

Also update the SPICE path later in the file:

```python
load_iuvs_spice(spice_directory="D:/periapse_quicklooks_updated/spice")
```

`CR_REFERENCE_ORBIT` must exist under `BASE_L1C`; its wavelength grid defines the CNN input grid. It can realistically be any l1c file, but safe to use the one that is in there by default. `RUN_END` is exclusive, so the example processes orbits 18700 through 18799.

## Generate quicklooks

After configuring the paths and orbit range:

```powershell
python limb_aurora_ql_CNN_ssd_updated_v4.py
```

For each 100-orbit block, the script writes:

```text
tables_cnn/
└── 18700_18800_detections_short_cnn.csv

quicklooks_out_cnn/
└── 18700/
    └── orbit18742.pdf

spectra_out/
└── 18700/
    └── orbit18742_spectra.npz
```

The block detection CSV is replaced at the start of a run for that block. Existing quicklook PDFs and spectrum archives with matching names are overwritten.

### Detection and cosmic-ray annotations

Scan labels use the following detection colors:

- Green: candidate detection with more than five passing pixels and either R-squared greater than 0.25 or Pearson R greater than 0.55.
- Yellow: marginal candidate with three to five passing pixels and Pearson R of at least 0.60.
- Red: did not meet the detection criteria.
- White: correlation could not be calculated.

Detector-image shading identifies rejected integrations:

- Cyan: rejected by the CNN only.
- Red: rejected by the legacy fit-quality filters only.
- Green: rejected by both methods.

The fit-quality rejection combines the orbit baseline threshold, chi-square threshold, and a solar-continuum coefficient threshold. Low-solar-zenith-angle integrations are also excluded. Yellow outlines on the scan images mark integrations rejected as cosmic rays by either the CNN or the fit-quality method.

## Candidate SEP event criteria

`sep_events/sep_events_test.py` starts with green-flagged MUV detections with Pearson R of at least 0.55 and a `count` of at least 6, excluding rows typed `NONE` or `NON-DET`. A scan can qualify at either level if the detection altitude is at least 50 km and the CO₂⁺ UVD signal-to-noise ratio is at least 3.

| Scan level | Minimum CO₂⁺ UVD radiance (`CO2p_radiance`) | Altitude ceiling |
| --- | --- | --- |
| Strong | 0.18 | Below 112 km |
| Supporting | 0.10 | Below 130 km |

An orbit is flagged as a candidate solar energetic particle (SEP) event if it has at least one strong scan plus a second distinct strong or supporting scan, or at least three distinct supporting scans.

## Inspect MLR fits interactively

Before starting the viewer, update the L1C and L1B roots in `open_l1c()` and `open_l1b()` in `mlr_app.py`. Then run:

```powershell
python mlr_app.py
```

Open the local URL printed by Dash. Enter a one-based orbit, scan, and integration, then select the components to compare: measured spectrum, total fit, solar continuum, CO Cameron, CO2+ UVD, NO nightglow, and the constant term. The lower panel shows the corresponding detector image.

## Export CNN predictions only

`cr_labeling.py` runs the CNN without generating full quicklooks. Configure `BASE_L1C`, `CR_MODEL_FILE`, `OUTPUT_CSV`, `CR_THRESHOLD`, `CR_REFERENCE_ORBIT`, `RUN_START`, and `RUN_END` at the top of the file, then run:

```powershell
python cr_labeling.py
```

The output contains `orbit_number`, zero-based `scan` and `integration`, `cr_probability`, and the thresholded `cr_pred`. Orbits already present in the CSV are skipped, and new block results are appended.

## Merge detected-orbit PDFs

Edit `orbit_start`, `orbit_end`, and `drive_letter` at the top of `pdf_merger2.py`, then run:

```powershell
python pdf_merger2.py
```

The merger reads the block detection CSV, excludes yellow (marginal) rows, finds PDFs for the remaining unique orbit numbers, and writes one combined PDF into the block's quicklook directory.

## Troubleshooting

- **No files are found:** verify the zero-padded 100-orbit directory and the `periapse`/`orbitNNNNN` filename pattern.
- **`NO band` lookup fails:** the input is likely a default L1C product rather than the required custom product containing the NO nightglow feature.
- **The CNN reference file is missing:** download or copy the orbit set by `CR_REFERENCE_ORBIT`, or change that setting to an available compatible reference orbit.
- **SPICE fails to load:** ensure both `generic_kernels/generic.tm` and `mvn/mvn.tm` exist under the exact directory passed to `load_iuvs_spice`.
- **The Dash viewer cannot find an orbit:** update both path-building functions in `mlr_app.py`; they do not reuse the main script's settings.
- **A run skips an orbit:** the main workflow skips orbits with missing L1B/L1C inputs, insufficient matching L1B scans, or no pixels above its solar-zenith-angle gate.


## Next-up / TODO

- CNN implementation for inlimb/outlimb scans would be beneficial. Sounds really hard, but just follow the steps in my other repo for CNN stuff and it should be straightforward. 
- SEP event detection may need to be tinkered with. I have it pretty good (I think) right now, but you may need to get creative to try and exclude bright aurora examples. Also should be updated to work with inlimb/outlimb data

## Credits

The limb-aurora quicklook code was written by Zac Milby and Kyle Connour, with annotations by Jay Cessna and Alexander Kunz. The PDF merger, MLR viewer, CNN implementation, SEP event identifier (WIP) were written by Alexander Kunz.
Feel free to email if there are any questions. Or find my linkedin or something :D
