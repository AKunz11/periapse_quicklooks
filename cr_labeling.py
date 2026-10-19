import numpy as np
import pandas as pd
import tensorflow as tf
import glob
from pathlib import Path
from astropy.io import fits


BASE_L1C = Path("E:/nightglow/c")
CR_MODEL_FILE = Path("cosmic_ray_model.keras")
OUTPUT_CSV = Path("cr_predictions_all.csv")
CR_THRESHOLD = 0.75
CR_REFERENCE_ORBIT = 8212 # to get shape of data for building cnn cube

RUN_START = 8000
RUN_END = 8100      # exclusive, so the last orbit processed is 11999
BLOCK_SIZE = 100 


#What needs to happen
#1. Open L1c
#2. Extract WL dim
#3. If not same as what model uses, interpolate down/up
#4. Build cube stuff
#5. Predict!
#6. Output csv with predictions

def format_orbit(orbit):
    """Return a five-digit orbit/folder string."""
    return f"{orbit:05d}"

def orbit_block_start(orbit, block_size=100):
    """Return the beginning of the orbit's 100-orbit block."""
    return (orbit // block_size) * block_size

def get_fits(orbit):
    orbit_str = format_orbit(orbit)
     # Determine which 100-orbit folder contains this orbit.
    block_start = orbit_block_start(orbit, BLOCK_SIZE)
    block_folder = format_orbit(block_start)

    l1c_pattern = (BASE_L1C / block_folder / f"*periapse*orbit{orbit_str}*.gz")

    #l1b_pattern = (BASE_L1B / block_folder / f"*periapse*orbit{orbit_str}-{channel}*.gz")

    l1c_files = sorted(glob.glob(str(l1c_pattern)))
    #l1b_files = sorted(glob.glob(str(l1b_pattern)))

    return l1c_files

def get_dims(l1c_hdul):
    dimensions = l1c_hdul['spectrum_noaltbin'].data['muv'].shape
    scans_count = dimensions[0]
    integration_count = dimensions[1]
    spatial_bin_count = dimensions[2]
    spectral_bin_count = dimensions[3]
    return scans_count, integration_count, spatial_bin_count, spectral_bin_count

def get_reference_wavelength(reference_orbit=CR_REFERENCE_ORBIT):
    """Load the 256-bin wavelength grid used to train the CNN."""
    folder = BASE_L1C / f"{(reference_orbit // 100) * 100:05d}"
    filepath = next(
        folder.glob(f"mvn_iuv_l1c_periapse-orbit{reference_orbit:05d}*"),
        None,
    )
    if filepath is None:
        raise FileNotFoundError(
            f"No reference L1C file found for orbit {reference_orbit} in {folder}"
        )

    with fits.open(filepath, ignore_missing_simple=True, ignore_missing_end=True) as hdul:
        wavelength = np.asarray(
            hdul["observation"].data["WAVELENGTH_MUV"][0],
            dtype=float,
        )

    return wavelength[0, 0]

def preprocess_cnn_input(cube):
    """Match cnn_train2 preprocessing exactly.

    The training script loads X and passes it directly to model.fit without
    scaling, standardization, clipping, or per-integration normalization.
    Therefore inference must preserve the raw detector-value scale.  Casting
    to float32 matches TensorFlow's normal input dtype without changing the
    numerical scale.
    """
    cube = np.asarray(cube, dtype=np.float32)

    if not np.all(np.isfinite(cube)):
        bad_count = np.size(cube) - np.count_nonzero(np.isfinite(cube))
        raise ValueError(
            f"CNN input contains {bad_count} non-finite values. "
            "These were not present or replaced during training, so inference "
            "is stopped rather than applying an unmatched normalization."
        )

    return cube

def build_cnn_orbit_input(l1c_hdul, target_wavelength):
    """Return training-matched CNN input: (scan * integration, spatial, 256)."""
    cube = np.asarray(
        l1c_hdul["SPECTRUM_NOALTBIN"].data["MUV"],
        dtype=np.float32,
    )
    wavelength = np.asarray(
        l1c_hdul["observation"].data["WAVELENGTH_MUV"][0],
        dtype=float,
    )

    if cube.shape[-1] != len(target_wavelength):
        cube_interp = np.empty(
            (*cube.shape[:-1], len(target_wavelength)),
            dtype=np.float32,
        )
        for scan_index in range(cube.shape[0]):
            for integration in range(cube.shape[1]):
                for spatial_bin in range(cube.shape[2]):
                    cube_interp[scan_index, integration, spatial_bin] = np.interp(
                        target_wavelength,
                        wavelength[scan_index, spatial_bin],
                        cube[scan_index, integration, spatial_bin],
                    )
        cube = cube_interp

    cube = cube.reshape(-1, cube.shape[2], cube.shape[3])
    return preprocess_cnn_input(cube)

#Check if there is already a csv with CNN predictions
if OUTPUT_CSV.exists():
    existing = pd.read_csv(OUTPUT_CSV)
    completed_orbits = set(existing["orbit_number"].unique())
else:
    completed_orbits = set()


#Load CNN once
cr_model = tf.keras.models.load_model(CR_MODEL_FILE)
cr_target_wavelength = get_reference_wavelength()
print(f"Cosmic-ray CNN loaded: {CR_MODEL_FILE}")


#Start looping through blocks
for block_start in range(RUN_START, RUN_END, BLOCK_SIZE):

    block_end = min(block_start + BLOCK_SIZE, RUN_END)

    print()
    print("=" * 70)
    print(f"Processing orbit block {block_start}-{block_end - 1}")
    print("=" * 70)

    #Store output rows for this 100-orbit block
    output_rows = []

    #Begin orbit loop
    for orbit_number in range(block_start, block_end):
        #Check if already in the CNN predictions csv
        if orbit_number in completed_orbits:
            print(f"Orbit {orbit_number} already processed, skipping")
            continue

        l1cfiles = get_fits(orbit_number)
        if len(l1cfiles) == 0:
            continue

        print(f"Processing orbit {orbit_number}")

        #Use most recent/matching L1C file
        l1c_file = l1cfiles[-1]

        try:
            with fits.open(l1c_file, ignore_missing_simple=True, ignore_missing_end=True) as hdull1c:

                #Determine dimensions
                n_scans, n_int, n_spa, n_spec = get_dims(hdull1c)

                #Build CNN input:
                #shape becomes (n_scans * n_int, spatial, wavelength)
                cr_X = build_cnn_orbit_input(hdull1c, cr_target_wavelength)

                #Predict
                cr_probabilities = cr_model.predict(cr_X, verbose=0).reshape(n_scans, n_int)

                #Threshold probabilities
                cr_predictions = cr_probabilities >= CR_THRESHOLD

                print(
                    f"  CNN cosmic rays: "
                    f"{cr_predictions.sum()} / {cr_predictions.size} "
                    f"integrations"
                )

                #Save one row per integration
                for scan in range(n_scans):
                    for integration in range(n_int):
                        probability = float(cr_probabilities[scan, integration])
                        prediction = int(cr_predictions[scan, integration])

                        output_rows.append({
                            "orbit_number": int(orbit_number),
                            "scan": int(scan),
                            "integration": int(integration),
                            "cr_probability": round(float(cr_probabilities[scan, integration]), 4),
                            "cr_pred": prediction,
                        })

        except Exception as e:
            print(f"  Failed orbit {orbit_number}: {e}")
            continue

    #Create DF for the current block
    block_df = pd.DataFrame(
        output_rows,
        columns=[
            "orbit_number",
            "scan",
            "integration",
            "cr_probability",
            "cr_pred",
        ]
    )

    #Append if file already exists, otherwise create it with headers
    if OUTPUT_CSV.exists():
        block_df.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)

    else:
        block_df.to_csv(OUTPUT_CSV, mode="w", header=True, index=False)

    print(f"Added {len(block_df)} rows to {OUTPUT_CSV}")