import numpy as np
import pandas as pd
from astropy.io import fits
from collections import defaultdict
from datetime import datetime
import glob
import os
import re
from PyUVS.data import calculate_calibration_curve
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go
from scipy.stats import pearsonr
import sklearn.metrics as metrics

#Open l1c file
def open_l1c(orbit_num):
    orbit_num0=orbit_num
    orbit_num=(orbit_num0//100)*100 #rounds down to nearest 100

    path=f"E:/nightglow/c/{orbit_num:05d}"
    file_list=glob.glob(path+'/mvn_iuv_l1c_periapse-orbit**.fits.gz')

    orbit_groups=defaultdict(list)

    for file in file_list:
        match = re.search(r'orbit(\d+)', file)
        if match:
            orbit=int(match.group(1))
            orbit_groups[orbit].append(file)

    return orbit_groups[orbit_num0][0]

def open_l1b(orbit_num):
    orbit_num0=orbit_num
    orbit_num=(orbit_num0//100)*100 #rounds down to nearest 100

    path=f"E:/nightglow/b/{orbit_num:05d}"
    file_list=glob.glob(path+'/mvn_iuv_l1b_periapse-orbit**.fits.gz')

    orbit_groups=defaultdict(list)

    for file in file_list:
        match = re.search(r'orbit(\d+)', file)
        if match:
            orbit=int(match.group(1))
            orbit_groups[orbit].append(file)

    return orbit_groups[orbit_num0]

def build_l1b_scan_mapping(l1b_files, l1c_hdul):
    """Map each L1C scan to the L1B file with the nearest start UTC."""
    if not l1b_files:
        raise ValueError("No L1B files were found for this orbit.")

    utc_format = "%Y/%j %b %d %H:%M:%S.%fUTC"
    l1b_times = []

    for filepath in l1b_files:
        with fits.open(filepath) as hdul:
            utc = hdul["integration"].data["utc"][0]
            l1b_times.append(
                (filepath, datetime.strptime(utc, utc_format))
            )

    mapping = []
    l1c_utc = l1c_hdul["integration"].data["utc"]

    for scan_index in range(l1c_utc.shape[0]):
        scan_time = datetime.strptime(l1c_utc[scan_index, 0], utc_format)
        best_file, best_time = min(
            l1b_times,
            key=lambda item: abs((item[1] - scan_time).total_seconds()),
        )
        time_difference = abs((best_time - scan_time).total_seconds())
        mapping.append(best_file)

        print(
            f"L1C scan {scan_index + 1} -> {os.path.basename(best_file)} "
            f"(dt={time_difference:.3f} s)"
        )

    return mapping

def get_dims(l1c_hdul):
    dimensions = l1c_hdul['spectrum_noaltbin'].data['muv'].shape
    scans_count = dimensions[0]
    integration_count = dimensions[1]
    spatial_bin_count = dimensions[2]
    spectral_bin_count = dimensions[3]
    return scans_count, integration_count, spatial_bin_count, spectral_bin_count

def get_l1c_baseline(l1c_file, scan, int):
    hdul=l1c_file
    orbit_const = hdul['radiance_noaltbin'].data['baseline_muv']
    const = hdul['radiance_noaltbin'].data['baseline_muv'][scan-1][int-1]

    threshold_const = np.nanmean(orbit_const) + np.nanstd(orbit_const)

    return const, threshold_const

# function to calculate the MLR coefficient in DN space from the calibrated MLR value
def get_coeff(MLR, template_DN, dlambda, cal):
    return MLR / (np.sum(template_DN / cal) * dlambda)

#calibration_curve = calculate_calibration_curve(hdull1b)
#wavelength = np.squeeze(hdull1b['observation'].data['wavelength'])
#spectra = hdull1b['detector_dark_subtracted'].data

def get_feature_indices(l1c_hdul):
    features = l1c_hdul['emission_features'].data['emission']
    solar_index = np.where(features == 'Solar cont. muv')[0][0]
    co_index = np.where(features == 'CO Cameron')[0][0]
    co2puvd_index = np.where(features == 'CO2pUVD')[0][0]
    o2972_index = np.where(features == 'O2972')[0][0]
    no_nightglow_index = np.where(features == 'NO band')[0][0]
    o1304_index = np.where(features == 'OI 130.4')[0][0]
    o1356_index = np.where(features == 'OI 135.6')[0][0]

    return solar_index, co_index, co2puvd_index, o2972_index, no_nightglow_index, o1304_index, o1356_index

def get_radiances(hdull1c, scan, solar_ind, co_ind, co2puvd_ind, o2972_ind, o1304_ind, o1356_ind, no_nightglow_ind):
    # get the radiances
    solar_radiance = hdull1c['radiance_noaltbin'].data['profile'][scan, :, :, solar_ind]
    co_cameron_radiance = hdull1c['radiance_noaltbin'].data['profile'][scan, :, :, co_ind]
    co2p_uvd_radiance = hdull1c['radiance_noaltbin'].data['profile'][scan, :, :, co2puvd_ind]
    o2972_radiance = hdull1c['radiance_noaltbin'].data['profile'][scan, :, :, o2972_ind]
    o1304_radiance = hdull1c['radiance_noaltbin'].data['profile'][scan, :, :, o1304_ind]
    o1356_radiance = hdull1c['radiance_noaltbin'].data['profile'][scan, :, :, o1356_ind]
    no_nightglow_radiance = hdull1c['radiance_noaltbin'].data['profile'][scan, :, :, no_nightglow_ind]

    return solar_radiance, co_cameron_radiance, co2p_uvd_radiance, o2972_radiance, o1304_radiance, o1356_radiance, no_nightglow_radiance

def get_all_coeffs(solar_radiance, co_cameron_radiance, co2p_uvd_radiance, no_nightglow_radiance,
                   template_solar_continuum, template_co_cameron, template_co2p_uvd, template_no_nightglow,
                   dwavelength, calibration_curve, good_int, good_spa):
    
    # get DN coefficients
    solar_coeff = get_coeff(solar_radiance[good_int, good_spa], template_solar_continuum, dwavelength,
                            calibration_curve[3, :])
    co_coeff = get_coeff(co_cameron_radiance[good_int, good_spa], template_co_cameron, dwavelength,
                         calibration_curve[3, :])
    co2p_coeff = get_coeff(co2p_uvd_radiance[good_int, good_spa], template_co2p_uvd, dwavelength,
                           calibration_curve[3, :])
    no_coeff = get_coeff(no_nightglow_radiance[good_int, good_spa], template_no_nightglow, dwavelength,
                         calibration_curve[3, :])

    return solar_coeff, co_coeff, co2p_coeff, no_coeff

def get_templates(n_spec):
    if n_spec == 219:
        load_spec = 256
    else:
        load_spec = n_spec
    templates = np.load('PyUVS/ancillary/templates_%i.npy' % load_spec, allow_pickle=True)
    template_wavelength = templates.item().get('wavelength')
    dwavelength = np.diff(template_wavelength)[0]
    template_solar_continuum = templates.item().get('solar_continuum')
    template_co_cameron = templates.item().get('co_cameron')
    template_co2p_uvd = templates.item().get('co2p_uvd')
    template_no_nightglow = templates.item().get('no_nightglow')

    # account for 219 spectral bins
    if n_spec == 219:
        start = 6
        template_wavelength = template_wavelength[start:start + n_spec]
        template_solar_continuum = template_solar_continuum[start:start + n_spec]
        template_co_cameron = template_co_cameron[start:start + n_spec]
        template_co2p_uvd = template_co2p_uvd[start:start + n_spec]
        template_no_nightglow = template_no_nightglow[start:start + n_spec]

    # scaling factors
    co_cameron_scale = 1.0
    if n_spec == 200:
        co_cameron_scale = 1.2010
    if n_spec == 219:
        co_cameron_scale = 1.0029

    return template_wavelength, dwavelength, template_solar_continuum, template_co_cameron, template_co2p_uvd, template_no_nightglow, co_cameron_scale

def calculate_correlation(plot_wavelength, background_spectrum, aurora_spectrum):
    # calcualte quality of fit using only the bright Cameron band features
    minwave_ind = np.argmin(np.abs(plot_wavelength - 190))
    maxwave_ind = np.argmin(np.abs(plot_wavelength - 246))
    y_pred = aurora_spectrum[minwave_ind:maxwave_ind]
    y_true = background_spectrum[minwave_ind:maxwave_ind]
    try:
        r2 = metrics.r2_score(y_true, y_pred)
        r = pearsonr(y_true, y_pred)[0]
    except ValueError:
        r2 = np.nan
        r = np.nan

    # return the goodness-of-fit metrics
    return r2, r

def build_spectrum(orbit_num, scan, integ):
    # Treat scan and integration as human-readable, starting at 1
    scan_index = scan - 1
    integration_index = integ - 1

    l1c_path = open_l1c(orbit_num)
    l1b_paths = open_l1b(orbit_num)

    with fits.open(l1c_path) as hdull1c:
        n_scans, n_int, n_spa, n_spec = get_dims(hdull1c)

        if not 1 <= scan <= n_scans:
            raise ValueError(
                f"Scan must be between 1 and {n_scans}; received {scan}."
            )

        if not 1 <= integ <= n_int:
            raise ValueError(
                f"Integration must be between 1 and {n_int}; received {integ}."
            )

        matched_l1b_paths = build_l1b_scan_mapping(l1b_paths, hdull1c)
        matched_l1b_path = matched_l1b_paths[scan_index]

        with fits.open(matched_l1b_path) as hdull1b:
            # Calculate these for the time-matched L1B file.
            calibration_curve = calculate_calibration_curve(hdull1b)
            wavelength = np.squeeze(
                hdull1b["observation"].data["wavelength"]
            ).copy()
            spectra = hdull1b["detector_dark_subtracted"].data.copy()

        # Select the requested integration for every spatial bin
        good_int = np.full(n_spa, integration_index, dtype=int)
        good_spa = np.arange(n_spa)

        const, threshold_const = get_l1c_baseline(
            hdull1c,
            scan,
            integ
        )

        (
            solar_ind,
            co_ind,
            co2puvd_ind,
            o2972_ind,
            no_nightglow_ind,
            o1304_ind,
            o1356_ind,
        ) = get_feature_indices(hdull1c)

        (
            template_wavelength,
            dwavelength,
            template_solar_continuum,
            template_co_cameron,
            template_co2p_uvd,
            template_no_nightglow,
            co_cameron_scale,
        ) = get_templates(n_spec)

        (
            solar_radiance,
            co_cameron_radiance,
            co2p_uvd_radiance,
            o2972_radiance,
            o1304_radiance,
            o1356_radiance,
            no_nightglow_radiance,
        ) = get_radiances(
            hdull1c,
            scan_index,
            solar_ind,
            co_ind,
            co2puvd_ind,
            o2972_ind,
            o1304_ind,
            o1356_ind,
            no_nightglow_ind,
        )

        (
            solar_coeff,
            co_coeff,
            co2p_coeff,
            no_coeff,
        ) = get_all_coeffs(
            solar_radiance,
            co_cameron_radiance,
            co2p_uvd_radiance,
            no_nightglow_radiance,
            template_solar_continuum,
            template_co_cameron,
            template_co2p_uvd,
            template_no_nightglow,
            dwavelength,
            calibration_curve,
            good_int,
            good_spa,
        )

        # Average the seven spatial bins for the measured spectrum
        measured_spectrum = np.nanmean(
            spectra[good_int, good_spa],
            axis=0
        )

        # Fitted components
        solar_component = (
            template_solar_continuum * np.nanmean(solar_coeff)
        )

        co_component = (
            template_co_cameron
            * np.nanmean(co_coeff)
            * co_cameron_scale
        )

        co2p_component = (
            template_co2p_uvd * np.nanmean(co2p_coeff)
        )

        no_component = (
            template_no_nightglow * np.nanmean(no_coeff)
        )

        constant_value = np.nanmean(const)

        # Convert scalar constant into an array for plotting
        constant_component = np.full(
            n_spec,
            constant_value,
            dtype=float
        )

        aurora_spectrum = co_component + co2p_component

        total_fit = (
            solar_component
            + co_component
            + co2p_component
            + no_component
            + constant_component
        )

        # Background after removing only the background-model terms
        background_spectrum = (
            measured_spectrum
            - solar_component
            - no_component
            - constant_component
        )

        plot_wavelength = np.nanmean(
            wavelength[good_spa],
            axis=0
        )

    return {
        "wavelength": plot_wavelength,
        "measured": measured_spectrum,
        "aurora": aurora_spectrum,
        "solar": solar_component,
        "co": co_component,
        "co2p": co2p_component,
        "no": no_component,
        "constant": constant_component,
        "total_fit": total_fit,
        "background": background_spectrum,
        "threshold_constant": threshold_const,
        "constant_value": constant_value,
        "l1b_path": matched_l1b_path,
    }

def build_detector_image(orbit_num, scan, integ, l1b_path=None):
    if l1b_path is None:
        l1c_path = open_l1c(orbit_num)
        l1b_paths = open_l1b(orbit_num)

        with fits.open(l1c_path) as hdull1c:
            n_scans, _, _, _ = get_dims(hdull1c)
            if not 1 <= scan <= n_scans:
                raise ValueError(
                    f"Scan must be between 1 and {n_scans}; received {scan}."
                )
            l1b_path = build_l1b_scan_mapping(l1b_paths, hdull1c)[scan - 1]

    with fits.open(l1b_path) as hdull1b:
        wavelength = np.squeeze(
            hdull1b["observation"].data["wavelength"]
        ).copy()
        spectra = hdull1b["detector_dark_subtracted"].data[
            integ - 1, :, :
        ].copy()
        log_spectra = np.log10(np.clip(spectra, 50, None))
        #detector_image = np.nanmean(spectra, axis=0)
    return wavelength, spectra, log_spectra

app = Dash(__name__)

#Build check boxes for components to display
COMPONENT_OPTIONS = [
    {"label": "Measured spectrum", "value": "measured"},
    #{"label": "Background spectrum", "value": "background"},
    {"label": "Total MLR fit", "value": "total_fit"},
    {"label": "Solar continuum", "value": "solar"},
    {"label": "CO Cameron", "value": "co"},
    {"label": "CO₂⁺ UVD", "value": "co2p"},
    {"label": "NO nightglow", "value": "no"},
    {"label": "Constant", "value": "constant"},
]

#Add labels and line styles for each component
TRACE_NAMES = {
    "measured": "Measured spectrum",
    #"background": "Background spectrum",
    "total_fit": "Total MLR fit",
    "solar": "Solar continuum",
    "co": "CO Cameron",
    "co2p": "CO₂⁺ UVD",
    "no": "NO nightglow",
    "constant": "Constant",
}

LINE_STYLES = {
    "measured": {"width": 3},
    #"background": {"width": 3},
    "total_fit": {"width": 3, "dash": "dash"},
    "solar": {"width": 2},
    "co": {"width": 2},
    "co2p": {"width": 2},
    "no": {"width": 2},
    "constant": {"width": 2, "dash": "dot"},
}

#Div for the entire page
app.layout = html.Div(
    [
        #Header for the title page
        html.H2("IUVS MLR Spectrum Viewer"),

        #Section for all input fields
        html.Div(
            [
                #Orbit input
                html.Div(
                    [
                        html.Label("Orbit"),
                        dcc.Input(
                            id="orbit-input",
                            type="number",
                            value=8604,
                            min=1,
                            step=1,
                            style={"width": "60px"}
                        ),
                    ]
                ),
                #Scan input
                html.Div(
                    [
                        html.Label("Scan"),
                        dcc.Input(
                            id="scan-input",
                            type="number",
                            value=7,
                            min=1,
                            step=1,
                            style={"width": "60px"}
                        ),
                    ]
                ),
                #Integration input
                html.Div(
                    [
                        html.Label("Integration"),
                        dcc.Input(
                            id="integration-input",
                            type="number",
                            value=2,
                            min=1,
                            step=1,
                            style={"width": "60px"}
                        ),
                    ]
                ),
                #Reload data button
                html.Button(
                    "Load spectrum",
                    id="load-button",
                    n_clicks=0,
                ),
            ],
            style={
                "display": "flex",
                "gap": "20px",
                "alignItems": "end",
                "flexWrap": "wrap",
                "marginBottom": "20px",
            },
        ),

        html.Label("Displayed components"),

        dcc.Checklist(
            id="component-checklist",
            options=COMPONENT_OPTIONS,
            value=[
                "measured",
                #"background",
                "total_fit",
                "solar",
                "co",
                "co2p",
                "no",
                "constant",
            ],
            inline=True,
            style={"marginBottom": "15px"},
        ),

        html.Div(
            id="spectrum-information",
            style={"marginBottom": "10px"},
        ),
        #Error message display (if we have any)
        html.Div(
            id="error-message",
            style={
                "color": "red",
                "fontWeight": "bold",
                "marginBottom": "10px",
            },
        ),
        #Add graph section
        dcc.Graph(
            id="detector_image",
            config={
                "displaylogo": False,
                "scrollZoom": False,
            },
        ),

        dcc.Graph(
            id="spectrum-graph",
            config={
                "displaylogo": False,
                "scrollZoom": True,
            },
        ),

        
    ],

    style={
        "maxWidth": "1400px",
        "margin": "30px auto",
        "padding": "0 25px",
    },
)

@app.callback(
    #Outputs: what get updated when a callback is initialized
    Output("spectrum-graph", "figure"),
    Output("detector_image", "figure"),
    Output("spectrum-information", "children"),
    Output("error-message", "children"),
    #Inputs: what initializes a callback (e.g. clicking the load button)
    Input("load-button", "n_clicks"),
    Input("component-checklist", "value"),
    #States: changes that only take effect AFTER the callback
    State("orbit-input", "value"),
    State("scan-input", "value"),
    State("integration-input", "value"),
)
#Function that is run after the callback (updates spectrum after box tick or button press)
#NOTE: This is actually run at initialization with n_clicks=0, and just initializes a
def update_spectrum(
    n_clicks,
    selected_components,
    orbit_num,
    scan,
    integration,
):
    #Error handling figure (safe mode)
    empty_figure = go.Figure()
    empty_figure_d = go.Figure()

    if orbit_num is None or scan is None or integration is None:
        return (
            empty_figure,
            empty_figure_d,
            "",
            "Enter an orbit, scan, and integration.",
        )
    #Get spectrum data and build figure based on inputs
    try:
        orbit_num = int(orbit_num)
        scan = int(scan)
        integration = int(integration)

        spectrum_data = build_spectrum(
            orbit_num,
            scan,
            integration,
        )

        detector_wavelength, detector_spectra, detector_log_spectra = build_detector_image(
            orbit_num,
            scan,
            integration,
            spectrum_data["l1b_path"],
        )

    #If error, set figure to empty and print error
    except Exception as error:
        return (
            empty_figure,
            empty_figure_d,
            "",
            f"Could not load spectrum: {error}",
        )

    #The initial figure that is loaded with the app (starts empty)
    fig = go.Figure()
    fig_detector = go.Figure()

    wavelength = spectrum_data["wavelength"]

    #Build MLR figure
    for component in selected_components:
        #Skip un-checked components
        if component not in spectrum_data:
            continue
        #Add in the selected components
        fig.add_trace(
            go.Scatter(
                x=wavelength,
                y=spectrum_data[component],
                mode="lines",
                name=TRACE_NAMES[component],
                line=LINE_STYLES[component],
                hovertemplate=(
                    "Wavelength: %{x:.2f} nm"
                    "<br>DN: %{y:.3f}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )

    #Build detector fig
    fig_detector.add_trace(
        go.Heatmap(
            z=detector_log_spectra,
            x=np.nanmean(detector_wavelength, axis=0),
            colorscale="Inferno",
            zmin=np.log10(50),
            zmax=np.log10(5000),
            colorbar=dict(showticklabels=False),
            showscale=False,
            hovertemplate="Wavelength: %{x:.2f} nm<br>DN: %{z:.3f}<extra></extra>",
        )
    )

    peaks = [193, 199.5, 206, 216, 228, 240, 289]
    for peak in peaks:
        fig_detector.add_vline(
            x=peak, 
            line_width=1, 
            line_dash="dash", 
            line_color="white"
        )

    #Load in labels and stuff
    fig.update_layout(
        title=(
            f"Orbit {orbit_num} — "
            f"Scan {scan}, Integration {integration}"
        ),
        xaxis_title="Wavelength (nm)",
        yaxis_title="DN",
        hovermode="x unified",
        template="plotly_white",
        height=750,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        margin={
            "l": 70,
            "r": 30,
            "t": 60,
            "b": 35,
        },
        uirevision=f"{orbit_num}-{scan}-{integration}",
    )

    #Load in labels and stuff for detector image
    fig_detector.update_layout(
        xaxis_title="Wavelength (nm)",
        xaxis=dict(showticklabels=True),
        yaxis_title="Spatial Bin",
        template="plotly_white",
        height=250,
        margin={
            "l": 70,
            "r": 30,
            "t": 120,
            "b": 70,
        },
        uirevision=f"{orbit_num}-{scan}-{integration}",
    )

    #Calculate Pearson R
    r2, r = calculate_correlation(spectrum_data['wavelength'], spectrum_data['background'], spectrum_data['aurora'])

    #Bonus fun facts to be plotted (NOTE: check to see if threshold working)
    information = (
        f"Constant = {spectrum_data['constant_value']:.3f} DN | "
        f"Orbit threshold = "
        f"{spectrum_data['threshold_constant']:.3f} DN | "
        f"Pearson R: {r:.2f}"
    )

    return fig, fig_detector, information, ""


def main():
    #Stuff to view the specifics on each integration
    orbit_num=8604
    hdull1c=fits.open(open_l1c(orbit_num))
    n_scans, n_int, n_spa, n_spec = get_dims(hdull1c)
    headers=['orbit','scan','int','threshold', 'max_base', 'ratio', 'mean_base']
    scan_ints=[[7,2],[4,14],[4,6]]
    print(f'{headers[0]:_^16}|{headers[1]:_^16}|{headers[2]:_^16}|{headers[3]:_^16}|{headers[4]:_^16}|{headers[5]:_^16}|{headers[6]:_^16}')
    for coord in scan_ints:
        const, threshold_const = get_l1c_baseline(hdull1c, coord[0], coord[1])
        print(f'{orbit_num:<16}|{coord[0]:<16}|{coord[1]:<16}|{int(threshold_const):<16}|{int(np.max(const)):<16}|{np.max(const)/threshold_const:16.4f}|{np.mean(const):<16}')
        print(f'{const}')

    
    


if __name__ == "__main__":
    print('Copy link into browser')
    app.run(debug=True)


