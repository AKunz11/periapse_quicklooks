import glob
import os
import time
from datetime import datetime
from warnings import filterwarnings

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import spiceypy as spice
from astropy.io import fits
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.patches import Rectangle
from scipy.io import readsav
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
from shapely.geometry import LineString
from PyUVS.graphics import CO_colormap, CO2p_colormap, H_colormap, JGR_format, color_dict
from PyUVS.variables import R_Mars_km
from PyUVS.data import calculate_calibration_curve
from PyUVS.spice import load_iuvs_spice
from pathlib import Path

# Orbit-block configuration
RUN_START = 8000
RUN_END = 8100      # exclusive; last orbit processed is 11999
BLOCK_SIZE = 100

BASE_L1C = Path("D:/nightglow/c")
BASE_L1B = Path("D:/nightglow/b")
TABLE_OUTPUT_DIR = Path("D:/nightglow/tables")
QUICKLOOK_OUTPUT_DIR = Path("D:/nightglow/quicklooks_out")
SPECTRA_OUTPUT_DIR = Path("D:/nightglow/spectra_out")

TABLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
QUICKLOOK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SPECTRA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def format_orbit(orbit):
    """Return a five-digit orbit/folder string."""
    return f"{orbit:05d}"


def orbit_block_start(orbit, block_size=BLOCK_SIZE):
    """Return the beginning of the block containing an orbit."""
    return (orbit // block_size) * block_size

# print progress report
def progress_report(start_time, iterator, orbit, n_orbits):
    current_time = time.time()
    seconds = current_time - start_time
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    print(' ' * 100, end='\r')
    print('Processing orbit %i...%i/%i files (%.2f%%), %d:%02d:%02d elapsed'
          % (orbit, iterator + 1, n_orbits, (iterator + 1) / float(n_orbits) * 100., hours, minutes, seconds), end='\r')


# function to return FITS files for levels 1B and 1C
def get_fits(orbit, channel):
    orbit_str = format_orbit(orbit)

    # Determine the input folder from the orbit itself.
    block_folder = format_orbit(orbit_block_start(orbit))

    l1c_pattern = BASE_L1C / block_folder / f"*periapse*orbit{orbit_str}*.gz"
    l1b_pattern = BASE_L1B / block_folder / f"*periapse*orbit{orbit_str}-{channel}*.gz"

    l1c_files = sorted(glob.glob(str(l1c_pattern)))
    l1b_files = sorted(glob.glob(str(l1b_pattern)))

    return l1c_files, l1b_files


def get_dims(l1c_hdul):
    dimensions = l1c_hdul['spectrum_noaltbin'].data['muv'].shape
    scans_count = dimensions[0]
    integration_count = dimensions[1]
    spatial_bin_count = dimensions[2]
    spectral_bin_count = dimensions[3]
    return scans_count, integration_count, spatial_bin_count, spectral_bin_count


def rearrange_dimensions(array):
    array = np.swapaxes(array, 1, 2)
    array = np.swapaxes(array, 0, 1)
    return array


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


def plot_spacecraft_altitude():
    x = np.arange(1, n_scans + 1, 1 / n_int)
    y = sc_alt.flatten()
    alt_dist_axis.plot(x, y, marker='.', ms=2, color='white', label='Spacecraft Altitude')
    alt_dist_axis.axvline(x[np.where(y == np.nanmin(y))], color=color_dict['red'], linestyle='--')


def plot_tangent_distance():
    x = np.arange(scan + 1, scan + 2, 1 / n_int)
    a = np.sin((latitude_rad[:, 3, 4] - sc_lat_rad) / 2) ** 2 + \
        np.cos(latitude_rad[:, 3, 4]) * np.cos(sc_lat_rad) * np.sin((longitude_rad[:, 3, 4] - sc_lon_rad) / 2) \
        ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dist0 = R_Mars_km * c
    tangent_dist_ax.plot(x, dist0.flatten().T, color=color_dict['grey'])


def place_scan_image(axis, altitude_array, radiance, colormap, norm):
    for integration in range(n_int):
        for spatial_bin in range(n_spa):
            x_coordinates = [1 / 7 * (6 - spatial_bin) + scan, 1 / 7 * (6 - spatial_bin) + scan,
                             1 / 7 * (6 - spatial_bin + 1) + scan, 1 / 7 * (6 - spatial_bin + 1) + scan]
            y_coordinates = [altitude_array[integration, spatial_bin, 3], altitude_array[integration, spatial_bin, 2],
                             altitude_array[integration, spatial_bin, 0], altitude_array[integration, spatial_bin, 1]]
            vertices = list(zip(x_coordinates, y_coordinates))
            pixel_color = colormap(norm(radiance[integration, spatial_bin]))
            pixel = Polygon(vertices, closed=True, facecolor=pixel_color)
            axis.add_patch(pixel)


def outline_pixels(axes, flags):
    # make a new flag array for temporary storage
    temp_flag = np.zeros_like(flags)

    # calculate signal-to-noise ratios
    co_snr = co_cameron_radiance / co_cameron_uncertainty
    co2p_snr = co2p_uvd_radiance / co2p_uvd_uncertainty
    good_ind = np.squeeze(np.where(((co_snr > 4) | (co2p_snr > 3)) & (sza > 105) & (flags == 1)))
    if np.isscalar(good_ind[0]):
        good_int = np.array([good_ind[0]])
        good_spa = np.array([good_ind[1]])
    else:
        good_int = good_ind[0]
        good_spa = good_ind[1]

    for integration, spatial_bin in zip(good_int, good_spa):
        x_coords = [1 / 7 * (6 - spatial_bin) + scan, 1 / 7 * (6 - spatial_bin) + scan,
                    1 / 7 * (6 - spatial_bin + 1) + scan, 1 / 7 * (6 - spatial_bin + 1) + scan,
                    1 / 7 * (6 - spatial_bin) + scan]
        y_coords = [altitude[integration, spatial_bin, 3], altitude[integration, spatial_bin, 2],
                    altitude[integration, spatial_bin, 0], altitude[integration, spatial_bin, 1],
                    altitude[integration, spatial_bin, 3]]
        [axis.plot(x_coords, y_coords, linewidth=0.5, zorder=2, color=color_dict['grey']) for axis in axes]
        temp_flag[integration, spatial_bin] = 1

    # multiply the flag arrays
    flags *= temp_flag

    return flags


def calculate_periapse_distance(l1c_hdul, r):
    # find spacecraft position at periapse
    spacecraft_altitude = l1c_hdul['spacecraftgeometry'].data['spacecraft_alt']
    periapse_ind = np.unravel_index(spacecraft_altitude.argmin(), spacecraft_altitude.shape)
    latitude_1 = np.radians(hdull1c['spacecraftgeometry'].data['sub_spacecraft_lat'][periapse_ind])
    longitude_1 = np.radians(hdull1c['spacecraftgeometry'].data['sub_spacecraft_lon'][periapse_ind])

    # find coordinates of first scan center element which intersects 135 km altitude
    first_scan_altitude = l1c_hdul['pixelgeometry'].data['pixel_corner_mrh_alt'][0]
    ind135_0 = np.abs(first_scan_altitude[:, 3, 4] - 135).argmin()
    latitude_radians = rearrange_dimensions(np.radians(l1c_hdul['pixelgeometry'].data['pixel_corner_lat'][0]))
    longitude_radians = rearrange_dimensions(np.radians(l1c_hdul['pixelgeometry'].data['pixel_corner_lon'][0]))
    lat135_0 = latitude_radians[ind135_0, 3, 4]
    lon135_0 = longitude_radians[ind135_0, 3, 4]
    sc_latitude_rad = np.radians(hdull1c['spacecraftgeometry'].data['sub_spacecraft_lat'][0])
    sc_longitude_rad = np.radians(hdull1c['spacecraftgeometry'].data['sub_spacecraft_lon'][0])
    sclat135_0 = sc_latitude_rad[ind135_0]
    sclon135_0 = sc_longitude_rad[ind135_0]

    # calculate physical distance between spacecraft and first scan 135 km position calculated above
    a = abs(np.sin((lat135_0 - sclat135_0) / 2) ** 2 +
            np.cos(lat135_0) * np.cos(sclon135_0) * np.sin((lon135_0 - sclon135_0) / 2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dist = r * c

    # calculate coordinates of first integration, first bin corner projected into 135 km plane
    llos = dist  # length of line-of-sight [km]
    dlos = 1  # sample spacing along line-of-sight [km]
    n_los_samp = int(llos / dlos)  # number of samples along line-of-sight
    body_radii = spice.bodvrd('Mars', 'RADII', 3)[1]
    los_vec_arr = np.swapaxes(hdull1c['pixelgeometry'].data['pixel_vec'][:, :, :, :], 1, 4)
    sc_vec_arr = np.swapaxes(hdull1c['spacecraftgeometry'].data['v_spacecraft'], 1, 2)
    sc_vec = sc_vec_arr[0, 0]
    los_vec = los_vec_arr[0, 0, -1, 2]
    los_samp_vec_arr = np.matmul(np.linspace(0, llos, n_los_samp)[:, None], los_vec[None, :])
    los_samp_vec_arr += sc_vec
    npoint, samp_alt = spice.nearpt(los_samp_vec_arr[-1], body_radii[0], body_radii[1], body_radii[2])
    rpoint, colatpoint, lonpoint = spice.recsph(npoint)
    latitude_0 = np.pi / 2 - colatpoint
    longitude_0 = lonpoint
    if longitude_0 < 0:
        longitude_0 += 2 * np.pi

    # calculate distance between periapse and the first integration, first bin corner
    a = np.sin((latitude_1 - latitude_0) / 2) ** 2 + \
        np.cos(latitude_1) * np.cos(latitude_0) * np.sin((longitude_1 - longitude_0) / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    r_periapse = r * c

    return r_periapse, latitude_0, longitude_0


def place_scan_image_physical(axis, r_periapse, latitude_0, longitude_0, radiance, colormap, norm):
    # find where center element is closest to 135 km and get coordinates
    ind135 = np.abs(altitude[:, 3, 4] - 135).argmin()
    lat135 = latitude_rad[ind135, 3, 4]
    lon135 = longitude_rad[ind135, 3, 4]
    sclat135 = sc_lat_rad[ind135]
    sclon135 = sc_lon_rad[ind135]

    # calculate angular distance between points
    a = abs(np.sin((lat135 - sclat135) / 2) ** 2 +
            np.cos(lat135) * np.cos(sclon135) * np.sin((lon135 - sclon135) / 2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dist = R_Mars_km * c

    # line-of-sight parameters
    llos = dist  # length of line-of-sight [km]
    dlos = 1  # sample spacing along line-of-sight [km]
    n_los_samp = int(llos / dlos)  # number of samples along line-of-sight

    # SPICE parameters
    target = 'Mars'
    body_radii = spice.bodvrd(target, 'RADII', 3)[1]

    # get pixel and spacecraft vectors
    los_vec_arr = hdull1c['pixelgeometry'].data[scan]['pixel_vec'][:, :, :, :]
    sc_vec_arr = hdull1c['spacecraftgeometry'].data[scan]['v_spacecraft']

    # change dimensions to match (n_int, n_spa, 3)
    los_vec_arr = np.swapaxes(los_vec_arr, 0, 3)
    sc_vec_arr = np.swapaxes(sc_vec_arr, 0, 1)

    # arrays to hold newly calculated values
    new_altitude = np.zeros_like(altitude) * np.nan
    new_latitude_radians = np.zeros_like(latitude_rad) * np.nan
    new_longitude_radians = np.zeros_like(longitude_rad) * np.nan

    # loop through and calculate positions
    for integration in range(n_int):
        sc_vec = sc_vec_arr[integration]
        for spatial_bin in range(n_spa):
            for corner in range(5):
                los_vec = los_vec_arr[integration, spatial_bin, corner]
                los_samp_vec_arr = np.matmul(np.linspace(0, llos, n_los_samp)[:, None], los_vec[None, :])
                los_samp_vec_arr += sc_vec
                try:
                    npoint, samp_alt = spice.nearpt(los_samp_vec_arr[-1], body_radii[0], body_radii[1], body_radii[2])
                    rpoint, colatpoint, lonpoint = spice.recsph(npoint)
                    new_altitude[integration, spatial_bin, corner] = samp_alt
                    new_longitude_radians[integration, spatial_bin, corner] = lonpoint
                    new_latitude_radians[integration, spatial_bin, corner] = np.pi / 2 - colatpoint
                except IndexError:
                    continue

    # reset arrays to new calculations
    latitude_radians = new_latitude_radians
    longitude_radians = new_longitude_radians
    altitude_array = new_altitude

    # calculate distance from starting longitude and latitude points
    a = np.sin((latitude_radians - latitude_0) / 2) ** 2 + np.cos(latitude_radians) * np.cos(latitude_0) * np.sin(
        (longitude_radians - longitude_0) / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dist = R_Mars_km * c - r_periapse

    for integration in range(n_int):
        for spatial_bin in range(n_spa):
            x_coordinates = [dist[integration, spatial_bin, 3], dist[integration, spatial_bin, 2],
                             dist[integration, spatial_bin, 0], dist[integration, spatial_bin, 1]]
            y_coordinates = [altitude_array[integration, spatial_bin, 3], altitude_array[integration, spatial_bin, 2],
                             altitude_array[integration, spatial_bin, 0], altitude_array[integration, spatial_bin, 1]]
            vertices = list(zip(x_coordinates, y_coordinates))
            pixel_color = colormap(norm(radiance[integration, spatial_bin]))
            pixel = Polygon(vertices, closed=True, facecolor=pixel_color)
            axis.add_patch(pixel)

    return latitude_radians, longitude_radians, altitude_array


def plot_spacecraft_position(axis, spacecraft_latitude, spacecraft_longitude):
    spacecraft_longitude = spacecraft_longitude.flatten()
    spacecraft_latitude = spacecraft_latitude.flatten()
    if (len(np.where(spacecraft_longitude > 355)[0]) != 0) & (len(np.where(spacecraft_longitude < 5)[0]) != 0):
        xx1 = spacecraft_longitude
        xx1[np.where(xx1 < 180)] += 360
        map_axis.plot(xx1, spacecraft_latitude, color='white', linewidth=0.5)
        xx2 = spacecraft_longitude
        xx2[np.where(xx2 > 180)] -= 360
        axis.plot(xx2, spacecraft_latitude, color='white', linewidth=0.5)
    else:
        axis.plot(spacecraft_longitude, spacecraft_latitude, color='white', linewidth=0.5)


def place_detector_images(axes, plot_wavelength, plot_spectra):
    spec_cmap = plt.get_cmap('inferno')
    spec_cmap.set_bad((0, 0, 0))

    for integration in range(plot_spectra.shape[0]):
        x, y = np.meshgrid(np.linspace(plot_wavelength[4, 0], plot_wavelength[4, -1], plot_spectra.shape[2] + 1),
                           np.linspace(0, n_spa, plot_spectra.shape[1] + 1))
        axes[integration].pcolormesh(x, y, plot_spectra[integration, :, :], cmap=spec_cmap,
                                     norm=colors.LogNorm(vmin=50, vmax=5000), rasterized=True)
        axes[integration].set_frame_on(False)


def cosmic_ray_filtering(axes):
    # make data flag array to hold locations of good data and variable to indicate manual checking
    flags = np.ones((n_int, n_spa))
    check = 0

    # initial filtering
    for integration in range(n_int):
        for spatial_bin in range(n_spa):
            if (abs(const[integration, spatial_bin] > threshold_const) |
                    abs(chisq[integration, spatial_bin] > threshold_chisq) | (sza[integration, spatial_bin] < 102)):
                flags[integration, :] = 0

    # color the scans appropriately
    bad_ints = np.unique(np.where(flags == 0)[0])
    for integration in bad_ints:
        #axes[integration].axhspan(0, 7, color=color_dict['cyan'], linewidth=0, alpha=0.5)
        overlay = Rectangle(
                    (0, 0),
                    1,
                    1,
                    transform=axes[integration].transAxes,
                    facecolor="dodgerblue",
                    edgecolor="deepskyblue",
                    linewidth=1.5,
                    alpha=0.5,
                    zorder=10,
                    clip_on=False
                )
        axes[integration].add_patch(overlay)

    # return the flags and checks
    return flags, check


def cosmic_ray_filtering_fuv(axes):
    # make data flag array to hold locations of good data and variable to indicate manual checking
    flags = np.ones((n_int, n_spa))

    # initial filtering
    for integration in range(n_int):
        for spatial_bin in range(n_spa):
            if (abs(const[integration, spatial_bin] > threshold_const) |
                    abs(chisq[integration, spatial_bin] > threshold_chisq) | (sza[integration, spatial_bin] < 102)):
                flags[integration, :] = 0

    # color the scans appropriately
    bad_ints = np.unique(np.where(flags == 0)[0])
    for integration in bad_ints:
        axes[integration].axhspan(0, 7, color=color_dict['cyan'], linewidth=0, alpha=0.5)

    # return the flags and checks
    return flags


def positive_spectra(axes, flag_array, plot_color):
    # find passing pixels
    good_ind = np.squeeze(np.where(flag_array == 1))

    # show positive detections on detector images
    if np.isscalar(good_ind[0]):
        good_int = np.array([good_ind[0]])
        good_spa = np.array([good_ind[1]])
    else:
        good_int = good_ind[0]
        good_spa = good_ind[1]
    unique_ints = np.unique(good_int)
    for integration in unique_ints:
        axis = axes[integration]
        yticks = good_spa[np.where(good_int == integration)[0]] + 0.5
        axis.set_yticks(yticks)
        axis.set_yticklabels([])
        axis.text(1.01, 0.5, integration + 1, ha='left', va='center', fontsize=8, fontweight='bold', color=plot_color,
                  transform=axis.transAxes)


# function to calculate the MLR coefficient in DN space from the calibrated MLR value
def get_coeff(MLR, template_DN, dlambda, cal):
    return MLR / (np.sum(template_DN / cal) * dlambda)


def plot_average_spectrum(axis, flag_array):
    # find passing pixels
    good_ind = np.squeeze(np.where(flag_array == 1))
    if np.isscalar(good_ind[0]):
        good_int = np.array([good_ind[0]])
        good_spa = np.array([good_ind[1]])
    else:
        good_int = good_ind[0]
        good_spa = good_ind[1]

    # get DN coefficients
    solar_coeff = get_coeff(solar_radiance[good_int, good_spa], template_solar_continuum, dwavelength,
                            calibration_curve[3, :])
    co_coeff = get_coeff(co_cameron_radiance[good_int, good_spa], template_co_cameron, dwavelength,
                         calibration_curve[3, :])
    co2p_coeff = get_coeff(co2p_uvd_radiance[good_int, good_spa], template_co2p_uvd, dwavelength,
                           calibration_curve[3, :])
    no_coeff = get_coeff(no_nightglow_radiance[good_int, good_spa], template_no_nightglow, dwavelength,
                         calibration_curve[3, :])

    # make fitted spectra
    solar_component = template_solar_continuum * np.nanmean(solar_coeff)
    co_component = template_co_cameron * np.nanmean(co_coeff)
    co2p_component = template_co2p_uvd * np.nanmean(co2p_coeff)
    no_nightglow_component = template_no_nightglow * np.nanmean(no_coeff)
    constant = np.nanmean(const[good_int, good_spa])

    # calculate the wavelength
    plot_wavelength = np.nanmean(wavelength[good_spa], axis=0)

    # plot the background spectra
    background_spectrum = np.nanmean(spectra[good_int, good_spa],
                                     axis=0) - solar_component - no_nightglow_component - constant
    axis.plot(plot_wavelength, background_spectrum, color=color_dict['grey'], linewidth=0.5)

    # plot the aurora fit
    aurora_spectrum = co_component + co2p_component
    axis.plot(plot_wavelength, aurora_spectrum, color=color_dict['red'], linestyle='--', linewidth=0.5)

    # return the wavelength and spectra
    return plot_wavelength, background_spectrum, aurora_spectrum


def plot_average_spectrum_fuv(axis, flag_array):
    # find passing pixels
    good_ind = np.squeeze(np.where(flag_array == 1))
    if np.isscalar(good_ind[0]):
        good_int = np.array([good_ind[0]])
        good_spa = np.array([good_ind[1]])
    else:
        good_int = good_ind[0]
        good_spa = good_ind[1]

    # lists to hold components
    stored_wavelengths = []
    stored_spectra = []
    stored_templates = []

    # load MLR templates
    if n_spec < 256:
        lspec = 256
    else:
        lspec = n_spec
    fuv_templates = readsav('PyUVS/ancillary/peri_fuv_template_%i_bin_decal_v2.sav' % lspec)['peri_fuv_template'][0]

    # loop through components one at a time due to wavelength shift across bins
    for integration, spatial_bin in zip(good_int, good_spa):
        temp_wavelength = fuv_templates[2][:, :, :n_spec]
        absmin = np.abs(temp_wavelength[:, spatial_bin, 0] - wavelength[spatial_bin, 0])
        temp_wavelength = fuv_templates[2][absmin.argmin(), spatial_bin, :n_spec]
        template_o1304 = np.copy(fuv_templates[4][absmin.argmin(), spatial_bin, 0, :n_spec])
        template_o1304[np.where(np.isnan(template_o1304))] = 0
        template_o1356 = np.copy(fuv_templates[4][absmin.argmin(), spatial_bin, 1, :n_spec])
        template_o1356[np.where(np.isnan(template_o1356))] = 0
        dwave = np.diff(temp_wavelength)[0]

        # get calibration curve
        fuv_calibration_curve = calculate_calibration_curve(hdull1b)

        # get DN coefficients
        o1304_coeff = get_coeff(o1304_radiance[integration, spatial_bin] / 1000, template_o1304, dwave,
                                fuv_calibration_curve[3, :])
        o1356_coeff = get_coeff(o1356_radiance[integration, spatial_bin] / 1000, template_o1356, dwave,
                                fuv_calibration_curve[3, :])

        # make fitted spectra
        o1304_component = template_o1304 * np.nanmean(o1304_coeff)
        o1356_component = template_o1356 * np.nanmean(o1356_coeff)

        # store stuff
        stored_templates.append(o1304_component + o1356_component)
        stored_spectra.append(spectra[integration, spatial_bin] - const[integration, spatial_bin])
        stored_wavelengths.append(temp_wavelength)

    stored_templates = np.array(stored_templates)
    stored_spectra = np.array(stored_spectra)
    stored_wavelengths = np.array(stored_wavelengths)

    # interpolate the spectra to match the wavelengths of the first entry
    try:
        plot_wavelength = stored_wavelengths[0, :]
        interp_spec = [stored_spectra[0]]
        interp_template = [stored_templates[0]]
        for wave, spectrum, template in zip(stored_wavelengths[1:, :], stored_spectra[1:, :], stored_templates[1:, :]):
            interp_spec.append(np.interp(plot_wavelength, wave, spectrum))
            interp_template.append(np.interp(plot_wavelength, wave, template))

        interp_spec = np.nanmean(np.array(interp_spec), axis=0)
        interp_template = np.nanmean(np.array(interp_template), axis=0)

        # plot the background spectra
        axis.plot(plot_wavelength, interp_spec, color=color_dict['grey'], linewidth=0.5)

        # plot the aurora fit
        axis.plot(plot_wavelength, interp_template, color=color_dict['red'], linestyle='--', linewidth=0.5)

    except IndexError:
        plot_wavelength = None
        interp_spec = None
        interp_template = None

    # return the wavelength and spectra
    return plot_wavelength, interp_spec, interp_template


def calculate_correlation(axis, plot_wavelength, background_spectrum, aurora_spectrum):
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

    # draw the fit region lines
    axis.axvline(plot_wavelength[minwave_ind], color=color_dict['grey'], linestyle=':', linewidth=0.5)
    axis.axvline(plot_wavelength[maxwave_ind], color=color_dict['grey'], linestyle=':', linewidth=0.5)

    # return the goodness-of-fit metrics
    return r2, r


def calculate_correlation_fuv(axis, plot_wavelength, background_spectrum, aurora_spectrum):
    # calcualte quality of fit using only the bright Cameron band features
    minwave_ind = np.argmin(np.abs(plot_wavelength - (130.4 - 3)))
    maxwave_ind = np.argmin(np.abs(plot_wavelength - (135.6 + 3)))
    y_pred = aurora_spectrum[minwave_ind:maxwave_ind]
    y_true = background_spectrum[minwave_ind:maxwave_ind]
    try:
        r2 = metrics.r2_score(y_true, y_pred)
        r = pearsonr(y_true, y_pred)[0]
    except ValueError:
        r2 = np.nan
        r = np.nan

    # draw the fit region lines
    axis.axvline(plot_wavelength[minwave_ind], color=color_dict['grey'], linestyle=':', linewidth=0.5)
    axis.axvline(plot_wavelength[maxwave_ind], color=color_dict['grey'], linestyle=':', linewidth=0.5)

    # return the goodness-of-fit metrics
    return r2, r


def place_color_labels(axis, r2, r, flags):
    # determine color
    if np.isnan(r):
        plot_color = 'white'
    else:
        good_ind = np.squeeze(np.where(flags == 1))
        if np.isscalar(good_ind[0]):
            good_int = np.array([good_ind[0]])
            # good_spa = np.array([good_ind[1]])
        else:
            good_int = good_ind[0]
            # good_spa = good_ind[1]
        n = len(good_int)
        plot_color = color_dict['red']
        if (r >= 0.6) & (n <= 5) & (n >= 3):
            plot_color = color_dict['yellow']
        if ((r2 > 0.25) | (r > 0.55)) & (n > 5):
            plot_color = color_dict['green']

    # calculate counts, average radiances
    good_ind = np.squeeze(np.where(flags == 1))
    if np.isscalar(good_ind[0]):
        good_int = np.array([good_ind[0]])
        good_spa = np.array([good_ind[1]])
    else:
        good_int = good_ind[0]
        good_spa = good_ind[1]
    n = len(good_int)
    co_avg = float(np.nanmean(co_cameron_radiance[good_int, good_spa]))
    co2p_avg = float(np.nanmean(co2p_uvd_radiance[good_int, good_spa]))
    altitude_sigma = float(np.nanstd(altitude[good_int, good_spa, 4]))

    # place spectrum axis labels
    if ~np.isnan(r):
        axis.text(0.98, 0.95, '\n' + r'$n = %i$, R$^2$: %.2f, R = %.2f, $\sigma_{\mathrm{alt}} = %.2f$ km' %
                  (n, r2, r, altitude_sigma), ha='right', va='top', transform=axis.transAxes)
        axis.text(0.98, 0.95, '\n\n' + r'CO = %.2f kR, CO$_2^+$ = %.2f kR' % (co_avg, co2p_avg), ha='right', va='top',
                  transform=axis.transAxes)

    # place scan axis labels
    axis.text(0.98, 0.95, 'Scan %i' % (scan + 1), ha='right', va='top', color=plot_color, fontweight='bold',
              transform=axis.transAxes)
    scan_axis_co.text((0.5 + scan) / n_scans, 0.98, '%i' % (scan + 1), color=plot_color, ha='center', va='top',
                      fontsize=10, fontweight='bold', transform=scan_axis_co.transAxes)
    scan_axis_co2p.text((0.5 + scan) / n_scans, 0.98, '%i' % (scan + 1), color=plot_color, ha='center',
                        va='top', fontsize=10, fontweight='bold', transform=scan_axis_co2p.transAxes)
    scan_axis_solar.text((0.5 + scan) / n_scans, 0.98, '%i' % (scan + 1), color='white', ha='center',
                         va='top', fontsize=10, fontweight='bold', transform=scan_axis_solar.transAxes)
    detector_image_axes[scan, -1].set_title('Scan %i' % (scan + 1), fontweight='bold', color=plot_color)

    return plot_color, co_avg, co2p_avg


def place_color_labels_fuv(axis, r2, r, flags):
    # determine color
    if np.isnan(r):
        plot_color = 'white'
    else:
        good_ind = np.squeeze(np.where(flags == 1))
        if np.isscalar(good_ind[0]):
            good_int = np.array([good_ind[0]])
            # good_spa = np.array([good_ind[1]])
        else:
            good_int = good_ind[0]
            # good_spa = good_ind[1]
        n = len(good_int)
        plot_color = color_dict['red']
        if (r >= 0.6) & (n <= 5) & (n >= 3):
            plot_color = color_dict['yellow']
        if ((r2 > 0.25) | (r > 0.55)) & (n > 5):
            plot_color = color_dict['green']

    # calculate counts, average radiances
    good_ind = np.squeeze(np.where(flags == 1))
    if np.isscalar(good_ind[0]):
        good_int = np.array([good_ind[0]])
        good_spa = np.array([good_ind[1]])
    else:
        good_int = good_ind[0]
        good_spa = good_ind[1]
    n = len(good_int)
    o1304_avg = float(np.nanmean(o1304_radiance[good_int, good_spa]))
    o1356_avg = float(np.nanmean(o1356_radiance[good_int, good_spa]))
    altitude_sigma = float(np.nanstd(altitude[good_int, good_spa, 4]))

    # place spectrum axis labels
    if ~np.isnan(r):
        axis.text(0.98, 0.95, '\n' + r'$n = %i$, R = %.2f, $\sigma_{\mathrm{alt}} = %.2f$ km' %
                  (n, r, altitude_sigma), ha='right', va='top', transform=axis.transAxes)
        axis.text(0.98, 0.95, '\n\n' + r'O 130.4 = %.2f R, O 135.6 = %.2f R' % (o1304_avg, o1356_avg),
                  ha='right', va='top', transform=axis.transAxes)

    # place scan axis labels
    axis.text(0.98, 0.95, 'Scan %i' % (scan + 1), ha='right', va='top', color=plot_color, fontweight='bold',
              transform=axis.transAxes)
    scan_axis_co.text((0.5 + scan) / n_scans, 0.98, '%i' % (scan + 1), color=plot_color, ha='center', va='top',
                      fontsize=10, fontweight='bold', transform=scan_axis_co.transAxes)
    detector_image_axes[scan, -1].set_title('Scan %i' % (scan + 1), fontweight='bold', color=plot_color)

    return plot_color


def full_csv():
    good_ind = np.squeeze(np.where(flag == 1))
    if np.isscalar(good_ind[0]):
        good_int = np.array([good_ind[0]])
        good_spa = np.array([good_ind[1]])
    else:
        good_int = good_ind[0]
        good_spa = good_ind[1]
    if color == '#FDB813':
        color_flag = 'yellow'
    else:
        color_flag = 'green'
    n = len(good_int)

    # special in/out of CF region flag
    cf_region = np.zeros_like(latitude[good_int, good_spa, 4])
    cfind = np.where((longitude[good_int, good_spa, 4] >= 150) & (longitude[good_int, good_spa, 4] < 210) &
                     (latitude[good_int, good_spa, 4] >= -60) & (latitude[good_int, good_spa, 4] < -30))
    cf_region[cfind] = 1

    # new diffuse aurora
    discrete = 1
    if orbit_number in possible_diffuse:
        discrete = 0

    data_dict = {
        'orbit_number': np.ones(n, dtype=int) * orbit_number,
        'date': [datetime.strftime(datetime.strptime(utctime, '%Y/%j %b %d %H:%M:%S.%fUTC'), '%Y-%b-%d')
                 for utctime in hdull1c['integration'].data['utc'][scan, good_int]],
        'utc_time': [datetime.strftime(datetime.strptime(utctime, '%Y/%j %b %d %H:%M:%S.%fUTC'), '%H:%M:%S')
                     for utctime in hdull1c['integration'].data['utc'][scan, good_int]],
        'scan': np.ones(n, dtype=int) * scan + 1,
        'integration': good_int + 1,
        'nick_number': ['%.2f' % (orbit_number + (scan + 1) / 100)] * n,
        'spatial_bin': good_spa + 1,
        'latitude': latitude[good_int, good_spa, 4],
        'longitude': longitude[good_int, good_spa, 4],
        'intercept135_lat_near': near_lat[good_int, good_spa],
        'intercept135_lon_near': near_lon[good_int, good_spa],
        'intercept135_lat_far': far_lat[good_int, good_spa],
        'intercept135_lon_far': far_lon[good_int, good_spa],
        'cf_region': cf_region,
        'altitude': altitude[good_int, good_spa, 4],
        'local_time': local_time[good_int, good_spa],
        'et': et_reshaped[good_int, good_spa],
        'sza': sza[good_int, good_spa],
        'sc_latitude': sc_lat[good_int],
        'sc_longitude': sc_lon[good_int],
        'sc_altitude': sc_alt[good_int],
        'O1304_radiance': o1304_radiance[good_int, good_spa],
        'O1304_uncertainty': o1304_uncertainty[good_int, good_spa],
        'O1304_snr': o1304_radiance[good_int, good_spa] / o1304_uncertainty[good_int, good_spa],
        'O1356_radiance': o1356_radiance[good_int, good_spa],
        'O1356_uncertainty': o1356_uncertainty[good_int, good_spa],
        'O1356_snr': o1356_radiance[good_int, good_spa] / o1356_uncertainty[good_int, good_spa],
        'CO_radiance': co_cameron_radiance[good_int, good_spa] * co_cameron_scale,
        'CO_uncertainty': co_cameron_uncertainty[good_int, good_spa] * co_cameron_scale,
        'CO_snr': co_cameron_radiance[good_int, good_spa] / co_cameron_uncertainty[good_int, good_spa],
        'CO2p_radiance': co2p_uvd_radiance[good_int, good_spa],
        'CO2p_uncertainty': co2p_uvd_uncertainty[good_int, good_spa],
        'CO2p_snr': co2p_uvd_radiance[good_int, good_spa] / co2p_uvd_uncertainty[good_int, good_spa],
        'O2972_radiance': o2972_radiance[good_int, good_spa],
        'O2972_uncertainty': o2972_uncertainty[good_int, good_spa],
        'O2972_snr': o2972_radiance[good_int, good_spa] / o2972_uncertainty[good_int, good_spa],
        'CO/CO2p': co_cameron_radiance[good_int, good_spa] * co_cameron_scale / (co2p_uvd_radiance[good_int, good_spa]),
        'r_square_muv': np.ones(n) * r_square,
        'pearson_r_muv': np.ones(n) * pearson_r,
        'pearson_r_fuv': np.ones(n) * np.nan,
        'color_flag_muv': np.ones(n, dtype=object) * color_flag,
        'color_flag_fuv': np.ones(n, dtype=object) * 'red',
        'cosmic_ray_check_muv': np.ones(n, dtype=int) * manual_check,
        'discrete': np.ones(n, dtype=int) * discrete
    }
    if not os.path.exists('iuvs_statistical_detections_full_%s.csv' % version):
        pd.DataFrame.from_dict(data=data_dict, orient='columns').to_csv(
            'iuvs_statistical_detections_full_%s.csv' % version, header=True, index=False)
    else:
        pd.DataFrame.from_dict(data=data_dict, orient='columns').to_csv(
            'iuvs_statistical_detections_full_%s.csv' % version, mode='a', header=False, index=False)


def avg_unc(unc_arr, count):
    return np.sqrt(np.sum(unc_arr ** 2)) / count


def short_csv():
    good_ind = np.squeeze(np.where(flag == 1))
    if np.isscalar(good_ind[0]):
        good_int = np.array([good_ind[0]])
        good_spa = np.array([good_ind[1]])
    else:
        good_int = good_ind[0]
        good_spa = good_ind[1]
    if color == '#FDB813':
        color_flag = 'yellow'
    else:
        color_flag = 'green'

    # special longitude calculation
    lon_test = longitude[good_int, good_spa, 4]
    indbig = np.where(lon_test > 300)[0]
    indsmall = np.where(lon_test < 60)[0]
    if (len(indbig) != 0) & (len(indsmall) != 0):
        lon_test[np.where(lon_test < 180)] += 360
        tlon = np.nanmean(lon_test)
        if tlon >= 360:
            tlon -= 360
        if tlon < 0:
            tlon += 360
    else:
        tlon = np.nanmean(lon_test)

    # special SPICE calculation of average local time
    body = 499  # Mars IAU code
    et_avg = np.nanmean(et[good_int])
    lonpoint = np.radians(tlon)
    hr, mn, sc, _, _ = spice.et2lst(et_avg, body, lonpoint, 'planetocentric', timlen=256, ampmlen=256)

    n = len(good_int)
    median_int = int(np.median(good_int))

    # special in/out of CF region flag
    cf_region = 0
    if (tlon >= 150) & (tlon < 210) & (np.nanmean(latitude[good_int, good_spa, 4]) >= -60) & (
            np.nanmean(latitude[good_int, good_spa, 4]) < -30):
        cf_region = 1

    # new diffuse aurora
    discrete = 1
    if orbit_number in possible_diffuse:
        discrete = 0

    data_dict = {
        'orbit_number': [orbit_number],
        'date': [datetime.strftime(datetime.strptime(utctime, '%Y/%j %b %d %H:%M:%S.%fUTC'), '%Y-%b-%d')
                 for utctime in [hdull1c['integration'].data['utc'][scan, median_int]]],
        'utc_time': [datetime.strftime(datetime.strptime(utctime, '%Y/%j %b %d %H:%M:%S.%fUTC'), '%H:%M:%S')
                     for utctime in [hdull1c['integration'].data['utc'][scan, median_int]]],
        'scan': [scan + 1],
        'nick_number': ['%.2f' % (orbit_number + (scan + 1) / 100)],
        'latitude': [np.nanmean(latitude[good_int, good_spa, 4])],
        'longitude': [tlon],
        'cf_region': [cf_region],
        'altitude': [np.nanmean(altitude[good_int, good_spa, 4])],
        'altitude_std': [np.nanstd(altitude[good_int, good_spa, 4])],
        'spatial_bin_std': [np.nanstd(good_spa)],
        'local_time': [hr + mn / 60 + sc / 3600],
        'et': [np.nanmean(et_reshaped[good_int, good_spa])],
        'sza': [np.nanmean(sza[good_int, good_spa])],
        'sc_latitude': [np.nanmean(sc_lat[good_int])],
        'sc_longitude': [np.nanmean(sc_lon[good_int])],
        'sc_altitude': [np.nanmean(sc_alt[good_int])],
        'solar_longitude': [l_s],
        'O1304_radiance': [np.nanmean(o1304_radiance[good_int, good_spa])],
        'O1304_uncertainty': [avg_unc(o1304_uncertainty[good_int, good_spa], n)],
        'O1304_snr': [np.nanmean(o1304_radiance[good_int, good_spa]) /
                      avg_unc(o1304_uncertainty[good_int, good_spa], n)],
        'O1356_radiance': [np.nanmean(o1356_radiance[good_int, good_spa])],
        'O1356_uncertainty': [avg_unc(o1356_uncertainty[good_int, good_spa], n)],
        'O1356_snr': [np.nanmean(o1356_radiance[good_int, good_spa]) /
                      avg_unc(o1356_uncertainty[good_int, good_spa], n)],
        'CO_radiance': [np.nanmean(co_cameron_radiance[good_int, good_spa]) * co_cameron_scale],
        'CO_uncertainty': [avg_unc(co_cameron_uncertainty[good_int, good_spa], n) * co_cameron_scale],
        'CO_snr': [np.nanmean(co_cameron_radiance[good_int, good_spa]) /
                   avg_unc(co_cameron_uncertainty[good_int, good_spa], n)],
        'CO2p_radiance': [np.nanmean(co2p_uvd_radiance[good_int, good_spa])],
        'CO2p_uncertainty': [avg_unc(co2p_uvd_uncertainty[good_int, good_spa], n)],
        'CO2p_snr': [np.nanmean(co2p_uvd_radiance[good_int, good_spa]) /
                     avg_unc(co2p_uvd_uncertainty[good_int, good_spa], n)],
        'O2972_radiance': [np.nanmean(o2972_radiance[good_int, good_spa])],
        'O2972_uncertainty': [avg_unc(o2972_uncertainty[good_int, good_spa], n)],
        'O2972_snr': [np.nanmean(o2972_radiance[good_int, good_spa]) /
                      avg_unc(o2972_uncertainty[good_int, good_spa], n)],
        'CO/CO2p': [np.nanmean(co_cameron_radiance[good_int, good_spa]) * co_cameron_scale /
                    (np.nanmean(co2p_uvd_radiance[good_int, good_spa]))],
        'count': n,
        'r_square_muv': [r_square],
        'pearson_r_muv': [pearson_r],
        'pearson_r_fuv': [np.nan],
        'color_flag_muv': [color_flag],
        'color_flag_fuv': ['red'],
        'cosmic_ray_check_muv': [manual_check],
        'discrete': [discrete]
    }
    output_df = pd.DataFrame.from_dict(data=data_dict, orient='columns')
    if not block_csv_path.exists():
        output_df.to_csv(block_csv_path, header=True, index=False)
    else:
        output_df.to_csv(block_csv_path, mode='a', header=False, index=False)

def discrete_csv():
    good_ind = np.squeeze(np.where(flag == 1))
    if np.isscalar(good_ind[0]):
        good_int = np.array([good_ind[0]])
        good_spa = np.array([good_ind[1]])
    else:
        good_int = good_ind[0]
        good_spa = good_ind[1]
    if color == '#FDB813':
        color_flag = 'yellow'
    else:
        color_flag = 'green'
    n = len(good_int)

    # special in/out of CF region flag
    cf_region = np.zeros_like(latitude[good_int, good_spa, 4])
    cfind = np.where((longitude[good_int, good_spa, 4] >= 150) & (longitude[good_int, good_spa, 4] < 210) &
                     (latitude[good_int, good_spa, 4] >= -60) & (latitude[good_int, good_spa, 4] < -30))
    cf_region[cfind] = 1

    # new diffuse aurora
    discrete = 1
    if orbit_number in possible_diffuse:
        discrete = 0

    data_dict = {
        'orbit_number': np.ones(n, dtype=int) * orbit_number,
        'date': [datetime.strftime(datetime.strptime(utctime, '%Y/%j %b %d %H:%M:%S.%fUTC'), '%Y-%b-%d')
                 for utctime in hdull1c['integration'].data['utc'][scan, good_int]],
        'utc_time': [datetime.strftime(datetime.strptime(utctime, '%Y/%j %b %d %H:%M:%S.%fUTC'), '%H:%M:%S')
                     for utctime in hdull1c['integration'].data['utc'][scan, good_int]],
        'scan': np.ones(n, dtype=int) * scan + 1,
        'integration': good_int + 1,
        'nick_number': ['%.2f' % (orbit_number + (scan + 1) / 100)] * n,
        'spatial_bin': good_spa + 1,
        'latitude': latitude[good_int, good_spa, 4],
        'longitude': longitude[good_int, good_spa, 4],
        'intercept135_lat_near': near_lat[good_int, good_spa],
        'intercept135_lon_near': near_lon[good_int, good_spa],
        'intercept135_lat_far': far_lat[good_int, good_spa],
        'intercept135_lon_far': far_lon[good_int, good_spa],
        'cf_region': cf_region,
        'altitude': altitude[good_int, good_spa, 4],
        'local_time': local_time[good_int, good_spa],
        'et': et_reshaped[good_int, good_spa],
        'sza': sza[good_int, good_spa],
        'sc_latitude': sc_lat[good_int],
        'sc_longitude': sc_lon[good_int],
        'sc_altitude': sc_alt[good_int],
        'O1304_radiance': o1304_radiance[good_int, good_spa],
        'O1304_uncertainty': o1304_uncertainty[good_int, good_spa],
        'O1304_snr': o1304_radiance[good_int, good_spa] / o1304_uncertainty[good_int, good_spa],
        'O1356_radiance': o1356_radiance[good_int, good_spa],
        'O1356_uncertainty': o1356_uncertainty[good_int, good_spa],
        'O1356_snr': o1356_radiance[good_int, good_spa] / o1356_uncertainty[good_int, good_spa],
        'CO_radiance': co_cameron_radiance[good_int, good_spa] * co_cameron_scale,
        'CO_uncertainty': co_cameron_uncertainty[good_int, good_spa] * co_cameron_scale,
        'CO_snr': co_cameron_radiance[good_int, good_spa] / co_cameron_uncertainty[good_int, good_spa],
        'CO2p_radiance': co2p_uvd_radiance[good_int, good_spa],
        'CO2p_uncertainty': co2p_uvd_uncertainty[good_int, good_spa],
        'CO2p_snr': co2p_uvd_radiance[good_int, good_spa] / co2p_uvd_uncertainty[good_int, good_spa],
        'O2972_radiance': o2972_radiance[good_int, good_spa],
        'O2972_uncertainty': o2972_uncertainty[good_int, good_spa],
        'O2972_snr': o2972_radiance[good_int, good_spa] / o2972_uncertainty[good_int, good_spa],
        'CO/CO2p': co_cameron_radiance[good_int, good_spa] * co_cameron_scale / (co2p_uvd_radiance[good_int, good_spa]),
        'r_square_muv': np.ones(n) * r_square,
        'pearson_r_muv': np.ones(n) * pearson_r,
        'pearson_r_fuv': np.ones(n) * np.nan,
        'color_flag_muv': np.ones(n, dtype=object) * color_flag,
        'color_flag_fuv': np.ones(n, dtype=object) * 'red',
        'cosmic_ray_check_muv': np.ones(n, dtype=int) * manual_check,
        'discrete': np.ones(n, dtype=int) * discrete
    }
    if not os.path.exists('iuvs_statistical_detections_full_discrete_%s.csv' % version):
        pd.DataFrame.from_dict(data=data_dict, orient='columns').to_csv(
            'iuvs_statistical_detections_full_discrete_%s.csv' % version, header=True, index=False)
    else:
        pd.DataFrame.from_dict(data=data_dict, orient='columns').to_csv(
            'iuvs_statistical_detections_full_discrete_%s.csv' % version, mode='a', header=False, index=False)


def altitude_135(flags, color_string):
    # line-of-sight parameters
    llos = 3000  # length of line-of-sight [km]
    dlos = 1  # sample spacing along line-of-sight [km]
    n_los_samp = int(llos / dlos)  # number of samples along line-of-sight

    # SPICE parameters
    target = 'Mars'
    body_radii = spice.bodvrd(target, 'RADII', 3)[1]

    # get pixel and spacecraft vectors
    los_vec_arr = hdull1c['pixelgeometry'].data[scan]['pixel_vec'][:, :, 4, :]
    sc_vec_arr = hdull1c['spacecraftgeometry'].data[scan]['v_spacecraft']

    # change dimensions to match (n_spa, n_int, 3)
    los_vec_arr = np.swapaxes(los_vec_arr, 0, 2)
    sc_vec_arr = np.swapaxes(sc_vec_arr, 0, 1)

    # arrays to hold coordinates for both 135 km intercepts
    nearside_latitude = np.zeros((n_int, n_spa)) * np.nan
    nearside_longitude = np.zeros((n_int, n_spa)) * np.nan
    farside_latitude = np.zeros((n_int, n_spa)) * np.nan
    farside_longitude = np.zeros((n_int, n_spa)) * np.nan

    # index array
    indices = np.arange(n_los_samp).astype(int)
    #print(indices)
    # loop through and calculate positions
    for integration in range(n_int):
        sc_vec = sc_vec_arr[integration]
        for spatial_bin in range(n_spa):
            los_vec = los_vec_arr[integration, spatial_bin]
            los_samp_vec_arr = np.matmul(np.linspace(0, llos, n_los_samp)[:, None], los_vec[None, :])
            los_samp_vec_arr += sc_vec
            z_arr = np.zeros(n_los_samp) * np.nan
            for sample in range(n_los_samp):
                npoint, samp_alt = spice.nearpt(los_samp_vec_arr[sample], body_radii[0], body_radii[1], body_radii[2])
                z_arr[sample] = samp_alt
            lsalt = LineString(zip(indices, z_arr))
           # print(lsalt)
            ls135 = LineString([[0, 135], [n_los_samp, 135]])
            intersection = lsalt.intersection(ls135)
            if not intersection.is_empty:
                # Convert the Shapely result into a list of Point objects.
                if intersection.geom_type == "Point":
                    points = [intersection]

                elif intersection.geom_type == "MultiPoint":
                    points = list(intersection.geoms)

                elif intersection.geom_type == "GeometryCollection":
                    points = [geom for geom in intersection.geoms if geom.geom_type == "Point"]

                else:
                    # Unexpected intersection type, such as an overlapping LineString.
                    points = []

                # Put intersections in line-of-sight order.
                points = sorted(points, key=lambda point: point.x)

                def get_intercept_coordinates(point):
                    idx = int(np.round(point.x))
                    idx = np.clip(idx, 0, n_los_samp - 1)

                    npoint, _ = spice.nearpt(los_samp_vec_arr[idx], body_radii[0], body_radii[1], body_radii[2],)

                    _, colatpoint, lonpoint = spice.recsph(npoint)

                    if lonpoint < 0:
                        lonpoint += 2 * np.pi

                    latitude_deg = np.degrees(np.pi / 2 - colatpoint)
                    longitude_deg = np.degrees(lonpoint)

                    return latitude_deg, longitude_deg

                if len(points) >= 2:
                    # First crossing is the near-side intercept.
                    near_lat, near_lon = get_intercept_coordinates(points[0])
                    far_lat, far_lon = get_intercept_coordinates(points[-1])

                    nearside_latitude[integration, spatial_bin] = near_lat
                    nearside_longitude[integration, spatial_bin] = near_lon
                    farside_latitude[integration, spatial_bin] = far_lat
                    farside_longitude[integration, spatial_bin] = far_lon

                elif len(points) == 1:
                    # A single crossing generally means the LOS just reaches/crosses
                    # the 135-km surface once within the sampled distance.
                    far_lat, far_lon = get_intercept_coordinates(points[0])

                    farside_latitude[integration, spatial_bin] = far_lat
                    farside_longitude[integration, spatial_bin] = far_lon

    good_ind = np.squeeze(np.where(flags == 1))
    if np.isscalar(good_ind[0]):
        good_int = np.array([good_ind[0]])
        good_spa = np.array([good_ind[1]])
    else:
        good_int = good_ind[0]
        good_spa = good_ind[1]

    # scatterplot the points
    map_axis.scatter(sc_lon[good_int].flatten(), sc_lat[good_int].flatten(), s=2, edgecolors='none', color='white')
    map_axis.scatter(nearside_longitude[good_int, good_spa].flatten(), nearside_latitude[good_int, good_spa].flatten(),
                     s=1, edgecolors='none', color=color_dict['cyan'], zorder=3)
    map_axis.scatter(farside_longitude[good_int, good_spa].flatten(), farside_latitude[good_int, good_spa].flatten(),
                     s=1, edgecolors='none', color=color_dict['violet'], zorder=3)
    map_axis.scatter(longitude[good_int, good_spa].flatten(), latitude[good_int, good_spa].flatten(),
                     s=1, edgecolors='none', color=color_string, zorder=3)
    alt_dist_axis.scatter(np.array(good_int) / n_int + scan + 1, altitude[good_int, good_spa, 4],
                          color=color_string, s=4, zorder=3)

    # label the scan on the map axis
    longitude_test = np.copy(longitude)
    indbig = np.where(longitude_test > 300)[0]
    indsmall = np.where(longitude_test < 60)[0]
    if (len(indbig) != 0) & (len(indsmall) != 0):
        longitude_test[np.where(longitude_test < 180)] += 360
        obs_x = np.nanmean(longitude_test)
        if obs_x >= 360:
            obs_x -= 360
        if obs_x < 0:
            obs_x += 360
    else:
        obs_x = np.nanmean(longitude_test)
    sclon_test = np.copy(sc_lon)
    indbig = np.where(sclon_test > 300)[0]
    indsmall = np.where(sclon_test < 60)[0]
    if (len(indbig) != 0) & (len(indsmall) != 0):
        sclon_test[np.where(sclon_test < 180)] += 360
        spc_x = np.nanmean(sclon_test)
        if spc_x >= 360:
            spc_x -= 360
        if spc_x < 0:
            spc_x += 360
    else:
        spc_x = np.nanmean(sclon_test)
    obs_y = np.nanmean(latitude)
    spc_y = np.nanmean(sc_lat)
    if (spc_x < 60) & (obs_x > 300):
        obs_x0 = obs_x - 360
        dx = obs_x0 - spc_x
        dy = obs_y - spc_y
        theta = np.arctan2(dy, dx)
        label_x = spc_x + 6.5 * np.cos(theta + np.pi)
        label_y = spc_y + 6.5 * np.sin(theta + np.pi)
        map_axis.text(label_x, label_y, '%i' % (scan + 1), ha='center', va='center', clip_on=True)
        spc_x0 = spc_x + 360
        dx = obs_x - spc_x0
        dy = obs_y - spc_y
        theta = np.arctan2(dy, dx)
        label_x = spc_x + 6.5 * np.cos(theta + np.pi)
        label_y = spc_y + 6.5 * np.sin(theta + np.pi)
        map_axis.text(label_x, label_y, '%i' % (scan + 1), ha='center', va='center', clip_on=True)
    elif (obs_x < 60) & (spc_x > 300):
        obs_x0 = obs_x + 360
        dx = obs_x0 - spc_x
        dy = obs_y - spc_y
        theta = np.arctan2(dy, dx)
        label_x = spc_x + 6.5 * np.cos(theta + np.pi)
        label_y = spc_y + 6.5 * np.sin(theta + np.pi)
        map_axis.text(label_x, label_y, '%i' % (scan + 1), ha='center', va='center', clip_on=True)
        spc_x0 = spc_x - 360
        dx = obs_x - spc_x0
        dy = obs_y - spc_y
        theta = np.arctan2(dy, dx)
        label_x = spc_x + 6.5 * np.cos(theta + np.pi)
        label_y = spc_y + 6.5 * np.sin(theta + np.pi)
        map_axis.text(label_x, label_y, '%i' % (scan + 1), ha='center', va='center', clip_on=True)
    else:
        dx = obs_x - spc_x
        dy = obs_y - spc_y
        theta = np.arctan2(dy, dx)
        label_x = spc_x + 6.5 * np.cos(theta + np.pi)
        label_y = spc_y + 6.5 * np.sin(theta + np.pi)
        map_axis.text(label_x, label_y, '%i' % (scan + 1), ha='center', va='center', clip_on=True)

    # return the latitude and longitude arrays
    return nearside_latitude, nearside_longitude, farside_latitude, farside_longitude


def haversine(latitude0, longitude0, lat1, lon1):
    angles = 2 * np.arcsin(np.sqrt(np.sin((latitude0 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(latitude0) *
                                   np.sin((longitude0 - lon1) / 2) ** 2))
    return angles


def calc_sza(subsolar_latitude, subsolar_longitude):
    subsolar_latitude = np.radians(subsolar_latitude)
    subsolar_longitude = np.radians(subsolar_longitude)
    longitudes, latitudes = np.meshgrid(np.linspace(np.radians(0), np.radians(360), 3600),
                                        np.linspace(np.radians(-90), np.radians(90), 1800))
    solar_zenith_angles = haversine(subsolar_latitude, subsolar_longitude, latitudes, longitudes)
    return np.degrees(longitudes), np.degrees(latitudes), np.degrees(solar_zenith_angles)


def point_distance(latitude0, longitude0, lat_arr, lon_arr):
    aa = abs(np.sin((lat_arr - latitude0) / 2) ** 2 +
             np.cos(lat_arr) * np.cos(longitude0) * np.sin((lon_arr - longitude0) / 2) ** 2)
    dd = 2 * np.arctan2(np.sqrt(aa), np.sqrt(1 - aa))
    ddist = R_Mars_km * dd
    return ddist


# set version number
version = '1'

# filter warnings, there are a lot and they are annoying
filterwarnings('ignore')

# get starting time
t0 = time.time()

# set plot formatting and normalizations
JGR_format(display_widths=False, return_blue=False)
plt.style.use('dark_background')
norm_co = colors.PowerNorm(gamma=1 / 2, vmin=0, vmax=4)
norm_co2p = colors.PowerNorm(gamma=1 / 2, vmin=0, vmax=0.5)
norm_solar = colors.PowerNorm(gamma=1 / 2, vmin=0, vmax=10)
norm_o1304 = colors.PowerNorm(gamma=1 / 2, vmin=0, vmax=125)
norm_o1356 = colors.PowerNorm(gamma=1 / 2, vmin=0, vmax=125)

# set FUV colormaps
o1304_cmap = H_colormap()
o1356_cmap = H_colormap()

# load SPICE kernels
load_iuvs_spice(spice_directory='D:/periapse_quicklooks/spice')
print('SPICE kernels loaded')

# remove existing files
#os.system('rm *%s.csv' % version)
#os.system('rm quicklooks/*')

# list of diffuse aurora orbits
december2014 = np.arange(382, 453 + 1, 1)
march2015 = [806, 810, 824, 826, 842, 844, 846, 848, 850, 852]
september2017 = np.arange(5700, 5800 + 1, 1)
diffuse_aurora = np.concatenate((december2014, march2015, september2017))

# new set of possible diffuse aurora
possible_diffuse = np.concatenate((np.arange(2508, 2517 + 1), np.arange(3338, 3546 + 1), np.arange(4556, 4567 + 1),
                                   [6753, 4736, 3828]))

# Process the requested range one orbit block at a time.
for block_start in range(RUN_START, RUN_END, BLOCK_SIZE):
    block_end = min(block_start + BLOCK_SIZE, RUN_END)

    block_start_str = format_orbit(block_start)
    block_end_str = format_orbit(block_end)

    print()
    print("=" * 70)
    print(f"Processing orbit block {block_start}-{block_end - 1}")
    print("=" * 70)

    block_csv_path = (
        TABLE_OUTPUT_DIR
        / f"{block_start_str}_{block_end_str}_detections_short_{version}.csv"
    )

    # Remove old output so detections from previous runs aren't retained
    if block_csv_path.exists():
        block_csv_path.unlink()


    block_quicklook_dir = QUICKLOOK_OUTPUT_DIR / block_start_str
    block_spectra_dir = SPECTRA_OUTPUT_DIR / block_start_str
    block_quicklook_dir.mkdir(parents=True, exist_ok=True)
    block_spectra_dir.mkdir(parents=True, exist_ok=True)

    orbit_numbers = np.arange(block_start, block_end)
    block_t0 = time.time()

    for i, orbit_number in enumerate(orbit_numbers):
        progress_report(block_t0, i, orbit_number, len(orbit_numbers))
        # list to hold flags
        master_flags_list = []

        # ================================ #
        # === MUV Quicklook/Detections === #
        # ================================ #

        # get files
        l1cfiles, l1bfiles = get_fits(orbit_number, 'muv')
        if len(l1cfiles) == 0:
            continue
        if len(l1bfiles) ==0:
            continue

        # open FITS file
        print('Good orbit found')
        hdull1c = fits.open(l1cfiles[-1], ignore_missing_simple=True, ignore_missing_end=True)


        # it no pixels above SZA threshold, continue
        sza = hdull1c['pixelgeometry'].data['pixel_solar_zenith_angle']
        if np.nanmax(sza) < 112:
            print('No pixels above SZA threshold for orbit %i' % orbit_number)
            continue

        # variable to hold whether or not any pixels pass
        any_pixels = False

        # determine number of scans and integrations
        n_scans, n_int, n_spa, n_spec = get_dims(hdull1c)

        if len(l1bfiles) < n_scans:
                    print(
                        f"Skipping orbit {orbit_number}: "
                        f"expected {n_scans} L1B files, found {len(l1bfiles)}"
                    )
                    hdull1c.close()
                    continue

        # calculate number of rows for detector images (6 per row)
        det_rows = int(np.ceil(n_scans / 6))

        # get time of periapsis and convert to string
        altitude = hdull1c['pixelgeometry'].data['pixel_corner_mrh_alt'][:, 3, 4, :]
        peri_ind = np.where(altitude == np.nanmin(altitude))
        utc = hdull1c['integration'].data['utc'][peri_ind][0]
        date = datetime.strptime(utc, '%Y/%j %b %d %H:%M:%S.%fUTC')
        date_string = datetime.strftime(date, '%Y-%b-%d, %H:%M:%S UTC')
        sub_solar_lat = hdull1c['spacecraftgeometry'].data['sub_solar_lat'][peri_ind][0]
        sub_solar_lon = hdull1c['spacecraftgeometry'].data['sub_solar_lon'][peri_ind][0]

        # make a figure
        fig_width = 20.25
        det_offset = (0.125 * n_int + 0.5) * det_rows + 0.25
        fig_height = 7.75 + det_offset
        fig = plt.figure(figsize=(fig_width, fig_height))

        # make scan axis for CO Cameron Bands
        scan_axis_co = plt.axes([7.5 / fig_width, (0.5 + det_offset + 2.375 * 2) / fig_height, 5.5 / fig_width,
                                 2.125 / fig_height])
        scan_axis_co.set_title('Orbit %i, %s' % (orbit_number, date_string), fontsize=15, fontweight='bold')
        scan_axis_co.set_ylabel('Altitude [km]')
        scan_axis_co.set_xticks(np.arange(0.5, n_scans, 1))
        scan_axis_co.set_xticklabels([])
        scan_axis_co.set_xlim(0, n_scans)
        scan_axis_co.yaxis.set_minor_locator(ticker.MultipleLocator(5))
        [scan_axis_co.axvline(i, color=color_dict['darkgrey'], zorder=0, linewidth=0.5) for i in range(n_scans)]
        cax_co = plt.axes(
            [(6.875 + 6.25) / fig_width, (0.5 + det_offset + 2.375 * 2) / fig_height, 0.125 / fig_width,
             2.125 / fig_height])
        co_sm = plt.cm.ScalarMappable(cmap=CO_colormap(), norm=norm_co)
        co_sm.set_array(np.array([]))
        cbar_co = plt.colorbar(co_sm, cax=cax_co, label='CO Cameron Band Radiance [kR]')
        cbar_co.ax.yaxis.set_ticks(co_sm.norm(np.linspace(0, 4, 5)))
        cbar_co.ax.yaxis.set_ticklabels(np.linspace(0, 4, 5))
        cbar_co.ax.yaxis.set_ticks(co_sm.norm(np.linspace(0, 4, 17)), minor=True)

        # make scan axis for CO2+ UVD
        scan_axis_co2p = plt.axes([7.5 / fig_width, (0.5 + det_offset + 2.375) / fig_height, 5.5 / fig_width,
                                   2.125 / fig_height])
        scan_axis_co2p.set_ylabel('Altitude [km]')
        scan_axis_co2p.set_xticks(np.arange(0.5, n_scans, 1))
        scan_axis_co2p.set_xticklabels([])
        scan_axis_co2p.set_xlim(0, n_scans)
        scan_axis_co2p.yaxis.set_minor_locator(ticker.MultipleLocator(5))
        [scan_axis_co2p.axvline(i, color=color_dict['darkgrey'], zorder=0, linewidth=0.5) for i in range(n_scans)]
        cax_co2p = plt.axes(
            [(6.875 + 6.25) / fig_width, (0.5 + det_offset + 2.375) / fig_height, 0.125 / fig_width,
             2.125 / fig_height])
        co2p_sm = plt.cm.ScalarMappable(cmap=CO2p_colormap(), norm=norm_co2p)
        co2p_sm.set_array(np.array([]))
        cbar_co2p = plt.colorbar(co2p_sm, cax=cax_co2p, label=r'CO$_2^+$ UVD Radiance [kR]')
        cbar_co2p.ax.yaxis.set_ticks(co2p_sm.norm(np.linspace(0, 0.5, 5)))
        cbar_co2p.ax.yaxis.set_ticklabels(np.linspace(0, 0.5, 5))
        cbar_co2p.ax.yaxis.set_ticks(co2p_sm.norm(np.arange(0, 0.5, 0.125 / 4)), minor=True)

        # make scan axis for NO Nightglow
        scan_axis_solar = plt.axes([7.5 / fig_width,
                                    (0.5 + det_offset) / fig_height, 5.5 / fig_width, 2.125 / fig_height])
        scan_axis_solar.set_xlabel(r'Average Scan Solar Zenith Angle [$\degree$]')
        scan_axis_solar.set_xticks(np.arange(0.5, n_scans, 1))
        sza = hdull1c['pixelgeometry'].data['pixel_solar_zenith_angle']
        xlabels = ['%i' % i for i in np.nanmean(np.nanmean(sza, axis=1), axis=1)]
        scan_axis_solar.set_xticklabels(xlabels)
        scan_axis_solar.set_xlim(0, n_scans)
        scan_axis_solar.set_ylabel('Altitude [km]')
        scan_axis_solar.yaxis.set_minor_locator(ticker.MultipleLocator(5))
        [scan_axis_solar.axvline(i, color=color_dict['darkgrey'], zorder=0, linewidth=0.5) for i in range(n_scans)]
        cax_solar = plt.axes(
            [(6.875 + 6.25) / fig_width, (0.5 + det_offset) / fig_height, 0.125 / fig_width, 2.125 / fig_height])
        solar_sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('magma'), norm=norm_solar)
        solar_sm.set_array(np.array([]))
        cbar_solar = plt.colorbar(solar_sm, cax=cax_solar, label='Solar Continuum Radiance [kR]')
        cbar_solar.ax.yaxis.set_ticks(solar_sm.norm(np.linspace(0, 10, 6)))
        cbar_solar.ax.yaxis.set_ticklabels(np.linspace(0, 10, 6).astype(int))
        cbar_solar.ax.yaxis.set_ticks(solar_sm.norm(np.linspace(0, 10, 41)), minor=True)

        # make physical axis
        physical_axis = plt.axes(
            [0.5 / fig_width, (4.125 + det_offset) / fig_height, 6.25 / fig_width, 0.375 / fig_height])
        physical_axis.set_facecolor((0.25, 0.25, 0.25))
        physical_axis.set_xlabel('Tangent Surface Distance Relative to Periapsis [km]')
        physical_axis.xaxis.set_major_locator(ticker.MultipleLocator(base=1000))
        physical_axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=100))
        physical_axis.set_yticks([])
        physical_axis.set_ylabel('Alt.')

        # make map axis
        map_axis = plt.axes([0.5 / fig_width, (0.5 + det_offset) / fig_height, 6.25 / fig_width, 3.125 / fig_height])
        map_img = np.load('PyUVS/ancillary/magnetic_field_map.npy', allow_pickle=True).item()
        # noinspection PyUnresolvedReferences
        map_lat = map_img.get('latitude')
        # noinspection PyUnresolvedReferences
        map_lon = map_img.get('longitude')
        # noinspection PyUnresolvedReferences
        map_field = map_img.get('prob_open_field')
        V = map_field.copy()
        V[np.isnan(map_field)] = 0
        VV = gaussian_filter(V, sigma=1)
        W = 0 * map_field.copy() + 1
        W[np.isnan(map_field)] = 0
        WW = gaussian_filter(W, sigma=1)
        map_field = VV / WW
        map_field /= np.nanmax(map_field)
        mag_cmap = plt.get_cmap('Blues')
        mag_cmap.set_bad((1, 1, 1))
        map_axis.imshow(np.log10(map_field), cmap=mag_cmap, alpha=0.5, origin='lower', extent=[0, 360, -90, 90])

        # draw terminator
        terminator_lon, terminator_lat, terminator = calc_sza(sub_solar_lat, sub_solar_lon)
        cf = map_axis.contourf(terminator_lon, terminator_lat, terminator, levels=[0, 90],
                               colors=color_dict['darkgrey'], alpha=0.33)
        cs = map_axis.contour(terminator_lon, terminator_lat, terminator, levels=[90, 102], colors=color_dict['grey'])
        map_axis.clabel(cs, inline=1, fontsize=8, fmt=r'$%i\degree$')

        # place map axis legend
        legend_elements = [Line2D([0], [0], marker='.', markersize=10, markeredgecolor='none', linewidth=0,
                                  color=color_dict['green'], label='IUVS Confirmed Detection'),
                           Line2D([0], [0], marker='.', markersize=10, markeredgecolor='none', linewidth=0,
                                  color=color_dict['yellow'], label='IUVS Marginal Detection'),
                           Line2D([0], [0], marker='.', markersize=10, markeredgecolor='none', linewidth=0.5,
                                  color='white', label='MAVEN Position'),
                           Line2D([0], [0], marker='.', markersize=10, linewidth=0, markeredgecolor='none',
                                  color=color_dict['cyan'], label='Nearside 135 km Altitude Intercept'),
                           Line2D([0], [0], marker='.', markersize=10, linewidth=0, markeredgecolor='none',
                                  color=color_dict['violet'], label='Farside 135 km Altitude Intercept'),
                           ]
        leg_map = map_axis.legend(handles=legend_elements, edgecolor='white', fancybox=False, framealpha=1)
        leg_map.get_frame().set_linewidth(0.5)

        # make s/c altitude axis
        alt_dist_axis = plt.axes([0.5 / fig_width, (0.5 + det_offset + 4.125 + 0.375) / fig_height, 5.75 / fig_width,
                                  2.375 / fig_height])
        alt_dist_axis.xaxis.set_major_locator(ticker.MultipleLocator(base=1))
        alt_dist_axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=1 / n_int))
        alt_dist_axis.yaxis.set_major_locator(ticker.MultipleLocator(base=50))
        alt_dist_axis.yaxis.set_minor_locator(ticker.MultipleLocator(base=10))
        alt_dist_axis.set_xlabel('Scan, Integration')
        alt_dist_axis.set_xticks(np.arange(1, n_scans + 1, 1))
        alt_dist_axis.set_xticklabels(np.arange(1, n_scans + 1, 1))
        alt_dist_axis.set_xlim(1, n_scans + 1)
        alt_dist_axis.set_ylabel('Altitude [km]')
        alt_dist_axis.axhline(135, linestyle=':', color=color_dict['grey'])

        # plot spacecraft altitude
        sc_alt = hdull1c['spacecraftgeometry'].data['spacecraft_alt']
        plot_spacecraft_altitude()

        # make tangent distance axis
        tangent_dist_ax = alt_dist_axis.twinx()
        tangent_dist_ax.set_ylabel('Tangent Point Distance [km]')
        tangent_dist_ax.yaxis.set_major_locator(ticker.MultipleLocator(base=250))
        tangent_dist_ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=25))

        # place altitude/tangent distance axis legend
        legend_elements = [Line2D([0], [0], marker='.', markersize=4, linewidth=0.5,
                                  color='white', label='MAVEN Spacecraft Altitude'),
                           Line2D([0], [0], marker='.', markersize=4, linewidth=0,
                                  color=color_dict['green'], label='Statitical Detections Altitude'),
                           Line2D([0], [0], linewidth=0.5, color=color_dict['grey'], label='Tangent Point Distance'),
                           Line2D([0], [0], linewidth=0.5, color=color_dict['red'], linestyle='--', label='Periapsis'),
                           Line2D([0], [0], linewidth=0.5, color=color_dict['grey'], linestyle=':',
                                  label='135 km Altitude')
                           ]
        leg_alt = tangent_dist_ax.legend(handles=legend_elements, ncol=2, edgecolor='white', fancybox=False,
                                         framealpha=1)
        leg_alt.get_frame().set_linewidth(0.5)

        # make spectrum axes
        n_rows = int(np.ceil(n_scans / 2))
        spectrum_axes = [[], []]
        for row in range(n_rows):
            for col in range(2):
                ax = plt.axes(
                    [(14 + 2.925 * col) / fig_width, (0.5 + row * (6.875 / n_rows) + det_offset) / fig_height,
                     2.75 / fig_width, 6.875 / n_rows / fig_height])
                ax.set_xticks([])
                ax.set_yticks([])
                spectrum_axes[col].append(ax)
        spectrum_axes = np.transpose(spectrum_axes)

        # make detector image axes
        peaks = [193, 199.5, 206, 216, 228, 240, 289]
        detector_image_axes = []
        for row in range(det_rows):
            for col in range(6):
                for inte in range(n_int):
                    x0 = (0.5 + ((fig_width - 0.5) / 6) * col)
                    y0 = 0.5 * (row + 1) + 0.125 * n_int * row + 0.125 * inte
                    xw = 19.25 / 6 - 0.5
                    yw = 0.1
                    ax = plt.axes([x0 / fig_width, y0 / fig_height, xw / fig_width, yw / fig_height])
                    ax.tick_params(which='minor', axis='x', bottom=True, top=True, labeltop=False, labelbottom=False)
                    if inte == 0:
                        ax.set_xticks(peaks)
                        ax.tick_params(which='major', axis='x', bottom=True, top=False, labeltop=False,
                                       labelbottom=False)
                    elif inte == n_int - 1:
                        ax.set_xticks(peaks)
                        ax.tick_params(which='major', axis='x', bottom=False, top=True, labeltop=False,
                                       labelbottom=False)
                    else:
                        ax.set_xticks([])
                        ax.set_xticks(peaks, minor=True)
                    ax.set_yticks([])
                    detector_image_axes.append(ax)
        detector_image_axes = np.reshape(detector_image_axes, (int(np.ceil(n_scans / 6) * 6), n_int))

        # get feature indices
        try:
            solar_ind, co_ind, co2puvd_ind, o2972_ind, no_nightglow_ind, o1304_ind, o1356_ind = get_feature_indices(hdull1c)
        except IndexError:
            print('broken')
            print(get_feature_indices(hdull1c))

        # calculate periapse distance and starting coordinates
        periapse_distance, lat0, lon0 = calculate_periapse_distance(hdull1c, R_Mars_km)

        # calculate constant and chi-square thresholds for this orbit
        const = hdull1c['radiance_noaltbin'].data['baseline_muv']
        chisq = hdull1c['radiance_noaltbin'].data['chi_square_muv']
        threshold_const = np.nanmean(const) + np.nanstd(const)
        threshold_chisq = np.nanmedian(chisq) + 2 * np.nanstd(chisq)

        # load MLR templates
        if n_spec == 219:
            load_spec = 256
        else:
            load_spec = n_spec
        templates = np.load('PyUVS/ancillary/templates_%i.npy' % load_spec, allow_pickle=True)
        # noinspection PyUnresolvedReferences
        template_wavelength = templates.item().get('wavelength')
        dwavelength = np.diff(template_wavelength)[0]
        # noinspection PyUnresolvedReferences
        template_solar_continuum = templates.item().get('solar_continuum')
        # noinspection PyUnresolvedReferences
        template_co_cameron = templates.item().get('co_cameron')
        # noinspection PyUnresolvedReferences
        template_co2p_uvd = templates.item().get('co2p_uvd')
        # noinspection PyUnresolvedReferences
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

        collected_spectra=[] # list to hold collected spectra for all scans

        # loop through scans
        for scan in range(n_scans):

            # load level 1B file
            hdull1b = fits.open(l1bfiles[scan], ignore_missing_simple=True, ignore_missing_end=True)
            #try:
                #hdull1b = fits.open(l1bfiles[scan], ignore_missing_simple=True)
            #except OSError:
                #continue

            # get the radiances
            solar_radiance = hdull1c['radiance_noaltbin'].data['profile'][scan, :, :, solar_ind]
            co_cameron_radiance = hdull1c['radiance_noaltbin'].data['profile'][scan, :, :, co_ind]
            co2p_uvd_radiance = hdull1c['radiance_noaltbin'].data['profile'][scan, :, :, co2puvd_ind]
            o2972_radiance = hdull1c['radiance_noaltbin'].data['profile'][scan, :, :, o2972_ind]
            o1304_radiance = hdull1c['radiance_noaltbin'].data['profile'][scan, :, :, o1304_ind]
            o1356_radiance = hdull1c['radiance_noaltbin'].data['profile'][scan, :, :, o1356_ind]
            no_nightglow_radiance = hdull1c['radiance_noaltbin'].data['profile'][scan, :, :, no_nightglow_ind]

            # get the radiance uncertainties
            solar_uncertainty = hdull1c['radiance_noaltbin'].data['random_unc'][scan, :, :, solar_ind]
            co_cameron_uncertainty = hdull1c['radiance_noaltbin'].data['random_unc'][scan, :, :, co_ind]
            co2p_uvd_uncertainty = hdull1c['radiance_noaltbin'].data['random_unc'][scan, :, :, co2puvd_ind]
            o2972_uncertainty = hdull1c['radiance_noaltbin'].data['random_unc'][scan, :, :, o2972_ind]
            o1304_uncertainty = hdull1c['radiance_noaltbin'].data['random_unc'][scan, :, :, o1304_ind]
            o1356_uncertainty = hdull1c['radiance_noaltbin'].data['random_unc'][scan, :, :, o1356_ind]
            no_nightglow_uncertainty = hdull1c['radiance_noaltbin'].data['random_unc'][scan, :, :, no_nightglow_ind]

            # get fit metrics
            const = hdull1c['radiance_noaltbin'].data['baseline_muv'][scan]
            chisq = hdull1c['radiance_noaltbin'].data['chi_square_muv'][scan]

            # get observation geometry
            latitude = rearrange_dimensions(hdull1c['pixelgeometry'].data['pixel_corner_lat'][scan])
            latitude_rad = np.radians(latitude)
            longitude = rearrange_dimensions(hdull1c['pixelgeometry'].data['pixel_corner_lon'][scan])
            longitude_rad = np.radians(longitude)
            altitude = rearrange_dimensions(hdull1c['pixelgeometry'].data['pixel_corner_mrh_alt'][scan])
            altitude_center = rearrange_dimensions(hdull1c['pixelgeometry'].data['pixel_corner_mrh_alt'][scan])[:, 3, 4]
            sza = hdull1c['pixelgeometry'].data['pixel_solar_zenith_angle'][scan].T
            local_time = hdull1c['pixelgeometry'].data['local_time'][scan].T
            et = hdull1c['integration'].data['et'][scan]
            et_reshaped = np.repeat(et[:, None], n_spa, axis=1)

            # get spacecraft geometry
            sc_lat = hdull1c['spacecraftgeometry'].data['sub_spacecraft_lat'][scan]
            sc_lat_rad = np.radians(sc_lat)
            sc_lon = hdull1c['spacecraftgeometry'].data['sub_spacecraft_lon'][scan]
            sc_lon_rad = np.radians(sc_lon)
            sc_alt = hdull1c['spacecraftgeometry'].data['spacecraft_alt'][scan]

            # get solar longitude
            l_s=hdull1c['observation'].data['SOLAR_LONGITUDE'][0]

            # get spectra in DN
            wavelength = np.squeeze(hdull1b['observation'].data['wavelength'])
            spectra = hdull1b['detector_dark_subtracted'].data

            # get calibration curve
            calibration_curve = calculate_calibration_curve(hdull1b)

            # place the level 1C Cameron band data in the physical axis
            latitude_wall, longitude_wall, altitude_wall = place_scan_image_physical(physical_axis, periapse_distance,
                                                                                     lat0, lon0, co_cameron_radiance,
                                                                                     CO_colormap(), norm=norm_co)
            latitude_wall = np.degrees(latitude_wall)
            longitude_wall = np.degrees(longitude_wall)
            longitude_wall[np.where(longitude_wall < 0)] += 360

            # place the level 1C data in the scan axes
            place_scan_image(scan_axis_co, altitude, co_cameron_radiance, CO_colormap(), norm=norm_co)
            place_scan_image(scan_axis_co2p, altitude, co2p_uvd_radiance, CO2p_colormap(), norm=norm_co2p)
            place_scan_image(scan_axis_solar, altitude, solar_radiance, plt.get_cmap('magma'), norm=norm_solar)

            # plot spacecraft position
            plot_spacecraft_position(map_axis, sc_lat, sc_lon)

            # filter out cosmic rays
            flag, manual_check = cosmic_ray_filtering(detector_image_axes[scan])

            # outline pixels which meet signal-to-noise criteria
            flag = outline_pixels([scan_axis_co, scan_axis_co2p], flag)

            # store flags
            master_flags_list.append(flag)

            # place detector images
            place_detector_images(detector_image_axes[scan], wavelength, spectra)

            # plot average spectrum without solar and NO components
            ind1, ind0 = divmod(scan, n_rows)
            avg_wavelength, avg_background, avg_aurora = plot_average_spectrum(spectrum_axes[ind0, ind1], flag)

            # calculate goodness-of-fit
            r_square, pearson_r = calculate_correlation(spectrum_axes[ind0, ind1], avg_wavelength, avg_background,
                                                        avg_aurora)
            
            # place labels
            color, coavg, co2avg = place_color_labels(spectrum_axes[ind0, ind1], r_square, pearson_r, flag)

            if color == '#2CA02C' or color == '#FDB813':
                detection_flag = 1  # confirmed detection (marginal included)
            else:
                detection_flag = 0  # not a detection / unclassified

            # Get good-pixel indices
            good_ind = np.squeeze(np.where(flag == 1))
            if np.isscalar(good_ind[0]):
                good_int = np.array([good_ind[0]])
                good_spa = np.array([good_ind[1]])
            else:
                good_int = good_ind[0]
                good_spa = good_ind[1]

            # Compute representative values
            if len(good_int) > 0:
                sza_mean = float(np.nanmean(sza[good_int, good_spa]))
                n_pixels = int(len(good_int))
            else:
                sza_mean = float(np.nanmean(sza))  # fallback if no flagged pixels
                n_pixels = 0

            # save spectrum data
            if avg_wavelength is not None:
                collected_spectra.append({
                    "orbit": int(orbit_number),
                    "scan": int(scan + 1),
                    "wavelength_nm": avg_wavelength.astype(float),
                    "background_kR": avg_background.astype(float),
                    "aurora_fit_kR": avg_aurora.astype(float),
                    "r_squared": float(r_square),
                    "pearson_r": float(pearson_r),
                    "co_average_kR": float(coavg),
                    "co2p_average_kR": float(co2avg),
                    "sza_deg": float(sza_mean),
                    "n_pixels": int(n_pixels), 
                    "flag": detection_flag,
                    "anomalous": int(0)
                })


            # mark spectra which pass on detector images
            positive_spectra(detector_image_axes[scan], flag, color)

            # place possible detection region
            if (color == '#FDB813') | (color == '#2CA02C'):
                near_lat, near_lon, far_lat, far_lon = altitude_135(flag, color)
                any_pixels = True
            else:
                pass

            # plot tangent distance
            plot_tangent_distance()

            # save the data
            if (color == '#FDB813') | (color == '#2CA02C'):
                #full_csv()
                short_csv()

            # no need to continue if there weren't any good pixels
            if not any_pixels:
                print(f'No good pixels for scan {scan + 1}')
                continue

        # reset scan axes to fit data
        [axis.autoscale_view(tight=True) for axis in [scan_axis_co, scan_axis_co2p, scan_axis_solar, physical_axis]]

        # reset physical axis to equal spacing
        physical_axis.set_aspect('equal')

        # reset altitude/distance axis vertical limits
        ylim_alt = alt_dist_axis.get_ylim()
        alt_dist_axis.set_ylim(0, ylim_alt[1])
        ylim_tangent = tangent_dist_ax.get_ylim()
        tangent_dist_ax.set_ylim(0, ylim_tangent[1])

        # set map axis limits
        map_axis.set_xticks(np.arange(0, 360 + 30, 30))
        map_axis.set_xticks(np.arange(0, 360 + 5, 5), minor=True)
        map_axis.set_xlim(0, 360)
        map_axis.set_xlabel(r'Longitude [$\degree$]')
        map_axis.set_yticks(np.arange(-90, 90 + 30, 30))
        map_axis.set_yticks(np.arange(-90, 90 + 5, 5), minor=True)
        map_axis.set_ylim(-90, 90)
        map_axis.set_ylabel(r'Latitude [$\degree$]')

        # reset spectrum axis limits
        [ax.set_xticks([]) for ax in spectrum_axes.ravel()]
        [ax.set_yticks([]) for ax in spectrum_axes.ravel()]
        for col in range(2):
            spectrum_axes[0][col].set_xticks(np.arange(180, 340 + 40, 40))
            spectrum_axes[0][col].set_xticks(np.arange(180, 340 + 5, 5), minor=True)
            spectrum_axes[0][col].set_xlabel('Wavelength [nm]')
            for row in range(n_rows):
                ylims = spectrum_axes[row][col].get_ylim()
                if ylims[0] < ylims[1] * -0.25:
                    spectrum_axes[row][col].set_ylim(ylims[1] * -0.25, ylims[1])
                spectrum_axes[row][col].set_xlim(180, 340)

        # download spectra
        if collected_spectra:
            np.savez(
                block_spectra_dir / f"orbit{orbit_number:05d}_spectra.npz",
                items=np.array(collected_spectra, dtype=object)
            )

        orbit_str = f"{orbit_number:05d}"
        plt.savefig(block_quicklook_dir / f"orbit{orbit_str}.pdf", dpi=60)
        plt.close('all')
        print('Done!')