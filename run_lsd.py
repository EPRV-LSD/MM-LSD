#!/usr/bin/env python

# coding: utf-8

# In[2]:

trace_memory = False
if trace_memory:
    import tracemalloc
    tracemalloc.start()

import numpy as np
from pandas import read_csv, DataFrame, concat

import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.cm import rainbow
from matplotlib.colors import rgb2hex

from scipy.sparse import csc_matrix
from scipy.optimize import curve_fit
from scipy.stats import median_abs_deviation

from time import time
from sys import path as syspath
from sys import argv
import os

from multiprocessing import Pool, get_context
from functools import partial
import pickle


from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv, spsolve
from scipy.interpolate import interp1d, LSQUnivariateSpline
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

start_time = time()
def get_vstep(wlen):
    # get velocity step from wavelength array
    diffs = np.diff(wlen, prepend=np.nan)
    vel_space = 3e5 * diffs/wlen

    vel_space = np.nanmedian(vel_space)
    vel_space = np.round(vel_space, decimals=2)

    return vel_space

def worker3(order, weights, spectrum, wavelengths, vlambda, vdepth, vel):
    """
    For given order: perform LSD and extract the common profile, common profile uncertainties. Compute convolution model
    Parameters
    ----------
    order : int
        First array: periods [d]
    weights : array orders x pixels
        Weights of the individual fluxes
    spectrum : array orders x pixels
        Fluxes
    wavelengths : array orders x pixels
        Wavelength corresponding to the fluxes
    vlambda : array
        Central wavelength of absorption lines (VALD3)
    vdepth : array
        Depth of absorption lines (VALD3)
    vel : array
        Velocity grid to run LSD on (velocity grid for common profile)

    Output:
    ----------
    Z : array
        common profile
    Zerr : array
        uncertainty estimates of common profile
    M.dot(Z) : array
        convolution model
    selection : array
        indices of included pixels
    """

    # Get data of given order

    # Only include data with weight > 0
    selection = np.where((weights[order, :] > 0) & (~np.isnan(wavelengths[order, :])))[0]
    spectrum_o = spectrum[order, :]

    # Don't run if only 2% of order included. Bad order.
    # Change to 10 per cent of order included?
    if len(selection) < 0.1 * len(spectrum_o):
        return 0

    spectrum_o = spectrum[order, :][selection]
    wavelengths_o = wavelengths[order, :][selection]
    weights_o = weights[order, :][selection]

    # CREATE CONVOLUTION MATRIX
    # -----------------------------------------------------
    value, row, column = an.cvmt(wavelengths_o, vel, vlambda, vdepth)
    M = csc_matrix((value, (row, column)), shape=(len(wavelengths_o), len(vel)))
    # -----------------------------------------------------
    try:
        Z, Zerr = runLSD_inv(
            value, row, column, len(wavelengths_o), len(vel), weights_o, spectrum_o
        )
    except:
        import pdb; pdb.set_trace()
    
    return Z, Zerr, M.dot(Z), selection



t00 = time()

try:
    rc("text", usetex=True)
except:
    pass
rc("xtick", labelsize=15)
rc("ytick", labelsize=15)
rcParams["font.family"] = "STIXGeneral"
rcParams["mathtext.fontset"] = "stix"
rcParams["hatch.linewidth"] = 2.0

plt.rcParams.update(
    {
        "font.size": 15,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

try:
    get_ipython().__class__.__name__
    injupyternotebook = True

except:
    injupyternotebook = False


# In[3]:

injupyternotebook = False

# star name
star = argv[1]

output_intermediate_results = False


stardir = "./stars/" + star + "/"

inf_name = stardir + "input.py"
exec(open(inf_name).read())


# In[5]:


syspath.insert(0, "./helper_functions/")

from classes import (
    stellar_parameters,
    Analyse,
    # extract_rv_from_common_profiles,
    Gaussian,
    runLSD_inv,
    prep_spec3,
    RVerror,
    read_datafile
)



paramnr = int(argv[2])
# results so far
resfile = f"{resdir}results_{star}_{indic}.csv"
results = read_csv(resfile)

print(paramnr)


# load parameter combination
prms = read_csv(stardir + "params.csv")

maxdepthparam = prms["maxdepthparam"][paramnr]
mindepthparam = prms["mindepthparam"][paramnr]
telluric_cut = prms["telluric_cut"][paramnr]
velgridwidth = prms["velgridwidth"][paramnr]
modelspecdeviationcut = prms["modelspecdeviationcut"][paramnr]
max_nr_of_specs = prms["max_nr_of_specs"][paramnr]
exclwidelinesparam = prms["exclwidelinesparam"][paramnr]

rassoption = prms["rassoption"][paramnr]
erroption = prms["erroption"][paramnr]
telloption = prms["telloption"][paramnr]

# here we save the results of this parameter combination
newres = DataFrame()

# copy entries of parameter file
for key in prms.keys():
    newres[key] = [prms[key][paramnr]]


# ### Information about spectra

info_file = read_csv(dirdir + "Info.csv")

# set system RV. i.e. RV that is used to convert absorption line wavelengths from rest frame to stellar frame
# TODO: This should be handled in the preprocessing stage
if pipname == "ESSP":
    # systemrv = 83
    systemrv = info_file["rv_ccf"][0] / 1000.0
else:
    systemrv = info_file["rv_ccf"][0]


# ### Load data from VALD3
#

valddir = "./VALD_files/"
sp = stellar_parameters(star, valddir, dirdir, pipname, c)
sp.VALD_data()


# ### Load data from 1_preprocess notebook
#

an = Analyse(sp.VALDlambdas, sp.VALDdepths, pipname, dirdir)

# TODO: This has to change...
# with open(dirdir + "data_dict.pkl", "rb") as f:
#     prov = pickle.load(f)
#     an.alldata = {}
#     if not overlap_correction:
#         an.alldata["spectrum"] = prov["spectrum"]
#         an.alldata["err"] = prov["err"]
#         an.alldata["err_envelope"] = np.zeros_like(prov["err"])
#         an.alldata["wavelengths"] = prov["wavelengths"]
#     elif rassoption == 1:
#         an.alldata["spectrum"] = prov["spectrum_overlap_corrected"]
#         an.alldata["err"] = prov["err_overlap_corrected"]
#         an.alldata["err_envelope"] = prov["err_envelope_overlap_corrected"]
#         an.alldata["wavelengths"] = prov["wavelengths"]

#     if pipname == "ESSP":
#         an.alldata["t_map"] = prov["t_map"]
    

    

    # del prov


test_ii = 0
iis = list(info_file.index)
test_data = read_datafile(test_ii, dirdir)
nr_of_orders, nr_of_pixels = test_data["spectrum"].shape
if auto_vStep:
    test_wlen = test_data['wavelengths']

    vStep = get_vstep(test_wlen)
    print(f"Velocity spacing calculated from wavelength array: {vStep} km/s")
else:
    vStep = manual_vStep
    print(f"Velocity spacing set to {vStep} km/s")
del test_data


# index numbers of spectra
an.iis = iis

# see input.py
# TODO: change Analyse.init()
an.excllower = excllower
an.exclupper = exclupper
an.telluric_cut = telluric_cut
an.modelspecdeviationcut = modelspecdeviationcut
an.mindepthparam = mindepthparam
an.maxdepthparam = maxdepthparam
an.exclwidelinesparam = exclwidelinesparam
an.telloption = telloption



# shift fluxes to between -1 and 0 for lsd procedure


# for key in iis:
#     an.alldata["spectrum"][key] = an.alldata["spectrum"][key] - 1


an.tapas_tellurics = {}
an.resdir = resdir


# In[12]:

# TODO: Handle this in Analyse.init()
an.barycentric_to_stellar_restframe = {}
an.observatory_to_barycentric_restframe = {}
an.observatory_to_stellar_restframe = {}

for ii in iis:
    an.barycentric_to_stellar_restframe[ii] = 1.0 / (1.0 + systemrv / c)
    an.observatory_to_barycentric_restframe[ii] = 1.0 + info_file["berv"][0] / c
    an.observatory_to_stellar_restframe[ii] = (
        an.observatory_to_barycentric_restframe[ii]
        * an.barycentric_to_stellar_restframe[ii]
    )


# ### Get tapas telluric information


# TODO: This is not how I thought the TAPAS files worked. 
# Not a single array, nobs x spectrum pickle file...

# TODO: How to change this to work generally for spectrographs...
# TODO: 
if pipname != "ESSP":


    compute_tellurics = True

    if os.path.exists("./tellurics/tellurics" + star + ".pkl"):
        with open("./tellurics/tellurics" + star + ".pkl", "rb") as f:
            an.tapas_tellurics = pickle.load(f)

        if len(iis) == len(an.tapas_tellurics.keys()):
            compute_tellurics = False
            if output_intermediate_results:
                print(f"loaded tellurics from ./tellurics/tellurics" + star + ".pkl")
        else:
            compute_tellurics = True

    if compute_tellurics:
        print("produce tellurics")

        transmittance_file = None
        an.get_tapas_transmittance(pipname, transmittance_file, info_file)
        print("save tellurics in ", "./tellurics/tellurics" + star + ".pkl")
        with open("./tellurics/tellurics" + star + ".pkl", "wb") as f:
            pickle.dump(an.tapas_tellurics, f)


# set velocity grid

# roughly centre velocity grid around the rv of the star of the first measurement. add +- dvel km/s
dvel = 20
vel_inital = np.arange(systemrv - dvel, systemrv + dvel, vStep)

# set upper limit to number of absorption lines of depth min_depth_required within a region (other regions excluded)
an.vel_initial = vel_inital


#  FIRST RUN OF LSD (TO GET FWHM OF SPECTRA, FIRST COMMON PROFILE,
# AND TO CHECK DEVIATION OF SPECTRA FROM CONVOLUTION MODEL))


# choose test spectrum for the first LSD run

an.test_ii = test_ii

# get fluxes, wavelengths, and weights for first spectrum

# TODO: THis seems like it can be refactored

an.prep_spec(iis[test_ii], erroption)
print('DID THIS!!!')
# exit()
# choose echelle orders to run code on (all here)
testorders = np.arange(nr_of_orders)

# get rough common profile (equal weight for each order). this is only used to get an idea about the common profile shape.
zlast = np.zeros((len(vel_inital)))
model_h = np.zeros((nr_of_orders, nr_of_pixels))
count = 0

for order in testorders:
    output = an.worker(order, vel_inital)
    if not np.isnan(output[0]).any():
        model_o = np.nan * np.ones_like(output[2])

        # import pdb; pdb.set_trace()
        model_o[output[2]] = output[1]
        model_h[order, :] = model_o
        # model_h[order, :] = output[1]
        zlast += output[0]
        count += 1
zlast /= count

an.model_h = model_h
an.div = np.abs(model_h - an.spectrum)


# first common profile

# fit gaussian to common profile and extraxt hwhm
popt, pcov = curve_fit(Gaussian, vel_inital, zlast, [-1, systemrv, 3, 0])
fit = Gaussian(vel_inital, *popt)

vel_hwhm = np.abs(popt[2]) * np.sqrt(np.log(2.0) * 2.0)

if output_intermediate_results:
    plt.figure(figsize=(5, 3))
    plt.plot(vel_inital, zlast, ".", label="Common Profile")
    plt.plot(vel_inital, fit, label="Fit (Gaussian)")

    plt.xlabel("Vel")
    plt.title(np.round(vel_hwhm, 2))
    plt.legend()
    plt.close()

# estimate typical half-width of an absorption line as 5 times the hwhm
# will be multiplied by wvl when used

an.initial_v_halfwidth = vel_hwhm


# ### Set velocity grid


# new velocity grid based on first run.

dvel = np.round(vel_hwhm) * velgridwidth
vel = np.arange(systemrv - dvel, systemrv + dvel, vStep)

# set upper limit to number of absorption lines of depth min_depth_required within a region (other regions excluded)
an.vel = vel

# how much should we exclude near data points with high model-spectrum deviation?
an.absline_halfwidth_include = (vel.max() - vel.min() + 1.0) / 2.0 / c


# ### EXCLUDE SPECTRAL REGIONS WITH HIGH MODEL-SPECTRUM DEVIATION


an.get_wide_lines()
an.get_q_map(info_file)
# get telluric map.

# if not pipname == "ESSP":
#     # TODO: This will require many files to be read in 
#     an.get_t_map()

if paramnr == 0:
    an.show_map()


# ### RUN LSD ON ALL SPECTRA WITH QUALITY/TELLURIC MAP


# multiprocessing
num_processors = 4

# on which orders to run
testorders = np.arange(nr_of_orders)

# save results here
LSD_results = {}

vel = an.vel


t_start = time()

for ii in iis:
    if output_intermediate_results:
        print(ii, np.round(time() - t_start, 2))

    # get weights, spectrum, wavelengths after excluding some data according to parameters.
    weights, spectrum, wavelengths = an.prep_spec3(
        ii,
        erroption=erroption,
        usetapas=usetapas,
        pipname=pipname,
    )

    # empty containers
    LSD_results[ii] = {}
    common_profile_all_orders = np.zeros((np.shape(wavelengths)[0], len(vel)))
    common_profile_all_orders_err = np.zeros((np.shape(wavelengths)[0], len(vel)))

    # NOTE: I don't know why the following line doesn't have the first 20 orders...
    MZ = np.zeros((np.shape(wavelengths)[0], len(spectrum[20, :])))
    incl_map = np.zeros((np.shape(MZ)))

    # partial function for multiprocessing
    worker_partial3 = partial(
        worker3,
        weights=weights,
        spectrum=spectrum,
        wavelengths=wavelengths,
        vlambda=sp.VALDlambdas,
        vdepth=sp.VALDdepths,
        vel=vel,
    )
    # initialise multiprocessing
    if num_processors > 1:
        with get_context("fork").Pool(processes=num_processors) as p:
            output = p.map(worker_partial3, [order for order in testorders])
    else:
        output = []
        for order in testorders:
            output.append(worker_partial3(order))

    # with get_context("fork").Pool(processes=num_processors) as p:
    #     output = p.map(worker_partial3, [order for order in testorders])


    # save output into containers
    for order in testorders:
        if output[order] != 0:
            common_profile_all_orders[order, :] = output[order][0]
            common_profile_all_orders_err[order, :] = output[order][1]
            selection = output[order][3]
            MZ[order, :][selection] = output[order][2]
            incl_map[order, :][selection] = np.ones((len(selection)))

    # save results in dict
    # FIXME: This dict might not be good for memory efficiency
    LSD_results[ii]["common_profile"] = common_profile_all_orders
    LSD_results[ii]["common_profile_err"] = common_profile_all_orders_err
    LSD_results[ii]["LSD_spectrum_model"] = MZ  # LSD_spectrum
    LSD_results[ii]["incl_map"] = incl_map

    # plt.figure(figsize=(7, 5))

    # for order in range(len(LSD_results[test_ii]["common_profile"])):

    #     color = rgb2hex(rainbow(order / nr_of_orders))

    #     plt.plot(vel, LSD_results[ii]["common_profile"][order], color=color)

    #     if order // 10 == order / 10:
    #         plt.plot(
    #             vel,
    #             LSD_results[ii]["common_profile"][order],
    #             color=color,
    #             label="Order " + str(order),
    #         )
    #     else:
    #         plt.plot(vel, LSD_results[ii]["common_profile"][order], color=color)

    # plt.legend()
    # plt.ylim(-1.3, 0.2)

    # plt.xlabel("Velocity grid [km/s]")
    # plt.title("Common profiles of individual orders")

    # plt.savefig(an.resdir + f"Common_profiles.pdf")
    # plt.close()


# define weight matrix to compute order weights (to combine common profiles to master common profile)

wmat = np.ones((len(iis), nr_of_orders))

for count1, ii in enumerate(iis):

    data = read_datafile(ii, dirdir)
    if erroption == 0:
        pre_weights = 1.0 / (data["err"] ** 2)
    if erroption == 1:
        pre_weights = 1.0 / (data["err_envelope"] ** 2)
    if erroption == 2:
        err = np.transpose(
            np.tile(
                np.median(data["err"], axis=1),
                (np.shape(data["err"])[1], 1),
            )
        )
        pre_weights = 1.0 / (err ** 2)

    pre_weights[LSD_results[ii]["incl_map"] == 0] = 0

    for count, order in enumerate(testorders):
        wmat[count1, order] = np.nanmean(pre_weights[order, :])

an.order_weight = np.mean(wmat, axis=0)


# In[26]:


# check excluding outer parts of common profile. see paper.
testsigma = np.arange(1.0, 5, step=0.25) * an.initial_v_halfwidth


std_dep_on_sigma = np.zeros((len(testsigma)))
an.fitfunction = "Gaussian"
order_choice = np.arange(nr_of_orders)

for count, sigma in enumerate(testsigma):
    an.sigmafit = sigma

    lsd_rv_orig, Zs, Z, Zerrs = an.extract_rv_from_common_profiles(
        LSD_results,
        iis,
        order_choice,
        weight_orders=weight_schemes[0],
        use_uncertainties=True,
    )

    lsd_norm_t = lsd_rv_orig - np.median(lsd_rv_orig)
    no_outliers = np.where(
        np.abs(lsd_norm_t - np.median(lsd_norm_t)) < delta_rv_outlier
    )[0]

    lsd_norm_t = lsd_norm_t[no_outliers]
    std_dep_on_sigma[count] = np.std(lsd_norm_t)

an.sigmafit = testsigma[np.argmin(std_dep_on_sigma)]
an.sigmafit_used = np.copy(testsigma[np.argmin(std_dep_on_sigma)])


# In[27]:


if len(iis) != len(no_outliers):
    print(
        f"Removed {len(iis)-len(no_outliers)} out of {len(iis)} spectra due to |RV-med(RV)| >= delta_rv_outlier (= {delta_rv_outlier} m/s)"
    )


# In[42]:


# compare LSD results to DRS CCF method
if pipname == "ESSP":
    drs_rv_orig = info_file["rv_ccf"].values
else:
    drs_rv_orig = info_file["rv_ccf"].values * 1000.0
t = info_file["mjd"].values


for weight_scheme in weight_schemes:
    use_uncertainties = True
    # only second one is used for plots later on

    # "flux weight2b": weight of order o same for all spectra (=weight of order o in first spectrum)
    # "flux weight2c": weight of order o varies (depending on weight of fluxes in order o in spectrum ii)

    # choose an LSD container

    # this extracts the RV information
    lsd_rv_orig, Zs, Z, Zerrs = an.extract_rv_from_common_profiles(
        LSD_results,
        iis,
        order_choice,
        weight_orders=weight_scheme,
        use_uncertainties=use_uncertainties,
    )

    if pipname == "DRS_3.7":
        # drift correction
        lsd_rv_orig -= info_file["drift"].values
        # drs_rv_orig -= info_file["drift"].values

    # subtract median radial velocity to analyse rv change
    drs_norm = drs_rv_orig - np.median(drs_rv_orig)
    lsd_norm = lsd_rv_orig - np.median(lsd_rv_orig)

    # ------------------------
    # remove outliers
    no_outliers = np.where(np.abs(drs_norm - np.median(drs_norm)) < 200)[0]

    if len(no_outliers) < len(drs_norm):
        print(f"Removed {len(drs_norm)-len(no_outliers)} outliers.")

    drs_norm = drs_norm[no_outliers]
    lsd_norm = lsd_norm[no_outliers]

    t = t[no_outliers]

    # yerr = np.asarray(list(an.alldata["ccfrvs_err"].values()))[no_outliers]*1000.
    yerr = info_file["rv_ccf_error"].values[no_outliers] * 1000.0

    # ------------------------

    difference = drs_norm - lsd_norm

    print("LSD RMS: ", np.std(lsd_norm).round(2), "m/s")


# In[43]:


if output_intermediate_results:

    fig, ax = plt.subplots(
        2, 1, figsize=(14, 3.8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    ax[0].set_title("RVs in barycentric frame")
    # ax[0].set_ylim(-10,20)

    ax[0].plot(
        t, drs_norm, "D", label=f"CCF technique. RMS: {np.std(drs_norm):.2f} m/s"
    )
    ax[0].plot(
        t, lsd_norm, ".", label=f"LSD technique. RMS: {np.std(lsd_norm):.2f} m/s"
    )
    # ax[0].plot(t-2450000 ,lsd_norm_avg,".",label="LSD\_avg")

    ax[0].set_ylabel("RV [m/s]", fontsize=15)
    ax[0].legend(fontsize=15)
    ax[1].errorbar(t, difference, yerr=yerr, fmt="o", color="black")

    plt.xlabel(r"MJD", fontsize=15)

    print(
        f"STD:\t LSD: {np.std(lsd_norm):.2f} m/s \t \t DRS: {np.std(drs_norm):.2f} m/s"
    )
    print(
        f"MAD:\t LSD: {median_abs_deviation(lsd_norm):.2f} m/s \t \t DRS: {median_abs_deviation(drs_norm):.2f} m/s"
    )

    plt.savefig(an.resdir + f"RVs.pdf")


# In[34]:


# save results
injupyternotebook = False
if not injupyternotebook:
    yerr_wls = np.zeros((len(lsd_norm)))
    for count, ii in enumerate(np.asarray(iis)[no_outliers]):
        rverrc = RVerror(vel, Zs[ii], Zerrs[ii])
        yerr_wls[count] = rverrc * 1000.0

    newres["LSD RV std"] = [np.std(lsd_norm).round(3)]
    newres["LSD RV MAD"] = [median_abs_deviation(lsd_norm).round(3)]
    newres["DRS RV std"] = [np.std(drs_norm).round(3)]
    newres["DRS RV MAD"] = [median_abs_deviation(drs_norm).round(3)]
    newres["sigmafit_used"] = [an.sigmafit_used.round(3)]
    newres["comp time"] = [np.round(time() - t00, 1)]

    nn = concat([results, newres])
    nn.to_csv(resfile, index=False)

    print("Results saved in ", resfile)
    if os.path.exists(rvresfile):
        f = open(rvresfile, "rb")
        dth = pickle.load(f)
        dth[paramnr] = lsd_norm
        f.close()
    else:
        f = open(rvresfile, "wb")
        dth = {}
        dth["mjd"] = t
        dth["rv_ccf"] = drs_norm
        dth["rv_ccf_err"] = yerr

        dth[paramnr] = lsd_norm

    f = open(rvresfile, "wb")
    pickle.dump(dth, f)
    f.close()

    if os.path.exists(rverrresfile):
        f = open(rverrresfile, "rb")
        dth = pickle.load(f)
        dth[paramnr] = yerr_wls
        f.close()
    else:
        f = open(rverrresfile, "wb")
        dth = {}
        dth["mjd"] = t
        dth["rv_ccf"] = drs_norm
        dth["rv_ccf_err"] = yerr

        dth[paramnr] = yerr_wls

    f = open(rverrresfile, "wb")
    pickle.dump(dth, f)
    f.close()

    if os.path.exists(commonprofilefile):
        f = open(commonprofilefile, "rb")
        dth = pickle.load(f)
        dth[f"vel_{paramnr}"] = vel
        dth[f"Z_{paramnr}"] = Zs
        f.close()
    else:
        f = open(commonprofilefile, "wb")
        dth = {}
        dth["mjd"] = t
        dth[f"vel_{paramnr}"] = vel
        dth[f"Z_{paramnr}"] = Zs

    f = open(commonprofilefile, "wb")
    pickle.dump(dth, f)
    f.close()

if trace_memory:
    current, peak = tracemalloc.get_traced_memory()
    # import pdb; pdb.set_trace()
    tracemalloc.stop()

    print("Memory usage")
    print(f"Current usage: {current / 10 ** 6} MB")
    print(f"Peak usage: {peak / 10 ** 6} MB")
print("Time taken: ", np.round(time() - start_time, 2), "s")