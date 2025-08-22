#!/usr/bin/env python
# coding: utf-8

trace_memory = True
if trace_memory:
    import tracemalloc
    tracemalloc.start()
import numpy as np
from astropy.io import fits
from astropy.time import Time
import matplotlib.pyplot as plt
import sys
import glob
import subprocess
import os
import pandas as pd
import subprocess
from scipy import interpolate
from scipy.interpolate import (
    InterpolatedUnivariateSpline,
    CubicSpline,
    UnivariateSpline,
    LSQUnivariateSpline,
)
from scipy.signal import savgol_filter, find_peaks
import pickle

sys.path.insert(0, "./helper_functions/")

from preprocess_functions import (
    find_corresponding_s1d_and_ccf,
    upper_envelope2,
    open_pickle,
    find_closest,
)


try:
    get_ipython().__class__.__name__
    injupyternotebook = True

except:
    injupyternotebook = False

injupyternotebook = False


if not injupyternotebook:
    star = sys.argv[1]
else:
    star = "Sun"
    # star = "teststar"


def get_lambda_telluric_frame(hdul):

    # get lambda
    data = hdul[0].data
    # lm = deepcopy(data)
    lm = np.zeros((np.shape(data)))
    d = hdul[0].header["TNG DRS CAL TH DEG LL"]

    for o in range(np.shape(lm)[0]):
        xx = np.arange(len(data[o]))

        # aj = hdul[0].header["TNG DRS CAL TH COEFF LL"+str(jj(o,d,0))]
        # lm[o] = aj * xx**0

        for ii in range(d + 1):
            aj = hdul[0].header["TNG DRS CAL TH COEFF LL" + str(jj(o, d, ii))]
            lm[o] += aj * xx**ii
        # lm[o] = lm[o]

    return lm


def jj(o, d, i):
    return o * (d + 1) + i


def get_flux_and_blaze(hdul, hdul_blaze):
    # not divided by blaze

    flux = hdul[0].data
    blaze = hdul_blaze[0].data

    return flux, blaze


inf_name = "./stars/" + star + "/input.py"


exec(open(inf_name).read())

sys.path.insert(0, rassine_loc)

# rawdir must exist. dirdir and rassine_res will be created and populated

if not os.path.exists(dirdir):
    os.mkdir(dirdir)
if not os.path.exists(rassine_res):
    os.mkdir(rassine_res)

all_info = pd.DataFrame()


nr_of_orders = 0
if extract_info:
    if pipname == "DRS_2.3.5":

        # svnr is the unique identifier of the spectrum. file names e.g. Info_1, spec..._1, wave..._1 etc.
        svnr = 0

        # find all e2ds files
        e2ds_files = sorted(glob.glob(rawdir + "/*S2D_A.fits"))

        for ii in range(min(len(e2ds_files), number_of_spectra_to_reduce)):

            # choose iith e2ds file and search for corresponding s1d and ccf file
            try:
                # search files
                # is there a skysub file? (s2d/s1d/ccf with sky subtraction)
                skysub_name = e2ds_files[ii].replace("S2D", "S2D_SKYSUB")
                if len(glob.glob(skysub_name)) > 0 and False:
                    e2ds_files[ii] = skysub_name

                s1d_file = e2ds_files[ii].replace("S2D", "S1D")
                ccf_file = e2ds_files[ii].replace("S2D", "CCF")

                hdul = fits.open(e2ds_files[ii])
                hdul_s1d = fits.open(s1d_file)
                hdul_ccf = fits.open(ccf_file)

            except Exception as e:
                # not all necessary files present
                print("Some error here (1_preprocess).", e)
                continue

            if nr_of_orders == 0:
                nr_of_orders = hdul[1].data.shape[0]

            general = {}

            # needed
            general["rv_ccf"] = hdul_ccf[0].header["HIERARCH " + sid + " QC CCF RV"]
            general["ccf_fwhm"] = hdul_ccf[0].header["HIERARCH " + sid + " QC CCF FWHM"]
            general["berv"] = hdul_ccf[0].header[
                "HIERARCH " + sid + " " + pipeline_keyword + " BERV"
            ]
            general["mjd"] = hdul[0].header["MJD-OBS"]

            for snorder in np.arange(1, nr_of_orders + 1):
                try:
                    general[f"sn{snorder}"] = hdul_ccf[0].header[
                        f"HIERARCH " + sid + f" QC ORDER{snorder} SNR"
                    ]
                except:
                    general[f"sn{snorder}"] = 0

            try:
                general["rv_ccf_error"] = hdul_ccf[0].header[
                    "HIERARCH " + sid + " QC CCF RV ERROR"
                ]
                general["drift"] = 0.0
                general["e2dsname"] = (e2ds_files[ii],)
                general["s1dname"] = (s1d_file,)
                general["ccfname"] = ccf_file
                general["dateobs"] = hdul[0].header["DATE-OBS"]
                general["ccf_contrast"] = hdul_ccf[0].header[
                    "HIERARCH " + sid + " QC CCF CONTRAST"
                ]
                general["rv_ccf_uncorr"] = hdul_ccf[0].header[
                    "HIERARCH " + sid + " " + pipeline_keyword + " CCF RV"
                ]
                general["bjd_ccf"] = hdul_ccf[0].header[
                    "HIERARCH " + sid + " " + pipeline_keyword + " BJD"
                ]
                general["rhk"] = 0
                general["airmass"] = hdul[0].header["AIRMASS"]
                general["el"] = hdul[0].header["EL"]
                general["exptime"] = hdul[0].header["EXPTIME"]
                general["pip_id"] = hdul[0].header["HIERARCH ESO PRO REC1 PIPE ID"]
                general["humidity"] = hdul[0].header[
                    "HIERARCH " + sid + " METEO HUMIDITY"
                ]
                general["pressure"] = hdul[0].header[
                    "HIERARCH " + sid + " METEO PRESSURE"
                ]
                general["windspeed"] = hdul[0].header[
                    "HIERARCH " + sid + " METEO WINDSPEED"
                ]
                general["winddir"] = hdul[0].header[
                    "HIERARCH " + sid + " METEO WINDDIR"
                ]
                general["dust"] = hdul[0].header["HIERARCH " + sid + " INS DUST ST"]
                general["mask"] = hdul[0].header["HIERARCH " + sid + " QC CCF MASK"]
                general["CCF_asymmetry"] = hdul[0].header[
                    "HIERARCH " + sid + " QC CCF FLUX ASYMMETRY"
                ]
                general["bisector_span"] = hdul[0].header[
                    "HIERARCH " + sid + " QC CCF BIS SPAN"
                ]
            except Exception as e:
                print(f"At least one keyword missing. Not critical.", e)

            df = pd.DataFrame(general, index=[ii])

            all_info = pd.concat([all_info, df])
            hdul.close()
            hdul_s1d.close()
            hdul_ccf.close()

            print("spectrum id", ii, "iteration", svnr)
            svnr += 1

        all_info.to_csv(dirdir + "Info.csv", index=False)
        del df
        del all_info

    elif pipname == "DRS_3.7":

        # svnr is the unique identifier of the spectrum. file names e.g. Info_1, spec..._1, wave..._1 etc.
        svnr = 0

        # find all e2ds files
        e2ds_files = sorted(glob.glob(rawdir + "/*e2ds_A.fits"))
        print(f"found {(len(e2ds_files))} files")
        for ii in range(min(len(e2ds_files), number_of_spectra_to_reduce)):

            # choose iith e2ds file and search for corresponding s1d and ccf file
            try:
                # search files
                s1d_file, ccf_file = find_corresponding_s1d_and_ccf(
                    e2ds_files[ii], rawdir
                )

                hdul = fits.open(e2ds_files[ii])
                hdul_s1d = fits.open(s1d_file)
                hdul_ccf = fits.open(ccf_file)

                # blaze_file = rawdir+"/"+hdul[0].header["HIERARCH "+sid+" DRS BLAZE FILE"]
                # hdul_blaze = fits.open(blaze_file)

            except Exception as e:
                # not all necessary files present
                print("Some error here (1_preprocess).", e)
                continue

            if nr_of_orders == 0:
                nr_of_orders = hdul[0].data.shape[0]
            general = {}

            general["rv_ccf"] = hdul_ccf[0].header["HIERARCH " + sid + " DRS CCF RVC"]
            general["ccf_fwhm"] = hdul_ccf[0].header[
                "HIERARCH " + sid + " DRS CCF FWHM"
            ]
            general["mjd"] = hdul[0].header["MJD-OBS"]
            general["berv"] = hdul_ccf[0].header[
                "HIERARCH " + sid + " " + pipeline_keyword + " BERV"
            ]
            general["drift"] = hdul_ccf[0].header[
                "HIERARCH " + sid + " DRS DRIFT RV USED"
            ]

            for snorder in np.arange(1, nr_of_orders + 1):
                try:
                    general[f"sn{snorder}"] = hdul_ccf[0].header[
                        f"HIERARCH " + sid + f" DRS SPE EXT SN{snorder}"
                    ]
                except:
                    general[f"sn{snorder}"] = 0

            try:
                general["rhk"] = hdul[0].header["HIERARCH " + sid + " DRS CAII RHK"]
            except:
                general["rhk"] = 0

            try:
                general["rv_ccf_error"] = 0
                general["ccf_contrast"] = hdul_ccf[0].header[
                    "HIERARCH " + sid + " DRS CCF CONTRAST"
                ]
                general["rv_ccf_uncorr"] = hdul_ccf[0].header[
                    "HIERARCH " + sid + " " + pipeline_keyword + " CCF RV"
                ]
                general["bjd_ccf"] = hdul_ccf[0].header[
                    "HIERARCH " + sid + " " + pipeline_keyword + " BJD"
                ]
                general["airmass"] = hdul[0].header["AIRMASS"]
                general["dateobs"] = hdul[0].header["DATE-OBS"]
                general["el"] = hdul[0].header["EL"]
                general["exptime"] = hdul[0].header["EXPTIME"]
                general["e2dsname"] = (e2ds_files[ii],)
                general["s1dname"] = (s1d_file,)
                general["ccfname"] = ccf_file
            except Exception as e:
                print(f"At least one keyword missing. Not critical.", e)

            df = pd.DataFrame(general, index=[ii])

            all_info = pd.concat([all_info, df])

            hdul.close()
            hdul_s1d.close()
            hdul_ccf.close()

            print("spectrum id", ii, "iteration", svnr)
            svnr += 1

        all_info.to_csv(dirdir + "Info.csv", index=False)
        del all_info


    elif pipname == "ESSP":
        run_rassine = False
        svnr = 0

        summary_files = glob.glob(rawdir + "/*.csv")
        assert len(summary_files) == 1, "Need exactly one summary file"
        summary_file = summary_files[0]
        print(f"found summary file {summary_file}")
        summary = pd.read_csv(summary_file)

        # find all e2ds files
        spec_files = sorted(glob.glob(rawdir + "/*_spec_*.fits"))

        # first_instrument = spec_files[0].split("_")[-1].split(".")[0]
        # print(f"ESSP instrument name: {first_instrument}")

        instrument = None
        print(f"found {(len(spec_files))} files")

        for ii in range(min(len(spec_files), number_of_spectra_to_reduce)):
            general = {}

            with fits.open(spec_files[ii]) as hdul, fits.open(
                spec_files[ii].replace("spec", "ccfs")
            ) as hdul_ccf:
                header = hdul[0].header
                nr_of_orders = hdul[1].data.shape[0]
                header_ccf = hdul_ccf[0].header

            if instrument is None:
                instrument = header["INSTRUME"]
                print(f"ESSP instrument name: {instrument}")

            else:
                assert (
                    header["INSTRUME"] == instrument
                ), "All files must be from the same instrument"

            general["instrument"] = instrument
            general["rv_ccf"] = header_ccf["RV"]
            general["rv_ccf_error"] = header_ccf["E_RV"]
            general["ccf_fwhm"] = summary.loc[
                summary["Standard File Name"] == spec_files[ii].split("/")[-1],
                "CCF FWHM [km/s]",
            ].values[0]
            general["berv"] = header["BERV"]
            general["mjd"] = header["MJD_UTC"]
            general["bjd_ccf"] = header["MJD_UTC"] #FIXME: I'm not sure this is the right time object
            general["drift"] = 0
            general["dateobs"] = Time(header["JD_UTC"], format="jd").isot
            general["e2dsname"] = (spec_files[ii],)
            general["s1dname"] = "None"
            general["ccfname"] = spec_files[ii].replace("spec", "ccfs")
            for snorder in np.arange(1, nr_of_orders + 1):
                general[f"sn{snorder}"] = 0
            general["airmass"] = header["AIRMASS"]
            general["el"] = 0
            general["exptime"] = header["EXPTIME"]
            general["pip_id"] = "ESSP"
            general["humidity"] = 0
            general["pressure"] = 0
            general["windspeed"] = 0
            general["winddir"] = 0
            general["dust"] = 0
            general["mask"] = header_ccf["MASK"]
            general["CCF_asymmetry"] = 0
            general["bisector_span"] = 0

            df = pd.DataFrame(general, index=[ii])
            all_info = pd.concat([all_info, df])

            print("spectrum id", ii, "iteration", svnr)
            svnr += 1

        all_info.to_csv(dirdir + "Info.csv", index=False)
        del all_info    

info_file = pd.read_csv(dirdir + "Info.csv")


fewerdp = False

if run_rassine:

    print("Running RASSINE for ", len(info_file.index), " files")
    for ii in info_file.index:
        os.chdir(rt)

        info = info_file.iloc[ii]

        s1d_filename = info["s1dname"]
        with fits.open(s1d_filename) as s1d_fits:

            # get s1d data
            if pipname == "DRS_3.7":
                step = s1d_fits[0].header["CDELT1"]
                start = s1d_fits[0].header["CRVAL1"]
                yy = s1d_fits[0].data
                xx = start + step * np.arange(len(yy))
            if pipname == "DRS_2.3.5":
                xx = s1d_fits[1].data["wavelength_air"]
                yy = s1d_fits[1].data["flux"]
            if fewerdp:
                xx = xx[::2]
                yy = yy[::2]

            rassine_res_file = rassine_res + "rass_" + str(ii) + ".csv"
            # make sure spectrum_name is set to cwd+'/spectra_library/stars/spec_toreduce.csv' in Rassine_config.py.
            temp_name = rassine_res + "spec_toreduce.csv"
            copyto = rassine_loc + "/spectra_library/stars/"

            # save to temp_name
            df = pd.DataFrame({"wave": xx, "flux": yy})
            df.to_csv(temp_name, index=False)

            # copy to RASSINE folder
            command = "cp " + temp_name + " " + copyto
            os.system(command)

            # run RASSINE

            try:
                os.chdir(rassine_loc)
                subprocess.call([sys.executable, "Rassine.py"])

                # batcmd="python Rassine.py"
                # output = subprocess.check_output(batcmd, shell=True)
            except Exception as E:
                print(E)
                continue
            os.chdir(rt)

            # print(output)
            # get RASSINE results

            continuum_file = rassine_loc + "/output/RASSINE_spec_toreduce.p"
            continuum_file_open = open_pickle(continuum_file)
            continuum_flux = continuum_file_open["output"]["continuum_linear"]
            continuum_wvl = continuum_file_open["wave"]

            # save to separate files in case nomalisation fails for one spectrum
            df = pd.DataFrame(
                {
                    "s1d_wave": xx,
                    "s1d_flux": yy,
                    "continuum_wave": continuum_wvl,
                    "continuum_flux": continuum_flux,
                }
            )
            df.to_csv(rassine_res_file, index=False)
            print(rassine_res_file)
            # command = "rm "+temp_name
            # os.system(command)
            os.remove(temp_name)
            print(ii)


# In[10]:

if not pipname == "ESSP":
    rass_files = sorted(glob.glob(rassine_res + "/*rass*csv"))
    assert len(rass_files) == len(info_file.index), "not all files processed by RASSINE"

    ii = np.random.choice(info_file.index)

    # plt.title(f"spec {ii}")
    rassine_res_file = rassine_res + "rass_" + str(ii) + ".csv"
    # res = pd.read_csv(rassine_res_file)
    # plt.plot(res["s1d_wave"], res["s1d_flux"])
    # plt.plot(res["continuum_wave"], res["continuum_flux"])

print(pipname)

if save_spectra_as_pkl:
    if pipname == "DRS_2.3.5":
        ii_no_rassine = []

        print("Found", len(info_file.index), "processed spectra")

        for ii in info_file.index:

            # print(ii)

            info = info_file.iloc[ii]

            s1d_file = info["s1dname"]
            ccf_file = info["ccfname"]
            e2ds_file = info["e2dsname"]

            hdul = fits.open(e2ds_file)
            hdul_s1d = fits.open(s1d_file)
            hdul_ccf = fits.open(ccf_file)

            rassine_res_file = rassine_res + "rass_" + str(ii) + ".csv"

            try:
                res = pd.read_csv(rassine_res_file)
                spline = InterpolatedUnivariateSpline(
                    res["continuum_wave"], res["continuum_flux"]
                )
            except:
                print("No RASSINE file for this Info file")
                ii_no_rassine.append(ii)
                continue

            flux = hdul[1].data
            lmd = hdul[5].data
            dll = hdul[7].data

            flux_s1dadjusted = np.zeros((np.shape(flux)))
            continuum_s1dadjusted = np.zeros((np.shape(flux)))
            error_envelope_continuum_normalised = np.zeros((np.shape(flux)))
            error_continuum_normalised = hdul[2].data

            for ordr in range(np.shape(lmd)[0]):
                # mimic steps taken in DRS to create s1d from e2ds (except reinterpolation on grid)
                # result: flux level of e2ds same as s1d but on different wavelengths

                fctr = (hdul[0].header["HIERARCH " + sid + " RV STEP"] / c * lmd / dll)[
                    ordr, :
                ]
                flux_s1dadjusted[ordr, :] = flux[ordr, :] * fctr

                # -----------------------------------------------------
                # reinterpolate continuum from s1d-rassine to e2ds wavelengths. spline function from above
                continuum_s1dadjusted_o = spline(lmd[ordr, :])
                continuum_s1dadjusted[ordr, :] = continuum_s1dadjusted_o

                # -----------------------------------------------------
                # ERROR CALCULATION
                # -----------------------------------------------------

                # ERROR WITH ABSORPTION LINES
                error_continuum_normalised[ordr, :] *= fctr / continuum_s1dadjusted_o

                # -----------------------------------------------------
                # ERROR WITHOUT ABSORPTION LINES

                error_envelope_continuum_normalised[ordr, :] = upper_envelope2(
                    lmd[ordr, :], error_continuum_normalised[ordr, :]
                )
                # -----------------------------------------------------

            # now that the rassine-continuum has been reinterpolated on the e2ds wvl grid, we can continuum-normalise
            flux_continuum_normalised = flux_s1dadjusted / continuum_s1dadjusted

            # -----------------------------------------------------
            # save as npy bc matrices can be easily stored in this format.
            data_dict = {}
            data_dict["spectrum"] = flux_continuum_normalised
            # data_dict["spectrum_e2ds"][ii] = flux
            data_dict["wavelengths"] = lmd
            data_dict["continuum"] = continuum_s1dadjusted
            data_dict["err"] = error_continuum_normalised
            data_dict["err_envelope"] = error_envelope_continuum_normalised

            # -----------------------------------------------------

            del res
            del flux_continuum_normalised
            del lmd
            del continuum_s1dadjusted
            del flux_s1dadjusted
            del error_continuum_normalised
            del error_envelope_continuum_normalised

            with open(dirdir + f"data_{ii}.pkl", "wb") as f:
                pickle.dump(data_dict, f)

            hdul.close()
            hdul_s1d.close()
            hdul_ccf.close()

    elif pipname == "DRS_3.7":

        ii_no_rassine = []

        print("Found", len(info_file.index), "processed spectra")

        for ii in info_file.index:

            # print(ii)

            info = info_file.iloc[ii]

            s1d_file = info["s1dname"]
            ccf_file = info["ccfname"]
            e2ds_file = info["e2dsname"]

            hdul = fits.open(e2ds_file)
            hdul_s1d = fits.open(s1d_file)
            hdul_ccf = fits.open(ccf_file)

            blaze_file = rawdir + "/" + hdul[0].header["HIERARCH TNG DRS BLAZE FILE"]

            hdul_blaze = fits.open(blaze_file)

            rassine_res_file = rassine_res + "rass_" + str(ii) + ".csv"

            try:
                res = pd.read_csv(rassine_res_file)
                spline = InterpolatedUnivariateSpline(
                    res["continuum_wave"], res["continuum_flux"]
                )
            except:
                print("No RASSINE file for this Info file")
                ii_no_rassine.append(ii)
                continue
            berv = hdul_ccf[0].header[
                "HIERARCH " + sid + " " + pipeline_keyword + " BERV"
            ]
            lmd = get_lambda_telluric_frame(hdul)
            lmd = lmd * (1 + berv / c)
            flux, blaze = get_flux_and_blaze(hdul, hdul_blaze)

            flux_s1dadjusted = np.zeros((np.shape(flux)))
            continuum_s1dadjusted = np.zeros((np.shape(flux)))
            error_envelope_continuum_normalised = np.zeros((np.shape(flux)))
            error_continuum_normalised = np.sqrt(flux + 144.0) / blaze

            for ordr in range(np.shape(lmd)[0]):
                # mimic steps taken in DRS to create s1d from e2ds (except reinterpolation on grid)
                # result: flux level of e2ds same as s1d but on different wavelengths

                dx = lmd[ordr][1:] - lmd[ordr][:-1]
                dx = np.hstack((dx, dx[-1]))
                fctr = 0.01 / dx
                flux_s1dadjusted[ordr, :] = flux[ordr, :] / blaze[ordr, :] * fctr

                # -----------------------------------------------------
                # reinterpolate continuum from s1d-rassine to e2ds wavelengths. spline function from above
                continuum_s1dadjusted_o = spline(lmd[ordr, :])
                continuum_s1dadjusted[ordr, :] = continuum_s1dadjusted_o

                # -----------------------------------------------------
                # ERROR CALCULATION
                # -----------------------------------------------------

                # ERROR WITH ABSORPTION LINES
                error_continuum_normalised[ordr, :] *= fctr / continuum_s1dadjusted_o

                # -----------------------------------------------------
                # ERROR WITHOUT ABSORPTION LINES

                escn = np.sqrt(continuum_s1dadjusted_o / fctr * blaze[ordr, :] + 144.0)
                # perform same steps as on e2ds spectra (e2ds to s1d)
                escn = escn * fctr / blaze[ordr, :]
                # divide by continuum
                escn = escn / continuum_s1dadjusted_o
                error_envelope_continuum_normalised[ordr, :] = escn

                # -----------------------------------------------------

            # now that the rassine-continuum has been reinterpolated on the e2ds wvl grid, we can continuum-normalise
            flux_continuum_normalised = flux_s1dadjusted / continuum_s1dadjusted

            # -----------------------------------------------------
            # save as npy bc matrices can be easily stored in this format.
            data_dict = {}
            data_dict["spectrum"] = flux_continuum_normalised
            # data_dict["spectrum_e2ds"][ii] = flux
            data_dict["wavelengths"] = lmd
            data_dict["continuum"] = continuum_s1dadjusted
            data_dict["err"] = error_continuum_normalised
            data_dict["err_envelope"] = error_envelope_continuum_normalised
            # -----------------------------------------------------
            
            del flux_continuum_normalised
            del lmd
            del continuum_s1dadjusted
            del flux_s1dadjusted
            del error_continuum_normalised
            del error_envelope_continuum_normalised

            with open(dirdir + f"data_{ii}.pkl", "wb") as f:
                pickle.dump(data_dict, f)

            # close all files   
            hdul.close()
            hdul_s1d.close()
            hdul_ccf.close()
            hdul_blaze.close()

            del res

    elif pipname == "ESSP":

        orders_to_omit = {
            "HARPS-N": [0, 1, 2, 3],
            "NEID": [],
            "EXPRES": [0, 1],
            "HARPS": [0, 46],
        }

        print("Found", len(info_file.index), "processed spectra")
        for ii in info_file.index:
            s1d_file = info_file.loc[ii, "s1dname"]
            ccf_file = info_file.loc[ii, "ccfname"]
            e2ds_file = info_file.loc[ii, "e2dsname"]

            with fits.open(e2ds_file) as hdul, fits.open(ccf_file) as hdul_ccf:
                lmd = hdul[1].data
                flux = hdul[2].data
                error = hdul[3].data
                continuum = hdul[4].data
                telluric_map = hdul[7].data

                orders = range(lmd.shape[0])
                mask = ~(
                    np.isin(orders, orders_to_omit[info_file.loc[ii, "instrument"]])
                )

                lmd = lmd[mask, :]
                flux = flux[mask, :]
                error = error[mask, :]
                continuum = continuum[mask, :]
                telluric_map = telluric_map[mask, :]

                # flux[np.isnan(flux)] = 0

                assert lmd.shape[0] == flux.shape[0]
                assert lmd.shape[0] == error.shape[0]
                assert lmd.shape[0] == continuum.shape[0]
                assert lmd.shape[0] == telluric_map.shape[0]
                assert lmd.shape[1] == flux.shape[1]
                assert lmd.shape[1] == error.shape[1]
                assert lmd.shape[1] == continuum.shape[1]
                assert lmd.shape[1] == telluric_map.shape[1]

                flux[telluric_map == 1] = 0
                error[telluric_map == 1] = 1e6
                continuum[telluric_map == 1] = 1

                # vStep = hdul_ccf[0].header["VSPACING"]
                flux_s1dadjusted = np.zeros((np.shape(flux)))
                continuum_s1dadjusted = np.zeros((np.shape(flux)))
                error_envelope_continuum_normalised = np.zeros((np.shape(flux)))
                error_continuum_normalised = np.zeros((np.shape(flux)))
                for ordr in range(np.shape(lmd)[0]):
                    dll = np.diff(lmd[ordr], prepend=lmd[ordr][1] - lmd[ordr][0])
                    flux_s1dadjusted[ordr, :] = flux[ordr, :]
                    continuum_s1dadjusted[ordr, :] = continuum[ordr, :]
                    error_continuum_normalised[ordr, :] = (
                        error[ordr, :] / continuum_s1dadjusted[ordr, :]
                    )

                    # error_envelope_continuum_normalised[ordr, :] = upper_envelope2(
                    #     lmd[ordr, :], error_continuum_normalised[ordr, :]

                    # )

                flux_continuum_normalised = flux_s1dadjusted / continuum_s1dadjusted
                data_dict = {}
                data_dict["spectrum"] = flux_continuum_normalised
                data_dict["wavelengths"] = lmd
                data_dict["continuum"] = continuum_s1dadjusted
                data_dict["err"] = error_continuum_normalised
                data_dict["err_envelope"] = error_envelope_continuum_normalised
                data_dict["t_map"] = telluric_map
                # -----------------------------------------------------
                del flux 
                del error
                del continuum

                
                del flux_continuum_normalised
                del lmd
                del continuum_s1dadjusted
                del flux_s1dadjusted
                del error_continuum_normalised
                del error_envelope_continuum_normalised
                del telluric_map


                with open(dirdir + f"data_{ii}.pkl", "wb") as f:
                    pickle.dump(data_dict, f)
                del data_dict
                hdul.close()
                hdul_ccf.close()



# In[15]:

if overlap_correction:
    # MULTIPLICATIVE


    for ii in info_file.index:
        print(ii)

        info = info_file.iloc[ii]
        if not pipname == "ESSP":
            s1d_file = info["s1dname"]
            s1d_fits = fits.open(s1d_file)

            if pipname == "DRS_3.7":
                step = s1d_fits[0].header["CDELT1"]
                start = s1d_fits[0].header["CRVAL1"]
                fluxs1d = s1d_fits[0].data
                wvls1d = start + step * np.arange(len(fluxs1d))
            else:
                wvls1d = s1d_fits[1].data["wavelength_air"]
                fluxs1d = s1d_fits[1].data["flux"]
        with open(dirdir + f"data_{ii}.pkl", "rb") as f:
            data_dict = pickle.load(f)
        try:
            error_continuum_normalised = data_dict["err"]
            error_envelope_continuum_normalised = data_dict["err_envelope"]
        except:
            print(ii, "did not run")

        conts1d = data_dict["continuum"]
        wvl = data_dict["wavelengths"]
        flux = data_dict["spectrum"] * conts1d

        s1dspline = InterpolatedUnivariateSpline(wvls1d, fluxs1d)
        cont_new = np.copy(conts1d)

        for order in range(nr_of_orders - 1):

            wvl1 = np.copy(wvl[order])
            wvl2 = np.copy(wvl[order + 1])

            flux1 = np.copy(flux[order])
            flux2 = np.copy(flux[order + 1])

            # overlap
            nr = 500

            first_index = find_closest(wvl1, wvl2.min())[0]
            last_index = find_closest(wvl2, wvl1.max())[0]
            wvl1a = wvl1[first_index:]
            wvl2a = wvl2[:last_index]

            flux1a = flux1[first_index:]
            flux2a = flux2[:last_index]

            fluxs1d1_interp = s1dspline(wvl1a)
            fluxs1d2_interp = s1dspline(wvl2a)

            ratio1_raw = np.hstack((np.ones(nr), flux1a / fluxs1d1_interp))
            ratio2_raw = np.hstack((flux2a / fluxs1d2_interp, np.ones((nr))))

            to_test = np.copy(ratio1_raw)
            for ij in range(1, len(to_test)):
                previous = to_test[ij - 1]
                if to_test[ij] > 1.5 or to_test[ij] < 0.5:
                    to_test[ij] = previous
            savgolfit = savgol_filter(to_test, int(len(ratio1_raw) / 10) * 2 + 1, 3)
            ratio1 = savgolfit[nr:]

            to_test = np.copy(ratio2_raw)

            # ADDED 210114. FIRST RATIO DP SHOULD BE 1 BY DEFAULT (OTHERWISE SEQUENCE OF OUTLIERS POSSIBLE)
            to_test[0] = 1
            for ij in range(1, len(to_test)):
                previous = to_test[ij - 1]
                if to_test[ij] > 1.5 or to_test[ij] < 0.5:
                    to_test[ij] = previous

            savgolfit2 = savgol_filter(to_test, int(len(ratio2_raw) / 10) * 2 + 1, 3)
            ratio2 = savgolfit2[:-nr]

            cont_new[order, first_index:] = conts1d[order, first_index:] * ratio1
            cont_new[order + 1, :last_index] = conts1d[order + 1, :last_index] * ratio2

        data_dict["spectrum_overlap_corrected"] = flux / cont_new
        data_dict["err_overlap_corrected"] = (
            error_continuum_normalised * conts1d / cont_new
        )
        data_dict["err_envelope_overlap_corrected"] = (
            error_envelope_continuum_normalised * conts1d / cont_new
        )
        # data_dict["continuum_overlap_corrected"][ii] = cont_new
    f = open(dirdir + "data_dict.pkl", "wb")
    pickle.dump(data_dict, f)
    f.close()


# In[ ]:
if trace_memory:
    current, peak = tracemalloc.get_traced_memory()
    print("Preprocessing:")
    print("peak memory usage is", peak / 10 ** 6, "MB")
    tracemalloc.stop()