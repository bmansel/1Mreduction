# Copyright 2022 TPS 13A team
########################################################################
# This file is part of 1Mreduction.                                    #
#                                                                      #
# 1Mreduction is free software: you can redistribute it and/or modify  #
# it under the terms of the GNU General Public License as published by #
# the Free Software Foundation, either version 3 of the License, or    #
# (at your option) any later version.                                  #
#                                                                      #
# 1Mreduction is distributed in the hope that it will be useful,       #
# but WITHOUT ANY WARRANTY; without even the implied warranty of       #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        #
# GNU General Public License for more details.                         #
#                                                                      #
# You should have received a copy of the GNU General Public License    #
# along with 1Mreduction.  If not, see <https://www.gnu.org/licenses/>.#
########################################################################

import hdf5plugin
from pyFAI import azimuthalIntegrator
import fabio
import os
import argparse
import numpy as np
import sys
from datetime import datetime


def start_logging(expInfo, log_name):
    path = expInfo["exp_dir"] + "/" + log_name
    print("Running information is being  saved in: ", path)
    sys.stdout = open(path, 'a')


def uniquify(path):
    counter = 0
    if os.path.exists(path):
        while True:
            counter += 1
            newName = path + str(counter)
            if os.path.exists(newName):
                continue
            else:
                path = newName
                break
    return path


def lastPath(path):
    counter = 0
    if os.path.exists(path):
        while True:
            counter += 1
            newName = path + str(counter)
            if os.path.exists(newName):
                continue
            else:
                if counter != 1:
                    path = path + str(counter-1)
                break
    return path


def make_header_string(FIT2dParams, expInfo):
    # put each line in a list then loop through at the other end....
    temp_string = []
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    temp_string.append("1Mreduction data saved at " + dt_string)
    temp_string.append("1Mreduction release date: " + expInfo["release_date"] + "\n")
    
    for key in FIT2dParams:
        temp_string.append(str(key).ljust(20) + str(FIT2dParams[key]))

    for key in expInfo:
        if key != "civiSMP" and key != "rigiSMP" and key != "expSMP" and key != "civiBKG" and key != "rigiBKG" and key != "expBKG" and key != "release_date":
            temp_string.append(str(key).ljust(20) + str(expInfo[key]))
    temp_string.append("\nq [Ang^-1]    I(q)            error \n")
    header_string = '\n'.join(temp_string)

    return header_string


def write_1M_dat_file(header_string, file_name, dir, data):
    saveFileName = os.path.join(dir, file_name)
    np.savetxt(saveFileName, data, fmt='%1.6e',
               delimiter='    ', header=header_string)
    return


def makeFIT2dDic():
    FIT2dParams = {'directBeam': None, 'energy': None, 'beamX': None,
                   'beamY': None, 'tilt': None, 'tiltPlanRotation': None, 'detector': None}
    return FIT2dParams


def readWAXSpar(exp_dir, fname, detectorName):
    FIT2dParams = makeFIT2dDic()
    FIT2dParams['detector'] = detectorName  # "eiger1m"
    FIT2dParams['fname'] = fname
    lines = []
    count = 0
    with open(os.path.join(exp_dir, fname), encoding="utf8", errors='ignore') as f:
        lines = f.readlines()

    for count, line in enumerate(lines):
        if count == 0:
            FIT2dParams['directBeam'] = line.split()[0]
        if count == 1:
            FIT2dParams['energy'] = line.split()[0]
        if count == 2:
            FIT2dParams['beamX'] = line.split()[0]
        if count == 3:
            FIT2dParams['beamY'] = line.split()[0]
        if count == 4:
            FIT2dParams['tiltPlanRotation'] = line.split()[0]
        if count == 5:
            FIT2dParams['tilt'] = line.split()[0]
    return FIT2dParams


def makeAIobject(**kwargs):
    energy = kwargs['energy']
    directBeam = kwargs['directBeam']
    beamX = kwargs['beamX']
    beamY = kwargs['beamY']
    tilt = kwargs['tilt']
    tiltPlanRotation = kwargs['tiltPlanRotation']
    detector = kwargs['detector']

    plankC = float(4.135667696e-15)  # Planck's constant in ev/Hz
    speedLightC = float(299_792_458)  # speed of light m/s

    wavelengthcalc = plankC * speedLightC / 1000 / float(energy)
    ai = azimuthalIntegrator.AzimuthalIntegrator(
        detector=detector, wavelength=wavelengthcalc)
    ai.setFit2D(float(directBeam), float(beamX), 1065.0-float(beamY),
                tilt=float(tilt), tiltPlanRotation=float(tiltPlanRotation))

    return ai


def make_Eiger_mask(image):
    # mask vales are 2^32-1 = 4294967295 for 32 bit image
    maskPixels = np.squeeze(np.where(image == 4294967295))
    image.fill(0)
    # for pyFAI masked values are 1 and rest are 0
    image[maskPixels[0], maskPixels[1]] = 1
    return np.array(image)


def numLines(exp_dir, fname):
    lines = []
    with open(os.path.join(exp_dir, fname[2] + fname[0] + fname[1] + "002.txt")) as f:
        lines = f.readlines()
    return lines


def getNumFrames(lines):
    numLines = 0
    for line in lines:
        numLines += 1
    numFrames = numLines / 7
    return numFrames


def readHeaderFile(exp_dir, fname):
    lines = numLines(exp_dir, fname)
    numFrames = getNumFrames(lines)
    # get civi, rigi and exposure time values
    civi = []
    rigi = []
    expTime = []
    count = 0

    for line in lines:
        count += 1
        if count > numFrames and count <= 2*numFrames:
            civi.append(float(f'{line}'))
        if count > 2*numFrames and count <= 3*numFrames:
            rigi.append(float(f'{line}'))
        if count > 4*numFrames and count <= 5*numFrames:
            expTime.append(float(f'{line}'))

    return civi, rigi, expTime


def calc_norm(civi, rigi, thickness, TM, scale, use_rigi):
    if use_rigi is True:
        norm_value = rigi*thickness*TM/scale
    else:
        norm_value = civi*thickness*TM/scale
    return norm_value


def integrate(ai, verbose, FIT2dParams, **kwargs):

    # for now will unpack the dictonary in this ugly way
    # Future version change to class at the moment have
    # both

    # Needed for sample:
    exp_dir = kwargs["exp_dir"]
    mask = kwargs["mask"]
    smp_name = kwargs["smp_name"]
    average_smp_all = kwargs["average_smp_all"]
    if average_smp_all == False:
        avList_smp = kwargs['avList_smp']
    thickness = kwargs["thickness"]
    TM_smp = kwargs["TM_smp"]
    scale = kwargs["scale"]
    num_points = kwargs["num_points"]
    civiSmp = kwargs["civiSMP"]
    rigiSmp = kwargs["rigiSMP"]
    use_rigi = kwargs["use_rigi"]

    # Needed for background:
    bkg_name = kwargs['bkg_name']
    if bkg_name is not None:
        average_bkg_all = kwargs["average_bkg_all"]
        if average_bkg_all == False:
            avList_bkg = kwargs["avList_bkg"]
        civiBkg = kwargs["civiBKG"]
        rigiBkg = kwargs["rigiBKG"]
        TM_bkg = kwargs["TM_bkg"]

    imgSmp = fabio.open(os.path.join(exp_dir, smp_name + '_master.h5'))
    numFrames = imgSmp.nframes
    if verbose is True:
        print("Found ", str(numFrames), " frames in ",
              os.path.join(exp_dir, smp_name + '_master.h5'))

    header_string = make_header_string(
        FIT2dParams, kwargs)  # make header string

    if mask is None:
        if numFrames == 1:
            maskData = make_Eiger_mask(imgSmp.data)
        else:
            maskData = make_Eiger_mask(imgSmp.getframe(0).data)
        if verbose is True:
            print("Eiger mask (pixels with value = 4294967295 masked) applied")
    else:
        maskData = fabio.open(os.path.join(exp_dir, mask)).data
        if verbose is True:
            print("Mask file " + mask + " applied.")
    if average_smp_all is True:
        avList_smp = range(numFrames)

    # this next part is very ugly! Need to update and remove repeated code etc

    # average over some number of frames TIDY BY COMBING ALL AND SOME (FUTURE WORK)
    # elif average_smp_all is False and avList_smp is not None:
    if verbose is True:
        print("Starting integration of sample frames, completed:")
    ########################################################################
    # pyFAI integration options and vars
    ########################################################################
    # correctSolidAngle=True      # default
    # variance=None               # default
    # error_model="poisson"
    # radial_range=None           # default
    # azimuth_range=None          # default
    # mask=maskData
    # dummy=None                  # default
    # delta_dummy=None            # default
    # polarization_factor=None    # default
    # dark=None                   # default
    # flat=None                   # default
    # method='cython'
    # method='csr'                # default
    # unit='q_A^-1'
    # safe=False                  # default
    # normalization_factor=normValueSmp # set in code later
    # metadata=None               # default
    ###############################################################################

    if avList_smp is not None:
        numFrames = len(avList_smp)
        num_av = 0
        if len(avList_smp) > 1:
            for frame in avList_smp:
                normValueSmp = calc_norm(
                    civiSmp[frame], rigiSmp[frame], thickness, TM_smp, scale, use_rigi)
                q, ISmp, errSmp = ai.integrate1d(imgSmp.getframe(
                    frame).data, num_points, error_model="poisson", mask=maskData, unit='q_A^-1', normalization_factor=normValueSmp)
                if num_av == 0:
                    I_sum = ISmp
                    err_sum = np.power(errSmp, 2)
                elif num_av > 0:
                    I_sum = np.add(I_sum, ISmp)
                    err_sum = np.add(err_sum, np.power(errSmp, 2))
                num_av += 1
                if verbose is True:
                    print("    Sample frame number: " + str(frame) +
                          ", processed " + str(num_av) + " of " + str(len(avList_smp)))
            if verbose is True:
                print("Averaging sample frames is complete.")
            IMeanSmp = np.divide(I_sum, numFrames)
            errMeanSmp = np.divide(np.sqrt(err_sum), numFrames)

        elif numFrames == 1:
            #normValueSmp = civiSmp*thickness*TM_smp/scale
            normValueSmp = calc_norm(
                civiSmp[0], rigiSmp[0], thickness, TM_smp, scale, use_rigi)
            q, IMeanSmp, errMeanSmp = ai.integrate1d(
                imgSmp.data, num_points, error_model="poisson", mask=maskData, unit='q_A^-1', normalization_factor=normValueSmp)
            if verbose is True:
                print("    Sample frame number: " + str(frame) +
                      ", processed " + str(num_av) + " of " + str(len(avList_smp)))

        elif numFrames == 0:
            print("error at line 120 num frames 0")

        # If no subtract background then directly save
        if bkg_name is None:
            # save data to .dat file in experiment folder
            file_name = smp_name+"_av_" + \
                str(avList_smp[0])+"-"+str(avList_smp[-1])+"_1M.dat"
            write_1M_dat_file(header_string, file_name, exp_dir,
                              np.transpose([q, IMeanSmp, errMeanSmp]))
            if verbose is True:
                print("    Sample text file saved to " + exp_dir +
                      "/" + file_name + " no background set, finished.")
            return

    # no average
    # elif average_smp_all == False and avList_smp is None:
    if avList_smp is None:
        if numFrames > 1:
            IAll = []
            errAll = []
            if verbose is True:
                print("Starting radial integration of sample:")
            for frame in range(numFrames):
                normValueSmp = calc_norm(
                    civiSmp[frame], rigiSmp[frame], thickness, TM_smp, scale, use_rigi)
                q, I, err = ai.integrate1d(imgSmp.getframe(
                    frame).data, num_points, error_model="poisson", mask=maskData, unit='q_A^-1', normalization_factor=normValueSmp)
                IAll.append(I), errAll.append(err)
                if verbose is True:
                    print("   Completed: " + str(frame) + " / " +
                          str(numFrames) + " sample frame")
            if verbose:
                print("No multi frame averaging")

        elif numFrames == 1:
            normValueSmp = calc_norm(
                civiSmp[0], rigiSmp[0], thickness, TM_smp, scale, use_rigi)
            q, IAll, errAll = ai.integrate1d(
                imgSmp.data, num_points, error_model="poisson", mask=maskData, unit='q_A^-1', normalization_factor=normValueSmp)
            if verbose is True:
                print("Only one image in the experiment file, processed.")
        elif numFrames == 0:
            print("error at line 120 num frames 0")

        # If no subtract background then directly save
        if bkg_name is None:
            # save data to .dat file in dir inside experiment dir
            saveDir = uniquify(os.path.join(exp_dir, smp_name + "_1M"))
            os.mkdir(saveDir)
            if verbose:
                print("Saving Files")
            for frame in range(numFrames):
                file_name = smp_name+"_1M_" + str(frame) + ".dat"
                write_1M_dat_file(header_string, file_name, exp_dir, np.transpose(
                    [q, IAll[frame], errAll[frame]]))
                if verbose is True:
                    print("    Saved frame ", str(frame),
                          "  to: ", exp_dir, "/", file_name)
            return

    # calc background, dud code from above
    if bkg_name is not None:
        imgBkg = fabio.open(os.path.join(exp_dir, bkg_name + '_master.h5'))
        numFrames = imgBkg.nframes

        if average_bkg_all is True:
            avList_bkg = range(numFrames)

        numFrames = len(avList_bkg)
        count = 0
        if len(avList_bkg) > 1:
            for frame in avList_bkg:
                normValueBkg = calc_norm(
                    civiBkg[frame], rigiBkg[frame], thickness, TM_bkg, scale, use_rigi)
                q, IBkg, errBkg = ai.integrate1d(imgBkg.getframe(
                    frame).data, num_points, error_model="poisson", mask=maskData, unit='q_A^-1', normalization_factor=normValueBkg)
                if count == 0:
                    I_bkg_sum = IBkg
                    err_bkg_sum = np.power(errBkg, 2)
                elif count >= 1:
                    I_bkg_sum = np.add(I_bkg_sum, IBkg)
                    err_bkg_sum = np.add(err_bkg_sum, np.power(errBkg, 2))
                count += 1
                if verbose is True:
                    print("    Radially averaging background frames, completed " +
                          str(frame) + " number " + str(frame) + " of " + str(numFrames))
            IMeanBkg = np.divide(I_bkg_sum, numFrames)
            errMeanBkg = np.divide(np.sqrt(err_bkg_sum), numFrames)
            if verbose:
                print("Averaging background frames is complete")
        elif numFrames == 1:
            normValueBkg = calc_norm(
                civiBkg[0], rigiBkg[0], thickness, TM_bkg, scale, use_rigi)
            q, IMeanBkg, errMeanBkg = ai.integrate1d(
                imgBkg.data, num_points, error_model="poisson", mask=maskData, unit='q_A^-1', normalization_factor=normValueBkg)
            if verbose is True:
                print("A single background frame was selected and radially intergrated")
        elif numFrames == 0:
            print("error at line 120 num frames 0")

        # subtract Background from sample, save data and return from function
        if avList_smp is not None:  # case i
            ISubd = np.subtract(IMeanSmp, IMeanBkg)
            # adding error = sqrt(err1^2 + err2^2)
            errSubd = np.sqrt(
                np.add(np.power(errMeanSmp, 2), np.power(errMeanBkg, 2)))
            # save data
            file_name = smp_name+"_av_" + \
                str(avList_smp[0])+"-"+str(avList_smp[-1]) + "_bkg_sub_1M.dat"
            write_1M_dat_file(header_string, file_name, exp_dir,
                              np.transpose([q, ISubd, errSubd]))

            if verbose is True:
                print("    Subtracted data saved to " +
                      exp_dir + "/" + file_name)
        elif avList_smp is None:  # case ii
            saveDir = uniquify(os.path.join(exp_dir, smp_name + "_1M"))
            os.mkdir(saveDir)
            for frame in range(len(IAll)):
                ISubd = np.subtract(IAll[frame], IMeanBkg)
                # adding error = sqrt(err1^2 + err2^2)
                errSubd = np.sqrt(
                    np.add(np.power(errAll[frame], 2), np.power(errMeanBkg, 2)))
                # save data
                file_name = smp_name+"_1M_" + str(frame) + ".dat"
                write_1M_dat_file(header_string, file_name,
                                  saveDir, np.transpose([q, ISubd, errSubd]))
                if verbose is True:
                    print("    Saved background subtracted frame number " +
                          str(frame) + " of " + str(len(IAll)) + " to " + str(saveDir) + "/" + file_name)
            return


def hyphen_range(s):
    """ Takes a range in form of "a-b" and generate a list of numbers between a and b inclusive.
    Also accepts comma separated ranges like "a-b,c-d,f" will build a list which will include
    Numbers from a to b, a to d and f"""
    s = "".join(s.split())  # removes white space
    r = set()
    for x in s.split(','):
        t = x.split('-')
        if len(t) not in [1, 2]:
            raise SyntaxError("hash_range is given its arguement as " +
                              s+" which seems not correctly formated.")
        r.add(int(t[0])) if len(t) == 1 else r.update(
            set(range(int(t[0]), int(t[1])+1)))
    l = list(r)
    l.sort()
    return l


def run_parser():
    parser = argparse.ArgumentParser(
        description='Reduction program for TPS 13A 1M detector')
    #parser.add_argument('--out_dir', action='store', help='dir to store results inside exp_dir')
    parser.add_argument('-ed', '--exp_dir',
                        action='store', type=str, default=None,
                        help='Directory of experiment, if not set exp_dir == current working directory. (default: %(default)s)')

    parser.add_argument('-sn', '--smp_name',
                        action='store', type=str, default=None,
                        help='3 characters for sample file letter first. (default: %(default)s)')

    parser.add_argument('-bn', '--bkg_name',
                        action='store', type=str, default=None,
                        help='3 characters for bkground file letter first. (default: %(default)s)')

    parser.add_argument('-ts', '--TMsmp',
                        action='store', type=float, default=1.0,
                        help='Value of sample transmission. (default: %(default)s)')

    parser.add_argument('-tb', '--TMbkg',
                        action='store', type=float, default=1.0,
                        help='Value of background transmission. (default: %(default)s)')

    parser.add_argument('-s', '--scale',
                        action='store', type=float, default=1.0,
                        help='Value to scale (multiply) data by. (default: %(default)s)')

    parser.add_argument('-as', '--avg_smp',
                        action='store', type=str, default='all',
                        help='frames to average, all, none, a (-) and (,) seperated list. (default: %(default)s)')

    parser.add_argument('-ab', '--avg_bkg',
                        action='store', type=str, default='all',
                        help='frames to average, all or a (-) and (,) seperated list. (default: %(default)s)')

    parser.add_argument('-t', '--thickness',
                        action='store', type=float, default=1.0,
                        help='Sample thickness [mm]. (default: %(default)s)')

    parser.add_argument('-m', '--mask',
                        action='store', type=str, default=None,
                        help='Mask name in exp dir, if none use eiger mask. (default: %(default)s)')

    parser.add_argument('-np', '--num_points',
                        action='store', type=int, default=1000,
                        help='Number of I(q) data points. (default: %(default)s)')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='If -v then output information while running (default: %(default)s)')
    parser.add_argument('-l', '--logfile',
                        action='store_true',
                        help='If -l then output information to logfile 1Mreduction.log (default: %(default)s)')
    # parser.add_argument("--type", default="toto", choices=["toto","titi"],
    #                          help = "type (default: %(default)s)")
    args = parser.parse_args()
    return args, parser


def make_exp_dic(args, release_date):
    exp_dic = {'release_date': release_date}

    # logic for averaging sample and bkg
    args.avg_smp = args.avg_smp.replace(" ", "")
    if args.avg_smp == 'none':
        exp_dic['average_smp_all'] = False
        exp_dic['avList_smp'] = None
    elif args.avg_smp == 'all':
        exp_dic['average_smp_all'] = True
    else:
        exp_dic['avList_smp'] = hyphen_range(args.avg_smp)
        exp_dic['average_smp_all'] = False

    args.avg_bkg = args.avg_bkg.replace(" ", "")
    if args.avg_bkg == 'all':
        exp_dic['average_bkg_all'] = True
    else:
        exp_dic['avList_bkg'] = hyphen_range(args.avg_bkg)
        exp_dic['average_bkg_all'] = False

    #out_dir = args.out_dir
    exp_dic['TM_smp'] = args.TMsmp
    exp_dic['TM_bkg'] = args.TMbkg
    exp_dic['scale'] = args.scale
    # + '_master.h5' # change to the correct order app bit
    exp_dic['smp_name'] = args.smp_name[1] + \
        args.smp_name[2] + args.smp_name[0]
    exp_dic['bkg_name'] = args.bkg_name
    if args.bkg_name is not None:
        # + '_master.h5' # change to the correct order app bit
        exp_dic['bkg_name'] = args.bkg_name[1] + \
            args.bkg_name[2] + args.bkg_name[0]
    exp_dic['exp_dir'] = args.exp_dir
    if exp_dic['exp_dir'] is None:
        exp_dic['exp_dir'] = os.getcwd()
    exp_dic['thickness'] = args.thickness
    exp_dic['mask'] = args.mask
    exp_dic['num_points'] = args.num_points

    #FIT2dParams = readWAXSpar(exp_dic['exp_dir'], "WAXSpar.txt")
    #ai = makeAIobject(**FIT2dParams)

    exp_dic['civiSMP'], exp_dic['rigiSMP'], exp_dic['expSMP'] = readHeaderFile(
        exp_dic['exp_dir'], exp_dic['smp_name'])
    exp_dic['use_rigi'] = False
    if min(exp_dic['civiSMP']) == 0:
        exp_dic['use_rigi'] = True

    if exp_dic['bkg_name'] is not None:
        exp_dic['civiBKG'], exp_dic['rigiBKG'], exp_dic['expBKG'] = readHeaderFile(
            exp_dic['exp_dir'], exp_dic['bkg_name'])
        if min(exp_dic['civiBKG']) == 0:
            exp_dic['use_rigi'] = True
    verbose = args.verbose
    logfile = args.logfile
    return exp_dic, verbose, logfile


def print_main_info(FIT2dParams, expInfo, preamble, ai):
    # datetime object containing current date and time
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print("\n1Mreduction started at ", dt_string)

    print(preamble)

    print("Fit2d parameter file read: " +
          expInfo['exp_dir'] + "/" + FIT2dParams["fname"])

    print("Found the following values:")
    for key in FIT2dParams:
        print("    ", key, ':', FIT2dParams[key])

    print("pyFAI azimuthal integration object ai contains:")
    for key in ai.get_config():
        print("    ", key, ':', ai.get_config()[key])

    print("Using the following experimental information:")
    for key in expInfo:
        print("    ", key, ':', expInfo[key])


def main():
    preamble = """
    #################################################
    This is a initial release of 1Mreduction.py which
    delivers working code and has been tested.
    Future work will will tidy and follow better
    pythonic methods before the final release.
    #################################################
    """
    release_date = "10th Feb 2022"
    args, parser = run_parser()

    # check that we at least have an experiment directory and file name
    if args.exp_dir is None:
        print(
            "ERROR no experimental directory was given. Please see help below for the input")
        parser.print_help()
        print("Leaving.")
        return

    if args.smp_name is None:
        print("ERROR no sample name was given. Please see help below for the input")
        parser.print_help()
        print("Leaving.")
        return

    expInfo, verbose, logfile = make_exp_dic(args, release_date)

    if logfile is True:
        start_logging(expInfo, "1Mreduction.log")
        verbose = True

    # read fit2d parameters from this file
    FIT2dParams = readWAXSpar(
        expInfo['exp_dir'], fname="WAXSpar.txt", detectorName="eiger1m")

    ai = makeAIobject(**FIT2dParams)
    if verbose is True:
        print_main_info(FIT2dParams, expInfo, preamble, ai)
    else:
        print("Started, verbose off, supressing output.")
        # print(preamble)
        # print("Fit2d parameter file read: " + expInfo['exp_dir'] + "/" + fit2d_parameter_file)
        # print("Found the following values:")
        # for element in FIT2dParams:
        #     print("    ", element,':', FIT2dParams[element])
        # print("pyFAI azimuthal integration object ai contains:")
        # for element in ai.get_config():
        #     print("    ", element,':', ai.get_config()[element])

    integrate(ai, verbose, FIT2dParams, **expInfo)
    if not verbose:
        print("finished")


# Run if executed, but not if it is imported from other python script
if __name__ == "__main__":
    main()
