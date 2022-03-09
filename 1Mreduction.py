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
    if expInfo["out_dir"] is None:
        path = expInfo["exp_dir"] + "/" + log_name
    else:
        if not os.path.exists(expInfo["exp_dir"] + "/" + expInfo["out_dir"]):
            os.makedirs(expInfo["exp_dir"] + "/" + expInfo["out_dir"])
        path = expInfo["exp_dir"] + "/" + expInfo["out_dir"] + "/" + log_name
    print("Running information is being  saved in: ", path)
    sys.stdout = open(path, 'w')

def uniquify(path): # this is not being used currently
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

def lastPath(path): #This is not being used
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


def write_1M_dat_file(save_name, header_string, data):
    np.savetxt(save_name, data, fmt='%1.6e',
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

def make_Eiger_mask(image):
    if image.nframes == 1:
        frame = image.data
    else:
        frame = image.getframe(0).data

    # mask vales are 2^32-1 = 4294967295 for 32 bit image
    maskPixels = np.squeeze(np.where(frame == 4294967295))
    frame.fill(0)
    # for pyFAI masked values are 1 and rest are 0
    frame[maskPixels[0], maskPixels[1]] = 1
    return np.array(frame)

def import_reject_mask(exp_dir, reject_mask):
    data = np.loadtxt(exp_dir + "/" + reject_mask, usecols=(0,1))
    return data


def make_reject_mask(image, reject_data):
    # take x, y reject data and make 2d image mask
    #image[image == 1] = 0  # this is zero already
    image[ np.array(reject_data[:,1], dtype=np.int_) , np.array(reject_data[:,0], dtype=np.int_) ] = 1
    return np.array(image)

def make_counting_mask(image, counting_data):
    image[ np.array(counting_data[:,1], dtype=np.int_) , np.array(counting_data[:,0], dtype=np.int_) ] = 1
    image = 1 - image # invert maske
    return np.array(image)


def combine_masks(eiger_mask, user_mask, reject_mask, counting_mask):
    combined_mask = eiger_mask + user_mask + reject_mask + counting_mask
    combined_mask[combined_mask > 1] = 1
    return np.array(combined_mask)

def calc_norm(civi, rigi, thickness, TM, scale, use_rigi):
    if use_rigi is True:
        norm_value = rigi*thickness*TM/scale
    else:
        norm_value = civi*thickness*TM/scale
    return norm_value

def make_save_dir(exp_dir, out_dir):
    if not os.path.exists(exp_dir + "/" + out_dir):
        os.makedirs(exp_dir + "/" + out_dir)
        print("Created " + exp_dir + "/" + out_dir)

def make_save_filename(exp_dir, output_name_prefix, out_dir, smp_name, frames):
    # smp_name should be string 
    # frames a list of lists
    #check if save directory exists and make it if not

    #if output_name_prefix is not None:
    if len(smp_name) == 1:
        range_string = createRangeString(frames[0])
        #output_name_prefix = output_name_prefix + "_1M"
        if out_dir is not None:
            save_name = os.path.join(exp_dir, out_dir, output_name_prefix + "_1M_" + range_string + ".dat")
        else:
            save_name = os.path.join(exp_dir, output_name_prefix + "_1M_" + range_string + ".dat")

    elif len(smp_name) > 1:
        if out_dir is not None:
            save_name = os.path.join(exp_dir, out_dir, output_name_prefix + "_1M_MIX.dat")
        else:
            save_name = os.path.join(exp_dir, output_name_prefix + "_1M_MIX.dat")
    # else:
    #     if len(smp_name) == 1:
    #         range_string = createRangeString(frames[0])
    #         output_name_prefix = "1M_" + smp_name[0]
    #         if out_dir is not None:
    #             save_name = os.path.join(exp_dir, out_dir, output_name_prefix + "_" + range_string + ".dat")
    #         else:
    #             save_name = os.path.join(exp_dir, output_name_prefix + "_" + range_string + ".dat")
    #     elif len(smp_name) > 1:
    #         output_name_prefix = smp_name[0]
    #         for index, item in enumerate(smp_name): 
    #             if index != 0:
    #                 output_name_prefix = output_name_prefix + "_" + item
    #         output_name_prefix = output_name_prefix + "_"
    #         if out_dir is not None:
    #             save_name = os.path.join(exp_dir, out_dir, output_name_prefix + "1M_MIX.dat")
    #         else:
    #             save_name = os.path.join(exp_dir, output_name_prefix + "_1M_MIX.dat")
    return save_name

def integrate_scale(avList, civi, rigi, thickness, TM, scale, 
use_rigi, tot_num_Frames, maskData, img, ai, num_points, verbose, img_series_name):
    
    use_list = avList # make function for both average frames and no average. 
    num_av = 0
    numFrames = 0
    total_use_Frames = 0
    if use_list is None:
        average_frames = False
        use_list = []
        for n in tot_num_Frames:
            use_list.append([*range(n)])

        IAll = []
        errAll = []
    else:
        average_frames = True
    for list in use_list:
        total_use_Frames = total_use_Frames + len(list)
    cur_frame_count = 1
    for index, item in enumerate(img):
        numFrames = numFrames + len(use_list[index])
        for cur_no, frame in enumerate(use_list[index]):
            normValue = calc_norm(
                civi[index][frame], rigi[index][frame], thickness, float(TM[index]), scale, use_rigi)
            if tot_num_Frames[index] > 1:
                current_frame = img[index].getframe(frame).data
                
            elif tot_num_Frames[index] == 1:
                current_frame = img[index].data

            q, I, err = ai.integrate1d(current_frame,
                num_points, error_model="poisson", mask=maskData, unit='q_A^-1', normalization_factor=normValue)
            if average_frames:
                if num_av == 0:
                    I_sum = I
                    err_sum = np.power(err, 2)
                elif num_av > 0:
                    I_sum = np.add(I_sum, I)
                    err_sum = np.add(err_sum, np.power(err, 2))
                num_av += 1
            else: IAll.append(I), errAll.append(err)

            if verbose is True:
                
                print("    Frame ID number: " + str(frame) +
                            ". Processed image " + str(cur_no+1) + "/" + str(len(use_list[index])) + " in " + img_series_name[index] + "_master.h5 - " + str(cur_frame_count) + "/" + str(total_use_Frames) + " total integrated.")
            cur_frame_count += 1

    if average_frames:
        if verbose: print("Dividing by " + str(numFrames) + " for average.")
        IMean = np.divide(I_sum, numFrames)
        errMean = np.divide(np.sqrt(err_sum), numFrames)
        if verbose is True: print("Averaging frames is complete.")
        return q, IMean, errMean 

    
    else:
        if verbose: print("Returning all 1D profiles from h5 file(s).")
        return q, IAll, errAll

def all_frame_list(tot_num_Frames):
    all_frames = []
    for frame in tot_num_Frames: all_frames.append([*range(frame)])
    return all_frames


def integrate(ai, verbose, FIT2dParams, **dat):
    """
        This part performs the processing of 1D frames and logic to output the 
        data in a sensible way

    """

    imgSmp = []
    tot_num_Frames = []

    for index, item in enumerate(dat["smp_name"]):
        imgSmp.append(fabio.open(os.path.join(dat["exp_dir"], item + '_master.h5'))) 
        if index == 0: mask_info = fabio.open(os.path.join(dat["exp_dir"], item + '_master.h5'))
        tot_num_Frames.append(imgSmp[-1].nframes)
        
        if tot_num_Frames == 0:
            print("number of frames 0!!! There is a problem with the")

        if verbose is True:
            print("Found ", str(tot_num_Frames[-1]), " frames in ",
                os.path.join(dat["exp_dir"], item + '_master.h5'))

    header_string = make_header_string(
        FIT2dParams, dat)  # make header string

    # masked pixels are 1 unmasked pixels are 0
    eiger_mask = make_Eiger_mask(mask_info)
    
    if verbose: print("Eiger mask (pixels with value = 4294967295 masked) applied.")

    if dat["mask"] is not None:
        user_mask = fabio.open(os.path.join(dat["exp_dir"], dat["mask"])).data
        if verbose: print("Mask file " + dat["mask"] + " created.")
    else:
        user_mask = np.zeros(eiger_mask.shape) # user masks is all 0
    
    if dat["reject_mask"] is not None:
        reject_data = import_reject_mask(dat["exp_dir"], dat["reject_mask"])
        reject_mask = make_reject_mask(np.zeros(eiger_mask.shape), reject_data)
        if verbose: print("reject mask created")
    else:
        reject_mask = np.zeros(eiger_mask.shape)

    if dat["counting_mask"] is not None:
        counting_data = import_reject_mask(dat["exp_dir"], dat["counting_mask"])
        counting_mask = make_counting_mask(np.zeros(eiger_mask.shape), counting_data)
        if verbose: print("counting mask created")
    else:
        counting_mask = np.zeros(eiger_mask.shape)

    mask_data = combine_masks(eiger_mask, user_mask, reject_mask, counting_mask)

    if verbose: print("Combined masks.")

    if dat["average_smp_all"] is True:
        dat['avList_smp'] = all_frame_list(tot_num_Frames) # get list of frames to average over in each file, a bit strange updating input dictionary

    # this next part is very ugly! Need to update and remove repeated code etc

    # average over some number of frames TIDY BY COMBING ALL AND SOME (FUTURE WORK)
    # elif average_smp_all is False and avList_smp is not None:
   
    ########################################################################
    # pyFAI integration options and vars
    ########################################################################
    # correctSolidAngle=True      # default
    # variance=None               # default
    # error_model="poisson"
    # radial_range=None           # default
    # azimuth_range=None          # default
    # mask=mask_data
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

    if verbose is True: print("Starting integration of sample frames, completed:")

    if dat['avList_smp'] is not None:
        q, IMeanSmp, errMeanSmp = integrate_scale(dat['avList_smp'], dat["civiSMP"], dat["rigiSMP"], dat["thickness"], dat["TM_smp"], dat["scale"], 
        dat["use_rigi"], tot_num_Frames, mask_data, imgSmp, ai, dat["num_points"], verbose, dat["smp_name"])

        # If no subtract background then directly save
        if dat['bkg_name'] is None:
            # save data to .dat file
            if dat['out_dir'] is not None:
                make_save_dir(dat["exp_dir"], dat['out_dir'])
            save_name = make_save_filename(dat["exp_dir"], dat['output_name_prefix'], dat['out_dir'], dat["smp_name"], dat['avList_smp'])
            write_1M_dat_file(save_name, header_string, np.transpose([q, IMeanSmp, errMeanSmp]))
            if verbose is True:
                print("    Sample text file saved to " + save_name + " no background set, finished.")
            return

    # no average
    # elif average_smp_all == False and avList_smp is None:

    if dat['avList_smp'] is None:
        q, IAll, errAll = integrate_scale(dat['avList_smp'], dat["civiSMP"], dat["rigiSMP"], dat["thickness"], dat["TM_smp"], dat["scale"], 
        dat["use_rigi"], tot_num_Frames, mask_data, imgSmp, ai, dat["num_points"], verbose, dat["smp_name"])  # get one 1D profile with correct scale
    
        # NEED TO condense saving!!!
        if dat['bkg_name'] is None:
            #Check if save dir exists and make if not
            if dat['out_dir'] is not None:
                make_save_dir(dat["exp_dir"], dat['out_dir'])
            for index, item in enumerate(dat["smp_name"]):     
                if verbose: print("Saving Files from " + item)
                frame_count = 0

                for frame in range(tot_num_Frames[index]):
                    save_name = make_save_filename(dat["exp_dir"], dat['output_name_prefix'], dat['out_dir'], [dat["smp_name"][index]], [[frame]])
                    write_1M_dat_file(save_name, header_string, np.transpose([q, IAll[frame_count], errAll[frame_count]]))
                    frame_count += 1
                    if verbose is True: print("    Saved frame ", str(frame), "  to: ", save_name)
            return

    if dat['bkg_name'] is not None:
        # get background image data
        imgBkg = []
        tot_num_Frames_bkg = []
        for item in dat['bkg_name']:
            imgBkg.append( fabio.open( os.path.join(dat["exp_dir"], item + '_master.h5')) )
            tot_num_Frames_bkg.append( imgBkg[-1].nframes )
        if dat["average_bkg_all"] is True:
            dat["avList_bkg"] = []
            for item in tot_num_Frames_bkg:
                dat["avList_bkg"].append([*range(item)])

        if dat["avList_bkg"] is not None:
            if verbose is True: print("Starting integration of background frames, completed:")
            q, IMeanBkg, errMeanBkg = integrate_scale(dat["avList_bkg"], dat["civiBKG"], dat["rigiBKG"], dat["thickness"], dat["TM_bkg"], dat["scale"], 
            dat["use_rigi"], tot_num_Frames_bkg, mask_data, imgBkg, ai, dat["num_points"], verbose, dat['bkg_name'])          

        # subtract Background from sample and save data
        if dat['avList_smp'] is not None:  # case 1
            ISubd = np.subtract(IMeanSmp, IMeanBkg)
            # adding error = sqrt(err1^2 + err2^2)
            errSubd = np.sqrt(
                np.add(np.power(errMeanSmp, 2), np.power(errMeanBkg, 2)))
            # save data
            if dat['out_dir'] is not None: # make save dir
                make_save_dir(dat["exp_dir"], dat['out_dir'])
            save_name = make_save_filename(dat["exp_dir"], dat["output_name_prefix"], dat['out_dir'], dat["smp_name"], dat['avList_smp'])
            write_1M_dat_file(save_name, header_string, np.transpose([q, ISubd, errSubd]))

            if verbose is True:
                print("    Subtracted data saved to " +
                      save_name)

        # NEED TO FIX IN THE AM!!! 
        # save individual frames
        elif dat['avList_smp'] is None:  # case ii
            if dat['out_dir'] is not None: # make save dir
                make_save_dir(dat["exp_dir"], dat['out_dir'])

            for index, item in enumerate(dat["smp_name"]):     
                if verbose: print("Saving files from " + item)
                frame_count = 0
                for frame in range(tot_num_Frames[index]):
                    save_name = make_save_filename(dat["exp_dir"], dat['output_name_prefix'], dat['out_dir'], [item], [[frame]])
                    ISubd = np.subtract(IAll[frame], IMeanBkg)
                    errSubd = np.sqrt(np.add(np.power(errAll[frame_count], 2), np.power(errMeanBkg, 2)))
                    write_1M_dat_file(save_name, header_string, np.transpose([q, ISubd, errSubd]))
                    frame_count += 1
                    if verbose is True: print("    Saved frame ", str(frame), "  to: ", save_name)

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

def createRangeString(zones):
    """
    This does the revers of the hyphen range code. Makes the hyphen range list from the full number list.
    """
    buffer = []
    try:
        st = ed = zones[0]
        for i in zones[1:]:
            delta = i - ed
            if delta == 1: ed = i
            elif not (delta == 0):
                buffer.append((st, ed))
                st = ed = i
        else: buffer.append((st, ed))
    except IndexError:
        pass

    return ','.join(
            "%d" % st if st==ed else "%d-%d" % (st, ed)
            for st, ed in buffer)


def run_parser():
    parser = argparse.ArgumentParser(
        description='Reduction program for TPS 13A 1M detector')

    parser.add_argument('-ed', '--exp_dir',
                        action='store', type=str, default=None,
                        help='Directory of experiment, if not set exp_dir == current working directory. (default: %(default)s)')
                        
    parser.add_argument('-od', '--out_dir',
                        action='store', type=str, default=None,
                        help='Directory to write data to inside experiment directory, data will be saved to experiment directory. (default: %(default)s)')

    parser.add_argument('-sn', '--smp_name',
                        action='store', default=None, nargs='+',
                        help='3 characters for sample file letter first. Can use multiple file names space seperated. (default: %(default)s)')

    parser.add_argument('-bn', '--bkg_name',
                        action='store', default=None, nargs='+',
                        help='3 characters for bkground file letter first. Can use multiple file names space seperated. (default: %(default)s)')

    parser.add_argument('-op', '--output_name_prefix',
                        action='store', type=str, default=None,
                        help='Prefix for output save name. Save format for single input file: (output_name_prefix)_1M_(frame_selection).dat Save format for mulitple input file (output_name_prefix)_1M_MIX.dat (default: %(default)s)')

    parser.add_argument('-ts', '--TMsmp',
                        action='store', default=[1.0], nargs='+',
                        help='Value of sample transmission. One for each sample file. (default: %(default)s)')

    parser.add_argument('-tb', '--TMbkg',
                        action='store', default= [1.0], nargs='+',
                        help='Value of background transmission. One for each sample file. (default: %(default)s)')

    parser.add_argument('-s', '--scale',
                        action='store', type=float, default=1.0,
                        help='Value to scale (multiply) data by. (default: %(default)s)')

    parser.add_argument('-as', '--avg_smp',
                        action='store', default= ['all'], nargs='+',
                        help='frames to average, all, none, a (-) and (,) seperated list. One list for each sample file, space separated. (default: %(default)s)')

    parser.add_argument('-ab', '--avg_bkg',
                        action='store', default=['all'], nargs='+',
                        help='frames to average, all or a (-) and (,) seperated list. One list for each sample file, space separated. (default: %(default)s)')

    parser.add_argument('-t', '--thickness',
                        action='store', type=float, default=1.0,
                        help='Sample thickness [mm]. (default: %(default)s)')

    parser.add_argument('-m', '--mask',
                        action='store', type=str, default=None,
                        help='User defined (user mask) 2d mask file name in exp dir, if none use eiger mask or reject mask. (default: %(default)s)')

    parser.add_argument('-rm', '--reject_mask',
                        action='store', type=str, default=None,
                        help='Reject mask file name, if none, user mask can be applied, or if no other mask Eiger mask is applied (default: %(default)s)')

    parser.add_argument('-cm', '--counting_mask',
                        action='store', type=str, default=None,
                        help='Counting mask file name, invese of reject mask, same file format although pixels in the file are included and those not in excluded (default: %(default)s)')

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
    if args.avg_smp[0] == 'none':
        exp_dic['average_smp_all'] = False
        exp_dic['avList_smp'] = None
    elif 'all' in args.avg_smp:
        if len(args.avg_smp) == 1:
            exp_dic['average_smp_all'] = True
            exp_dic['avList_smp'] = None
        else:
            print('ERROR: average all for sample must be a single entry.')
            return
    else:
        exp_dic['avList_smp'] = []
        for item in args.avg_smp:
            exp_dic['avList_smp'].append(hyphen_range(item))
        exp_dic['average_smp_all'] = False
    
    if args.bkg_name is not None:
        if 'all' in args.avg_bkg:
            if len(args.avg_bkg) == 1:
                exp_dic['average_bkg_all'] = True
            else:
                print('ERROR: average all for background must be a single entry.')
                return
        else:
            exp_dic['avList_bkg'] = []
            for item in args.avg_bkg:
                exp_dic['avList_bkg'].append(hyphen_range(item))
            exp_dic['average_bkg_all'] = False

    exp_dic['out_dir'] = args.out_dir
    exp_dic['output_name_prefix'] = args.output_name_prefix
    exp_dic['TM_smp'] = args.TMsmp
    exp_dic['TM_bkg'] = args.TMbkg
    exp_dic['scale'] = args.scale
    # change to the correct order app bit
    exp_dic['smp_name'] = []
    for item in args.smp_name:
        exp_dic['smp_name'].append(item[1] + \
            item[2] + item[0])
    
    # change background to the correct order app bit
    if args.bkg_name is not None:
        exp_dic['bkg_name'] = []
        for item in args.bkg_name:
            exp_dic['bkg_name'].append( item[1] + \
                item[2] + item[0] )
    else:
        exp_dic['bkg_name'] = None
    
    exp_dic['exp_dir'] = args.exp_dir
    if exp_dic['exp_dir'] is None:
        exp_dic['exp_dir'] = os.getcwd()
    exp_dic['thickness'] = args.thickness
    exp_dic['mask'] = args.mask
    exp_dic['reject_mask'] = args.reject_mask
    exp_dic['counting_mask'] = args.counting_mask
    exp_dic['num_points'] = args.num_points

    # get sample transmission information
    for index, smpName in enumerate(exp_dic['smp_name']):
        if index == 0:
            exp_dic['civiSMP'] = []
            exp_dic['rigiSMP'] = []
            exp_dic['expSMP'] = []
        civi, rigi, expSMP = readHeaderFile(exp_dic['exp_dir'], smpName)
        exp_dic['civiSMP'].append(civi)
        exp_dic['rigiSMP'].append(rigi)
        exp_dic['expSMP'].append(expSMP)
    
    exp_dic['use_rigi'] = False
    for civi_values in exp_dic['civiSMP']:
        if min(civi_values) == 0:
            exp_dic['use_rigi'] = True
    
    # get background transmission information
    if exp_dic['bkg_name'] is not None:
        for index, bkgName in enumerate(exp_dic['bkg_name']):
            if index == 0:
                exp_dic['civiBKG'] = []
                exp_dic['rigiBKG'] = []
                exp_dic['expBKG'] = []
            civi, rigi, expSMP = readHeaderFile(exp_dic['exp_dir'], bkgName)
            exp_dic['civiBKG'].append(civi)
            exp_dic['rigiBKG'].append(rigi)
            exp_dic['expBKG'].append(expSMP)
        
        for civi_values in exp_dic['civiBKG']:
            if min(civi_values) == 0:
                exp_dic['use_rigi'] = True
    
    
    verbose = args.verbose
    logfile = args.logfile

    # Check that the number of sample files, average list and TM are the same length
    if exp_dic['avList_smp'] is not None:
        if len(exp_dic['avList_smp']) < len(exp_dic['smp_name']):
            print("Error!! " + str(len(exp_dic['smp_name'])) + " sample file names found and is not the same as the " + str(len(exp_dic['avList_smp'])) + " average sample input")
            return

    if len(exp_dic['TM_smp']) != len(exp_dic['smp_name']):
        print("Warning!! " + str(len(exp_dic['smp_name'])) + " sample file names found and is not the same as the " + str(len(exp_dic['TM_smp'])) + " transmission sample input")
        if len(exp_dic['TM_smp']) < len(exp_dic['smp_name']):
            print("setting the same smp TM for remaining input files the same as first, please be careful here!")
            #for index in range(len(exp_dic['TM_smp'])):
            while len(exp_dic['TM_smp']) < len(exp_dic['smp_name']):
                exp_dic['TM_smp'].append(exp_dic['TM_smp'][0])
        else:
            print("TM_smp larger than length sample name")

    # Check that the number of Background files, average list and TM are the same length
    if exp_dic['bkg_name'] is not None:

        if len(exp_dic['TM_bkg']) != len(exp_dic['bkg_name']):
            print("Warning!! " + str(len(exp_dic['bkg_name'])) + " background file names found and is not the same as the " + str(len(exp_dic['TM_bkg'])) + " transmission background input")
            if len(exp_dic['TM_bkg']) < len(exp_dic['bkg_name']):
                print("setting the same BKG TM for all input files, please be careful here!")
                #for index in range(len(exp_dic['bkg_name'])):
                while len(exp_dic['TM_bkg']) < len(exp_dic['bkg_name']):
                    exp_dic['TM_bkg'].append(exp_dic['TM_bkg'][0])
            else:
                print("TM_bkg larger than length background name")

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
    release_date = "9th March 2022"
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
        
    integrate(ai, verbose, FIT2dParams, **expInfo)
    if not verbose:
        print("finished")


# Run if executed, but not if it is imported from other python script
if __name__ == "__main__":
    main()
