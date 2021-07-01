"""
Script for comparing pipapr against TeraStitcher (single and multicore with OPENMPI).

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import pipapr
from time import time
import os
from pathlib import Path
import numpy as np
from skimage.io import imsave
import pyapr

def create_datasets():
    # Load data
    data = np.fromfile(file_path, dtype='uint16', count=-1)
    data = data.reshape((-1, 2048, 2048))
    Path(output_folder_apr).mkdir(parents=True, exist_ok=True)

    dv = int(data.shape[1] * (1 - overlap_V / 100) / dV)
    dh = int(data.shape[2] * (1 - overlap_H / 100) / dH)

    def get_coordinates(v, dV, h, dH):
        x = int(v * dV)
        y = int(h * dH)
        x_noise = int(max(v * dV + np.random.randn(1) * 5, 0))
        y_noise = int(max(h * dH + np.random.randn(1) * 5, 0))
        return (x, y, x_noise, y_noise)

    # Save data as separate tiles
    noise_coordinates = np.zeros((dH * dV, 4))
    for v in range(dV):
        for h in range(dH):
            (x, y, x_noise, y_noise) = get_coordinates(v, dv, h, dh)

            if h == 0:
                row_folder_tiff = os.path.join(output_folder_tiff, '{:06d}'.format(10 * x))
                Path(row_folder_tiff).mkdir(parents=True, exist_ok=True)

            depth_folder_tiff = os.path.join(row_folder_tiff, '{:06d}_{:06d}'.format(10 * x, 10 * y))
            Path(depth_folder_tiff).mkdir(parents=True, exist_ok=True)
            noise_coordinates[v * dH + h, :] = [x_noise, y_noise, x_noise + int(data.shape[1] / dV),
                                                y_noise + int(data.shape[2] / dH)]
            imsave(os.path.join(depth_folder_tiff, '000000.tif'),
                   data[:, x_noise:x_noise + int(data.shape[1] / dV), y_noise:y_noise + int(data.shape[2] / dH)],
                   check_contrast=False)

            # Parameters for PV_interneurons
            par = pyapr.APRParameters()
            par.auto_parameters = True
            par.Ip_th = 200.0
            par.rel_error = 0.3
            par.gradient_smoothing = 0.0

            # Convert data to APR
            apr, parts = pyapr.converter.get_apr(
                image=data[:, x_noise:x_noise + int(data.shape[1] / dV), y_noise:y_noise + int(data.shape[1] / dV)]
                , params=par, verbose=False)
            pyapr.io.write(os.path.join(output_folder_apr, '{}_{}.apr'.format(v, h)), apr, parts)

    # Save coordinates
    np.savetxt(os.path.join(output_folder_tiff, 'real_displacements.csv'), noise_coordinates, fmt='%1.d',
               delimiter=',')
    np.savetxt(os.path.join(output_folder_apr, 'real_displacements.csv'), noise_coordinates, fmt='%1.d',
               delimiter=',')


def create_multiple_CR_datasets(path):
    # Load data
    data = np.fromfile(file_path, dtype='uint16', count=-1)
    data = data.reshape((-1, 2048, 2048))


    for rel_error in [0.1, 0.2, 0.4, 0.8]:
        for Ip_th in [200, 300, 400]:

            output_folder_apr = os.path.join(path, '{}_{}'.format(Ip_th, rel_error))
            Path(output_folder_apr).mkdir(parents=True, exist_ok=True)

            dv = int(data.shape[1] * (1 - overlap_V / 100) / dV)
            dh = int(data.shape[2] * (1 - overlap_H / 100) / dH)

            def get_coordinates(v, dV, h, dH):
                x = int(v * dV)
                y = int(h * dH)
                x_noise = int(max(v * dV + np.random.randn(1) * 5, 0))
                y_noise = int(max(h * dH + np.random.randn(1) * 5, 0))
                return (x, y, x_noise, y_noise)

            # Save data as separate tiles
            noise_coordinates = np.zeros((dH * dV, 4))
            for v in range(dV):
                for h in range(dH):
                    (x, y, x_noise, y_noise) = get_coordinates(v, dv, h, dh)
                    noise_coordinates[v * dH + h, :] = [x_noise, y_noise, x_noise + int(data.shape[1] / dV),
                                                        y_noise + int(data.shape[2] / dH)]

                    # Parameters for PV_interneurons
                    par = pyapr.APRParameters()
                    par.auto_parameters = True
                    par.Ip_th = Ip_th
                    par.rel_error = rel_error
                    par.gradient_smoothing = 0.0

                    # Convert data to APR
                    apr, parts = pyapr.converter.get_apr(
                        image=data[:, x_noise:x_noise + int(data.shape[1] / dV), y_noise:y_noise + int(data.shape[1] / dV)]
                        , params=par, verbose=False)
                    pyapr.io.write(os.path.join(output_folder_apr, '{}_{}.apr'.format(v, h)), apr, parts)

            # Save coordinates
            np.savetxt(os.path.join(output_folder_apr, 'real_displacements.csv'), noise_coordinates, fmt='%1.d',
                       delimiter=',')


def parse_xml(xml_path):
    """
    This function allows to parse the data generated by TeraStitcher Place step (step 5).
    """

    import xml.etree.ElementTree as ET
    import pandas as pd

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Iterate over Stack nodes
    stitch_par = {'ABS_V': [],
                  'ABS_H': [],
                  'ABS_D': [],
                  'COL': [],
                  'ROW': [],
                  'DIR_NAME': []}

    for stack in root.iter('Stack'):
        for key in stitch_par.keys():
            stitch_par[key].append(stack.attrib[key])

    stitch_par = pd.DataFrame.from_dict(stitch_par)

    return stitch_par[['ABS_H', 'ABS_V', 'ABS_D']].to_numpy().astype('int16')


# Parameters
file_path = r'/run/user/1000/gvfs/smb-share:server=fcbgnasc.campusbiotech.ch,share=fcbgdata/0063_CBT_UNIGE_LAMY/Christophe/Mouse_Brain/200831_mouse iDISCO_FluoNissl_Sara/test_sample_from_DBE_in_ECI_20200825/647ECI.raw'
output_folder_apr = r'/home/apr-benchmark/Desktop/data/synthetic/ncells_128'
# output_folder_apr = r'/home/apr-benchmark/Desktop/data/sarah/multitile_apr/400Ip_0.8error'
dH = 4
dV = 4
overlap_H = 25
overlap_V = 25


# FIRST WE CREATE THE DATASET
# create_datasets()

# Parse data
tiles = pipapr.parser.tileParser(output_folder_apr, frame_size=512, overlap=128, ftype='apr')

# Stitch tiles
stitcher = pipapr.stitcher.tileStitcher(tiles)
t = time()
stitcher.compute_registration_fast()
print('Elapsed time with APR: {} s.'.format(time()-t))

# # Run Terastitcher from python (terastitcher folder must be in system path).
# t = time()
# # Do step 1 (load data and create XML)
# str_step1 = 'terastitcher --import --volin="{}" --ref1=y --ref2=x --ref3=z --vxl1=1 --vxl2=1 --vxl3=1 --volin_plugin="TiledXY|3Dseries"'.format(output_folder_tiff)
# os.system(str_step1)
# # Do step 2 (align)
# str_step2 = 'terastitcher --displcompute --projin="{}/xml_import.xml" --subvoldim=1200'.format(output_folder_tiff)
# os.system(str_step2)
# # Do step 3 (project)
# str_step3 = 'terastitcher --displproj --projin="{}/xml_displcomp.xml"'.format(output_folder_tiff)
# os.system(str_step3)
# # Do step 4 (threshold)
# str_step4 = 'terastitcher --displthres --projin="{}/xml_displproj.xml" --threshold=0.7'.format(output_folder_tiff)
# os.system(str_step4)
# # Do step 5 (place)
# str_step5 = 'terastitcher --placetiles --projin="{}/xml_displthres.xml"'.format(output_folder_tiff)
# os.system(str_step5)
# print('\n\nElapsed time: {} s.'.format(time()-t))

# Run Terastitcher multi-core from python
# Set cwd to terastitcher exe folder
# n_cores = [2, 4, 8, 16, 32, 48]
# time_parastitcher = []
# for n in n_cores:
#     t = time()
#     # Do step 1 (load data and create XML)
#     str_step1 = 'mpirun -np {} python ./parastitcher.py --import --volin="{}" --ref1=y --ref2=x --ref3=z --vxl1=1 --vxl2=1 --vxl3=1 --volin_plugin="TiledXY|3Dseries"'.format(n+1, output_folder_tiff)
#     os.system(str_step1)
#     # Do step 2 (align)
#     str_step2 = 'mpirun -np {} python ./parastitcher.py --displcompute --projin="{}/xml_import.xml" --subvoldim=1200'.format(n+1, output_folder_tiff)
#     os.system(str_step2)
#     # Do step 3 (project)
#     str_step3 = 'mpirun -np {} python ./parastitcher.py --displproj --projin="{}/xml_displcomp.xml"'.format(n+1, output_folder_tiff)
#     os.system(str_step3)
#     # Do step 4 (threshold)
#     str_step4 = 'mpirun -np {} python ./parastitcher.py --displthres --projin="{}/xml_displproj.xml" --threshold=0.7'.format(n+1, output_folder_tiff)
#     os.system(str_step4)
#     # Do step 5 (place)
#     str_step5 = 'mpirun -np {} python ./parastitcher.py --placetiles --projin="{}/xml_displthres.xml"'.format(n+1, output_folder_tiff)
#     os.system(str_step5)
#     time_parastitcher.append(time()-t)
#     print('\n\nElapsed time parallel: {} s.'.format(time()-t))


# # Compare results
# coords_apr = stitcher.database[['ABS_H', 'ABS_V', 'ABS_D']].to_numpy()
# coords_tera = parse_xml('/home/apr-benchmark/Desktop/data/sarah/multitile_tiff/xml_merging.xml')
# for i in range(3):
#     coords_tera[:, i] = coords_tera[:, i] - coords_tera[:, i].min()
#
# import pandas as pd
# coords_real = pd.read_csv('/home/apr-benchmark/Desktop/data/sarah/multitile_apr/real_displacements.csv', header=None).to_numpy()
# coords_real = coords_real[:, (1, 0)]
# coords_real = np.hstack((coords_real, np.zeros((16, 1))))

