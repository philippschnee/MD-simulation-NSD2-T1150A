from __future__ import print_function
import mdtraj as mdt
import os, shutil
import re
import numpy as np
from numpy import array
import collections
from collections import defaultdict
import pandas as pd
import itertools

# input parameters

peptide = 'H3K36'
sim_time = '100ns'
Variant = 'T1150A' # or WT

number_replicates = 1
count = 1

methylation_state = '2'
if methylation_state == '0':
    K36 = 'LYN'
if methylation_state == '1':
    K36 = 'MLZ'
if methylation_state == '2':
    K36 = 'MLY'
if methylation_state == '3':
    K36 = 'M3L'

# load trajectory and topology
while (count <= number_replicates):
 traj = mdt.load('/home/philipp/test_NSD2_T1150A/test_sMD_NSD2_T1150A/production_NSD2_{}_{}me{}_{}_{}.h5'.format(Variant, peptide, methylation_state, sim_time, count))
 topology=traj.topology

# calcualte SN2 Transition States
 table, bonds = topology.to_dataframe()
 df = pd.DataFrame(table)
 
 df_search = df[(df['chainID'] == 1) & (df['resName'] == '{}'.format(K36)) & (df['resSeq'] == 36) & (df['name'] == 'CE')]
 K36_CE = df_search.index.item()
 df_search = df[(df['chainID'] == 1) & (df['resName'] == '{}'.format(K36)) & (df['resSeq'] == 36) & (df['name'] == 'NZ')]
 K36_NZ = df_search.index.item()
 df_search = df[(df['chainID'] == 2) & (df['resName'] == 'SAM') & (df['resSeq'] == 1804) & (df['name'] == 'N')]
 SAM_N = df_search.index.item()
 df_search = df[(df['chainID'] == 2) & (df['resName'] == 'SAM') & (df['resSeq'] == 1804) & (df['name'] == 'SD')]
 SAM_SD = df_search.index.item()
 df_search = df[(df['chainID'] == 2) & (df['resName'] == 'SAM') & (df['resSeq'] == 1804) & (df['name'] == 'CE')]
 SAM_CE = df_search.index.item()
 
 NSD2_dist = [[K36_NZ,SAM_CE],[K36_NZ,SAM_N]]
 NSD2_109 = [[K36_CE, K36_NZ, SAM_CE]]
 NSD2_linear = [[SAM_SD, SAM_CE, K36_NZ]]
 
 dist_NAC = mdt.compute_distances(traj,NSD2_dist, periodic=True, opt=True)
 angle_109 = mdt.compute_angles(traj, array(NSD2_109), periodic=True, opt=True)
 angle_linear = mdt.compute_angles(traj, array(NSD2_linear), periodic=True, opt=True)

 # write all distances in a list but only from first atom pair
 dist_NAC_list = np.ndarray.tolist(dist_NAC)
 list_dist_NAC_merged = list(itertools.chain.from_iterable(dist_NAC_list))
 NAC = list_dist_NAC_merged[::2]

 # tranform all dihedrals (BB, SC) from rad in degrees and write them in a list
 angle_109=np.rad2deg(angle_109)
 angle_linear=np.rad2deg(angle_linear)
 
 angle_109_list=np.ndarray.tolist(angle_109)
 angle_linear_list=np.ndarray.tolist(angle_linear)
 angle_109_list_merged = list(itertools.chain.from_iterable(angle_109_list))
 angle_linear_list_merged = list(itertools.chain.from_iterable(angle_linear_list))
 
 # criteria lists in array
 criteria_array = np.column_stack([NAC,  angle_linear_list_merged, angle_109_list_merged])
 df = pd.DataFrame(criteria_array)
 df_sliced = df[(df[0] <0.4) & (df[1] >150) & (df[1] <210) & (df[2] >79) & (df[2] <139)] #+-30°, 4.0 A distance NZ-CE, 150°-210° angle_linear, 79°-139° angle_109
 
 indices = df_sliced.index.array
 indices_df = pd.DataFrame(indices)
 temp = indices_df.values.tolist()
 INT = list(itertools.chain.from_iterable(temp))

 print('number of SN2 frames in NSD2 {} {}me{} in replicate {}:'.format(Variant, peptide, methylation_state, count), len(INT))
 
 count = count+1
