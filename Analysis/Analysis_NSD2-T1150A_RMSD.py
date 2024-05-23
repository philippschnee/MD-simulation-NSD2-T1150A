from __future__ import print_function
import mdtraj as mdt
import os
import re
import numpy as np
import collections
import pandas as pd

peptide = 'H3K36'
methylation_state = '2' # methylation state of peptide
number_replicates = 1  # number of replicates to analyze
sim_time = '100ns'
Variant = 'T1150A'      # or WT 

# load trajectory
traj_dict = {}
for i in range(number_replicates):
    i=i+0
    folder = ('/home/philipp/test_NSD2_T1150A/test_sMD_NSD2_T1150A/production_NSD2_{}_{}me{}_{}_{}.h5'.format(Variant, peptide, methylation_state, sim_time, i+1))
    traj_dict[i+1] = mdt.load(folder)

traj_list = []
for key in traj_dict:
    traj_list.append(traj_dict[key])

traj_pre = mdt.join(traj_list, check_topology=True, discard_overlapping_frames=True)
traj = traj_pre.superpose(traj_pre,frame=0,parallel=True)
topology = traj.topology
print('Trajectories successfully loaded, joined and superposed')

# load reference
reference = mdt.load('/home/philipp/test_NSD2_T1150A/test_sMD_NSD2_T1150A/production_NSD2_T1150A_H3K36me2_100ns_1.h5')
ref_topology = reference.topology
print('reference successfully loaded')

# select SAM for RMSD calculation
table, bonds = topology.to_dataframe()
df = pd.DataFrame(table)

# in the final output the RMSD of each atom from the following selection will be shown. In this case, SAM hast 49 atoms as seen in the PDB. The RMSD excel will therefore have 49 rows.
df_search = df[(df['resName'] == 'SAM')]
SAM_ind = df_search.index.tolist()
SAM = traj.atom_slice(SAM_ind)
ref_SAM = reference.atom_slice(SAM_ind)

# calculate RMSD
rmsd = mdt.rmsd(SAM, ref_SAM, frame=1) # specify here the frame of the reference trajectory, to which the rmsd should be calcualted to
rmsd = rmsd.tolist()

#export rmsd values in excel
df = pd.DataFrame(rmsd)
df.to_excel('RMSD.xlsx')



