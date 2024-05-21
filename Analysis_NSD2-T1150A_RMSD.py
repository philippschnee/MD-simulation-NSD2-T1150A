from __future__ import print_function
import mdtraj as mdt
import os
import re
import numpy as np
import collections
import pandas as pd

peptide = 'H3K36'
methylation_state = '2' # methylation state of peptide
number_replicates = 10  # number of replicates to analyze

# load trajectory
traj_dict = {}
for i in range(number_replicates):
    i=i+0
    folder = ('/path_to_trajectory/production_NSD2_{}_{}_{}ns_{}.h5'.format(peptide, methylation_state, sim_time, i+1))
    traj_dict[i+1] = mdt.load(folder)

traj_list = []
for key in traj_dict:
    traj_list.append(traj_dict[key])

traj_pre = mdt.join(traj_list, check_topology=True, discard_overlapping_frames=True)
traj = traj_pre.superpose(traj_pre,frame=0,parallel=True)
topology = traj.topology
print('Trajectories successfully loaded, joined and superposed')

# load reference
reference = mdt.load('/path_to_reference/reference.h5')
ref_topology = reference.topology
print('reference successfully loaded')

# select SAM for RMSD calculation
table, bonds = topology.to_dataframe()
df = pd.DataFrame(table)

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



