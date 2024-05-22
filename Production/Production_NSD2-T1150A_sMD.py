from __future__ import print_function
from openmm.app import *
from openmm import *
import openmm as mm
from openmm.unit import *
from sys import stdout
from pdbfixer import PDBFixer
from mdtraj.reporters import HDF5Reporter
import mdtraj as mdt
import os
import re
import numpy as np

# file and folder variables, simulation variables specified below

Variant = 'T1150A'       # or WT
sim_time = '100ns'       # simulation time 
Eq = '1ns'               # equilibration
number_replicates = 1   # how many replicates will be produced
count = 1                # starting number of replicates
traj_folder = 'test_sMD_NSD2_T1150A'         # name of folder, where trajectories will be stored

while (count <= number_replicates):
 
 # Input Files
 
 pdb = PDBFile('NSD2-{}.pdb'.format(Variant))  # PDB File of NSD2
 protein = app.Modeller(pdb.topology, pdb.positions)
 sim_forcefield = ('amber14-all.xml')
 sim_watermodel = ('amber14/tip4pew.xml')
 sim_gaff = 'gaff.xml'
 
 methylation_state = '2' # define methylation state of K36; number between 0-3
 
 if methylation_state == '0':                        
     ligand_names = ['LYN', 'SAM', 'ZNB']
 if methylation_state == '1':
     ligand_names = ['MLZ', 'SAM', 'ZNB']
 if methylation_state == '2':                                
     ligand_names = ['MLY', 'SAM', 'ZNB']
 if methylation_state == '3':
     ligand_names = ['M3L', 'SAM', 'ZNB']
 
 ligand_xml_files = ['K36me{}_deprot.xml'.format(methylation_state), 'SAM.xml', 'ZNB.xml']           # list of ligand parameter xml files
 ligand_pdb_files = ['H3K36me{}-EQ.pdb'.format(methylation_state), 'SAM-sMD.pdb', 'ZNB-EQ.pdb']       # list of ligand pdb files
 
 
 
 # Integration Options
 
 dt = 0.002*picoseconds    # integration time-step
 temperature = 300*kelvin  # temperature
 friction = 1/picosecond   # friction
 sim_ph = 7.0              # pH
 
 # Simulation Options
 
 Simulate_Steps = 50000000      # production simulation time; 100ns
 
 npt_eq_Steps = 500000          # NPT equilibration; 1ns
 SAM_restr_eq_Steps = 500000    # SAM restrained equilibration; 1ns
 SAM_free_eq_Steps = 250000     # No restraints equilibration; 0.5ns
 
 restrained_eq_atoms = 'protein and chainid 1 name CA'   # MDTraj selection syntax; restrained backbone 
 force_eq_atoms = 50                                     # restraints in kilojoules_per_mole/unit.angstroms

 restrained_eq_atoms2 = 'resn SAM'                       # SAM
 force_eq_atoms2 = 2
 
 restrained_ligands = True #(TRUE|FALSE), no restraints ligands for protein only equilibration
 gpu_index = '0'   # specify which GPU is used
 platform = Platform.getPlatformByName('CUDA')
 platformProperties = {'Precision': 'single','DeviceIndex': gpu_index}
 trajectory_out_atoms = 'chainid 0 or chainid 1 or resname SAM or resname ZNB'  # atoms, which data are written in trajectory
 trajectory_out_interval = 10000                                                # steps at which data is written in trajectory
 
 protonation_dict = {('A',1015): 'CYX', ('A',1017): 'CYX', ('A',1025):'CYX', ('A',1031): 'CYX', ('A',1040): 'CYX', ('A',1045):'CYX', ('A',1051):'CYX', ('A',1144):'CYX', ('A',1191):'CYX', ('A',1193):'CYX', ('A',1198):'CYX'} #only for manual protonation
 # K36 deprotonation taken out, because specified in xml file
 
 # Prepare the Simulation
 
 os.system('mkdir {0}'.format(traj_folder))
 xml_list = [sim_forcefield, sim_gaff, sim_watermodel]
 for lig_xml_file in ligand_xml_files:
 	xml_list.append(lig_xml_file)
 forcefield = app.ForceField(*xml_list)
 protonation_list = []
 key_list=[]
 
 if len(protonation_dict.keys()) > 0:
     for chain in protein.topology.chains():
         chain_id = chain.id
         
         protonations_in_chain_dict = {}
         for protonation_tuple in protonation_dict:
             if chain_id == protonation_tuple[0]:
                 residue_number = protonation_tuple[1]
                 protonations_in_chain_dict[int(residue_number)] = protonation_dict[protonation_tuple]
                 key_list.append(int(residue_number))

         for residue in chain.residues():
            with open("log_prot.txt", "a") as myfile:
                residue_id = residue.id
                myfile.write(residue_id)
                if int(residue_id) in key_list:
                    myfile.write(': Protoniert!')
                    myfile.write(residue_id)
                    protonation_list.append(protonations_in_chain_dict[int(residue_id)])
                else:
                    protonation_list.append(None)
                    myfile.write('-')          
 protein.addHydrogens(forcefield, pH=sim_ph, variants = protonation_list)              
 
 # add ligand structures to the model
 for lig_pdb_file in ligand_pdb_files:
 	ligand_pdb = app.PDBFile(lig_pdb_file)
 	protein.add(ligand_pdb.topology, ligand_pdb.positions)
 
 # Generation and Solvation of Box
 print('Generation and Solvation of Box')
 boxtype = 'cubic' # ('cubic'|'rectangular')
 box_padding = 1.0 # nanometers
 x_list = []
 y_list = []
 z_list = []
 
 # get atom indices for protein plus ligands
 for index in range(len(protein.positions)):
 	x_list.append(protein.positions[index][0]._value)
 	y_list.append(protein.positions[index][1]._value)
 	z_list.append(protein.positions[index][2]._value)
 x_span = (max(x_list) - min(x_list))
 y_span = (max(y_list) - min(y_list))
 z_span = (max(z_list) - min(z_list))
 
 # build box and add solvent
 d =  max(x_span, y_span, z_span) + (2 * box_padding)
 
 d_x = x_span + (2 * box_padding)
 d_y = y_span + (2 * box_padding)
 d_z = z_span + (2 * box_padding)
 
 prot_x_mid = min(x_list) + (0.5 * x_span)
 prot_y_mid = min(y_list) + (0.5 * y_span)
 prot_z_mid = min(z_list) + (0.5 * z_span)
 
 box_x_mid = d_x * 0.5
 box_y_mid = d_y * 0.5
 box_z_mid = d_z * 0.5
 
 shift_x = box_x_mid - prot_x_mid
 shift_y = box_y_mid - prot_y_mid
 shift_z = box_z_mid - prot_z_mid
 
 solvated_protein = app.Modeller(protein.topology, protein.positions)
 
 # shift coordinates to the middle of the box
 for index in range(len(solvated_protein.positions)):
 	solvated_protein.positions[index] = (solvated_protein.positions[index][0]._value + shift_x, solvated_protein.positions[index][1]._value + shift_y, solvated_protein.positions[index][2]._value + shift_z)*nanometers
 
 # add box vectors and solvate
 if boxtype == 'cubic':
 	solvated_protein.addSolvent(forcefield, model='tip4pew', neutralize=True, ionicStrength=0.1*molar, boxVectors=(mm.Vec3(d, 0., 0.), mm.Vec3(0., d, 0.), mm.Vec3(0, 0, d)))
 elif boxtype == 'rectangular':
 	solvated_protein.addSolvent(forcefield, model='tip4pew', neutralize=True, ionicStrength=0.1*molar, boxVectors=(mm.Vec3(d_x, 0., 0.), mm.Vec3(0., d_y, 0.), mm.Vec3(0, 0, d_z)))
 
 
 # Building System
 
 print('Building system...')
 topology = solvated_protein.topology
 positions = solvated_protein.positions
 selection_reference_topology = mdt.Topology().from_openmm(solvated_protein.topology)
 trajectory_out_indices = selection_reference_topology.select(trajectory_out_atoms)
 restrained_eq_indices = selection_reference_topology.select(restrained_eq_atoms)
 restrained_eq_indices2 = selection_reference_topology.select(restrained_eq_atoms2)
 system = forcefield.createSystem(topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0*nanometers,ewaldErrorTolerance=0.0005, constraints=HBonds, rigidWater=True)
 integrator = LangevinIntegrator(temperature, friction, dt)
 simulation = Simulation(topology, system, integrator, platform, platformProperties)
 simulation.context.setPositions(positions)


 # Minimize
 
 print('Performing energy minimization...')
 print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
 app.PDBFile.writeFile(simulation.topology, positions, open(traj_folder + '/' + 'PreMin.pdb', 'w'), keepIds=True)
 simulation.minimizeEnergy()
 min_pos = simulation.context.getState(getPositions=True).getPositions()
 app.PDBFile.writeFile(simulation.topology, positions, open(traj_folder + '/' + 'PostMin.pdb', 'w'), keepIds=True)
 print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
 print('System is now minimized')
 
 
 # Restraints
 
 force = mm.CustomExternalForce("(k/2)*periodicdistance(x, y, z, x0, y0, z0)^2")
 force.addGlobalParameter("k", force_eq_atoms*kilojoules_per_mole/angstroms**2)
 force.addPerParticleParameter("x0")
 force.addPerParticleParameter("y0")
 force.addPerParticleParameter("z0")
 
 force2 = mm.CustomExternalForce("(k/2)*periodicdistance(x, y, z, x0, y0, z0)^2")
 force2.addGlobalParameter("k", force_eq_atoms2*kilojoules_per_mole/angstroms**2)
 force2.addPerParticleParameter("x0")
 force2.addPerParticleParameter("y0")
 force2.addPerParticleParameter("z0")
 
 
 if restrained_ligands:
     for res_atom_index in restrained_eq_indices2:
         force2.addParticle(int(res_atom_index), min_pos[int(res_atom_index)].value_in_unit(nanometers))
     system.addForce(force2)
 
 for res_atom_index in restrained_eq_indices:
 	force.addParticle(int(res_atom_index), min_pos[int(res_atom_index)].value_in_unit(nanometers))
 system.addForce(force)
 
 # NPT Equilibration
 
 # add barostat for NPT
 system.addForce(mm.MonteCarloBarostat(1*atmospheres, temperature, 25))
 simulation.context.setPositions(min_pos)
 simulation.context.setVelocitiesToTemperature(temperature)
 simulation.reporters.append(app.StateDataReporter(stdout, 10000, step=True, potentialEnergy=True, temperature=True, progress=True, remainingTime=True, speed=True, totalSteps=npt_eq_Steps, separator='\t'))
 simulation.reporters.append(HDF5Reporter(traj_folder + '/' + 'EQ_NPT.h5', 10000, atomSubset=trajectory_out_indices))
 print('restrained NPT equilibration...')
 simulation.step(npt_eq_Steps)
 state_npt_EQ = simulation.context.getState(getPositions=True, getVelocities=True)
 positions = state_npt_EQ.getPositions()
 app.PDBFile.writeFile(simulation.topology, positions, open(traj_folder + '/' + 'post_NPT_EQ.pdb', 'w'), keepIds=True)
 print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
 print('Successful NPT equilibration!')
 
 
 # Free Equilibration
 # forces: 0->HarmonicBondForce, 1->HarmonicAngleForce, 2->PeriodicTorsionForce, 3->NonbondedForce, 4->CMMotionRemover, 5->CustomExternalForce, 6->CustomExternalForce, 7->MonteCarloBarostat
 n_forces = len(system.getForces())
 system.removeForce(n_forces-2)
 print('force removed')
 
 # optional ligand restraint to force slight conformational changes
 if restrained_ligands:
     
     integrator = mm.LangevinIntegrator(temperature, 1/picosecond, 0.002*picoseconds)
     simulation = app.Simulation(solvated_protein.topology, system, integrator, platform, platformProperties)
     simulation.context.setState(state_npt_EQ)
     simulation.reporters.append(app.StateDataReporter(stdout, 10000, step=True, potentialEnergy=True, temperature=True, progress=True, remainingTime=True, speed=True, totalSteps=SAM_restr_eq_Steps, separator='\t'))
     simulation.reporters.append(HDF5Reporter(traj_folder + '/' + 'free_BB_restrained_SAM_NPT_EQ.h5', 10000, atomSubset=trajectory_out_indices))
     print('free BB NPT equilibration of protein with restrained SAM...')
     simulation.step(SAM_restr_eq_Steps)
     state_free_EQP = simulation.context.getState(getPositions=True, getVelocities=True)
     positions = state_free_EQP.getPositions()
     app.PDBFile.writeFile(simulation.topology, positions, open(traj_folder + '/' + 'free_BB_restrained_SAM_NPT_EQ.pdb', 'w'), keepIds=True)
     print('Successful free BB, SAM restrained equilibration!')
   
     # equilibration with free ligand   
     n_forces = len(system.getForces())
     system.removeForce(n_forces-2)
     integrator = mm.LangevinIntegrator(temperature, 1/picosecond, 0.002*picoseconds)
     simulation = app.Simulation(solvated_protein.topology, system, integrator, platform, platformProperties)
     simulation.context.setState(state_free_EQP)
     simulation.reporters.append(app.StateDataReporter(stdout, 10000, step=True, potentialEnergy=True, temperature=True, progress=True, remainingTime=True, speed=True, totalSteps=SAM_free_eq_Steps, separator='\t'))
     simulation.reporters.append(HDF5Reporter(traj_folder + '/' + 'SAM_free_NPT_EQ.h5', 10000, atomSubset=trajectory_out_indices))
     print('SAM free NPT equilibration...')
     simulation.step(SAM_free_eq_Steps)
     state_free_EQ = simulation.context.getState(getPositions=True, getVelocities=True)
     positions = state_free_EQ.getPositions()
     app.PDBFile.writeFile(simulation.topology, positions, open(traj_folder + '/' + 'SAM_free_NPT_EQ.pdb', 'w'), keepIds=True)
     print('Successful SAM free equilibration!')
     
 else:
     
     # remove ligand restraints for free equilibration (remove the second last force object, as the last one was the barostat)
     n_forces = len(system.getForces())
     system.removeForce(n_forces-2)
     simulation.context.setState(state_npt_EQ)
     simulation.reporters.append(app.StateDataReporter(stdout, 10000, step=True, potentialEnergy=True, temperature=True, progress=True, remainingTime=True, speed=True, totalSteps=SAM_free_eq_Steps, separator='\t'))
     simulation.reporters.append(HDF5Reporter(traj_folder + '/' + 'EQ_NPT_free.h5', 10000, atomSubset=trajectory_out_indices))
     print('free NPT equilibration...')
     simulation.step(SAM_free_eq_Steps)
     state_free_EQ = simulation.context.getState(getPositions=True, getVelocities=True)
     positions = state_free_EQ.getPositions()
     app.PDBFile.writeFile(simulation.topology, positions, open(traj_folder + '/' + 'free_NPT_EQ.pdb', 'w'), keepIds=True)
     print('Successful free equilibration!')
  
 
 # add steered MD pullforce to equilibrated system
 # indices of atoms to which spring is attached, further specified below
 
 if Variant == 'WT':
     K36_ind = 'index 3677 or index 3678 or index 3689'               
     SAM_ind = 'index 3836 or index 3866 or index 3867 or index 3868' 
     SAM_N1 = 'index 3847 or index 3848 or index 3849'
     L1202 = 'index 3293 or index 3294 or index 3295'
     SAM_N = 'index 3830 or index 3831 or index 3834'
     F1139 = 'index 2379 or index 2380 or index 2381'  
 

 if Variant == 'T1150A':
     K36_ind = 'index 3673 or index 3674 or index 3675'
     SAM_ind = 'index 3835 or index 3865 or index 3866 or index 3867'
     SAM_N1 = 'index 3846 or index 3847 or index 3848'
     L1202 = 'index 3289 or index 3291 or index 3293'
     SAM_N = 'index 3829 or index 3830 or index 3833'
     F1139 = 'index 2380 or index 2382 or index 2383'  
 
 # pullforce K36me2-SAM
 
 force_pullforce = 0.2
 pullforce_constant =  force_pullforce*kilojoules_per_mole/angstroms**2
 pull_atoms = K36_ind                                   # K36 (NZ,C01,C02)
 pull_target = SAM_ind                                  # SAM (CE,H10,H11,H12)
 pull_atoms_indices = selection_reference_topology.select(pull_atoms)
 pull_target_indices = selection_reference_topology.select(pull_target)
 g1 = pull_atoms_indices.tolist()
 g2 = pull_target_indices.tolist()
 
 pullforce = mm.CustomCentroidBondForce(2,'pullforce_constant*distance(g1,g2)')
 pullforce.addGlobalParameter('pullforce_constant',pullforce_constant)
 pullforce.addGroup(g1)
 pullforce.addGroup(g2)
 bondGroups = [0,1]
 system.addForce(pullforce)
 pullforce.addBond(bondGroups)

 # pullforce SAM-L1202
 
 force_pullforce2 = 0.1
 pullforce_constant2 =  force_pullforce2*kilojoules_per_mole/angstroms**2
 pull_atoms2 = SAM_N1                                   # SAM (N1,C2,N3)
 pull_target2 = L1202                                   # L1202 (N,CA,C)
 pull_atoms_indices2 = selection_reference_topology.select(pull_atoms2)
 pull_target_indices2 = selection_reference_topology.select(pull_target2)
 g1_2 = pull_atoms_indices2.tolist()
 g2_2 = pull_target_indices2.tolist()
 
 pullforce2 = mm.CustomCentroidBondForce(2,'pullforce_constant2*distance(g1,g2)')
 pullforce2.addGlobalParameter('pullforce_constant2',pullforce_constant2)
 pullforce2.addGroup(g1_2)
 pullforce2.addGroup(g2_2)
 bondGroups2 = [0,1]
 system.addForce(pullforce2)
 pullforce2.addBond(bondGroups2)
 
 # pullforce SAM-F1139
 
 force_pullforce3 = 0.05
 pullforce_constant3 =  force_pullforce3*kilojoules_per_mole/angstroms**2
 pull_atoms3 = SAM_N                                   # SAM (N, CA, CB)
 pull_target3 = SAM_ind                                # F1139 (CA, C, O)
 pull_atoms_indices3 = selection_reference_topology.select(pull_atoms3)
 pull_target_indices3 = selection_reference_topology.select(pull_target3)
 g1_3 = pull_atoms_indices3.tolist()
 g2_3 = pull_target_indices3.tolist()
 
 pullforce3 = mm.CustomCentroidBondForce(2,'pullforce_constant3*distance(g1,g2)')
 pullforce3.addGlobalParameter('pullforce_constant3',pullforce_constant3)
 pullforce3.addGroup(g1_3)
 pullforce3.addGroup(g2_3)
 bondGroups3 = [0,1]
 system.addForce(pullforce3)
 pullforce3.addBond(bondGroups3)

 if Variant == 'WT':
     print('pull K36 atoms', selection_reference_topology.atom(3677), selection_reference_topology.atom(3678), selection_reference_topology.atom(3689))
     print('SAM atoms', selection_reference_topology.atom(3836), selection_reference_topology.atom(3866), selection_reference_topology.atom(3867), selection_reference_topology.atom(3868))
     print('SAM_N1 atoms', selection_reference_topology.atom(3847), selection_reference_topology.atom(3848), selection_reference_topology.atom(3849))
     print('L1202 atoms', selection_reference_topology.atom(3293), selection_reference_topology.atom(3294), selection_reference_topology.atom(3295))
     print('SAM_N atoms', selection_reference_topology.atom(3830), selection_reference_topology.atom(3831), selection_reference_topology.atom(3834))
     print('F1139 atoms', selection_reference_topology.atom(2379), selection_reference_topology.atom(2380), selection_reference_topology.atom(2381))
 
 if Variant == 'T1150A':
     print('pull K36 atoms', selection_reference_topology.atom(3673), selection_reference_topology.atom(3674), selection_reference_topology.atom(3685))
     print('SAM atoms', selection_reference_topology.atom(3832), selection_reference_topology.atom(3862), selection_reference_topology.atom(3863), selection_reference_topology.atom(3864))
     print('SAM_N1 atoms', selection_reference_topology.atom(3843), selection_reference_topology.atom(3844), selection_reference_topology.atom(3845))
     print('L1202 atoms', selection_reference_topology.atom(3289), selection_reference_topology.atom(3291), selection_reference_topology.atom(3293))
     print('SAM_N atoms', selection_reference_topology.atom(3826), selection_reference_topology.atom(3827), selection_reference_topology.atom(3830))
     print('F1139 atoms', selection_reference_topology.atom(2380), selection_reference_topology.atom(2382), selection_reference_topology.atom(2383))
 
 state_steered = simulation.context.getState(getPositions=True, getVelocities=True, getForces=True)
 positions_steered = state_steered.getPositions()

 
 # Simulate
  
 print('Simulating...')
  
 # create new simulation object for production run with new integrator
 integrator = mm.LangevinIntegrator(temperature, 1/picosecond, 0.002*picoseconds)
 simulation = app.Simulation(solvated_protein.topology, system, integrator, platform, platformProperties)
 simulation.context.setState(state_free_EQ)
 simulation.reporters.append(app.StateDataReporter(stdout, 10000, step=True, potentialEnergy=True, temperature=True, progress=True, remainingTime=True, speed=True, totalSteps=Simulate_Steps, separator='\t'))
 simulation.reporters.append(HDF5Reporter(traj_folder + '/' + 'production_sMD_NSD2_{}_H3K36me{}_{}_{}.h5'.format(Variant,methylation_state, sim_time,count), 10000, atomSubset=trajectory_out_indices))
 print('production run of replicate {}...'.format(count))
 simulation.step(Simulate_Steps)
 state_production = simulation.context.getState(getPositions=True, getVelocities=True)
 state_production = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
 final_pos = state_production.getPositions()
 app.PDBFile.writeFile(simulation.topology, final_pos, open(traj_folder + '/' + 'production_sMD_NSD2_{}_H3K36me{}_{}_{}.pdb'.format(Variant,methylation_state, sim_time,count), 'w'), keepIds=True)
 print('Successful production of replicate {}...'.format(count))
 del(simulation)
 
 count = count+1 
 
