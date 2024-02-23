#!/usr/bin/env python

import MDAnalysis as mda
import numpy as np
import argparse
import sys
import os
import pandas as pd
import tqdm
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')

############################
#       FUNCTIONS
############################

# Function to get velocities, temperatures and kinetic energies
def velo(file_top, file_trj, sel, ts0, tsN):
    # Universe
    u = mda.Universe(file_top, file_trj, in_memory=False)
    
    # Select all water atoms
    wat = u.select_atoms(sel)

    w = u.select_atoms('resid 1')

    # Get the number of water molecules00547
    nWat = wat.n_residues

    # Get the number of atoms per water molecule
    nPoint = len(wat) // nWat

    # Universal gas constant
    R = 8.314 # J/mol/K
    
    # Mass of atoms in water (kg_amu)
    w = u.select_atoms(f'resid 1')
    mass = w.masses * 0.001  # amu to kg_amu
    M = np.sum(mass)
    
    time = []
    vel_trans = []
    vel_angul = []
    KE_tr = []
    KE_rot = []
    
    for ts in tqdm.tqdm(u.trajectory[ts0:tsN], desc="Processing trajectory"):
        time.append(ts.time)
        
        #------------------#
        # TRANSLATIONAL KE #
        #------------------#
        
        # Velocities of water atoms
        vel = wat.velocities * 100 # ang/ps to m/s
        vel_abs = np.reshape(vel, (nWat, nPoint, 3))

        # Velocity of water COM
        vel_com = np.sum(vel_abs*mass[:, np.newaxis], axis=1) / M
        vel_mag = np.linalg.norm(vel_com, axis=1)
        vel_trans.append(vel_com)

        # Kinetic energy (J)
        KE_tr.append(np.mean(0.5 * M * vel_mag**2))
        
        #---------------#
        # Rotational KE #
        #---------------#
        
        # Velcity relative to COM velocity
        vel_wat = vel_abs - vel_com[:, np.newaxis]
        
        # Positions of water atoms
        pos  = wat.positions * 1e-10 # ang to m
        pos_abs = np.reshape(pos, (nWat, nPoint, 3))
        
        # Velocity of water COM
        pos_com = np.sum(pos_abs*mass[:, np.newaxis], axis=1) / M
        pos_wat = pos_abs - pos_com[:, np.newaxis]
        
        # Calculating the RHS of the matrix equation
        rhs = np.sum(np.cross(pos_wat, vel_wat), axis=1)
              
        pospos = []
        
        for dim1 in range(3):
            for dim2 in range(dim1,3):
                pospos.append(pos_wat[:,:,dim1] * pos_wat[:,:,dim2])
                
        pospos = np.array(pospos)

        # Calculating the RHS of the matrix equation
        lhs = np.zeros((nWat,3,3))
        
        lhs[:,0,0] = np.sum(pospos[5,:,:]+pospos[3,:,:], axis=1)
        lhs[:,1,1] = np.sum(pospos[0,:,:]+pospos[5,:,:], axis=1)
        lhs[:,2,2] = np.sum(pospos[0,:,:]+pospos[3,:,:], axis=1)
        lhs[:,0,1] = lhs[:,1,0] = -np.sum(pospos[1,:,:], axis=1)
        lhs[:,0,2] = lhs[:,2,0] = -np.sum(pospos[2,:,:], axis=1)
        lhs[:,1,2] = lhs[:,2,1] = -np.sum(pospos[4,:,:], axis=1)
        
        # Calculating the moment of inertia tensor
        moi = np.zeros((nWat,3,3))
         
        moi[:,0,0] = np.sum((mass * (pos_wat[:,:,1]**2 + pos_wat[:,:,2]**2)), axis=1) # xx        
        moi[:,1,1] = np.sum((mass * (pos_wat[:,:,0]**2 + pos_wat[:,:,2]**2)), axis=1) # yy        
        moi[:,2,2] = np.sum((mass * (pos_wat[:,:,0]**2 + pos_wat[:,:,1]**2)), axis=1) # zz        
        moi[:,0,1] = moi[:,1,0] = -np.sum((mass * pos_wat[:,:,0] * pos_wat[:,:,1]), axis=1) # xy & yx
        moi[:,1,2] = moi[:,2,1] = -np.sum((mass * pos_wat[:,:,1] * pos_wat[:,:,2]), axis=1) # yz & zy
        moi[:,0,2] = moi[:,2,0] = -np.sum((mass * pos_wat[:,:,0] * pos_wat[:,:,2]), axis=1) # zx & xz
        
        # Solving the matrix equation for all water molecules and calculating rotational KE
        ang_vel = np.linalg.solve(lhs, rhs)
        vel_angul.append(ang_vel)
        
        kin_rot = 0.5 * np.einsum('jk,jkl,jl->j', ang_vel, moi, ang_vel) 
        KE_rot.append(np.mean(kin_rot))
        
    # Temperature
    T_tr  = [(2*x) / (3*R) for x in KE_tr]
    T_rot = [(2*x) / (3*R) for x in KE_rot]

    return time, vel_trans, vel_angul, KE_tr, KE_rot, T_tr, T_rot

# Function to calculate velocity autocorrelation function
def vacf(vel, time):
    t = time[:len(time)//2]
    v = vel[:len(vel)//2,:,:]

    nWat = v.shape[1]

    acf_matrix = np.zeros((nWat, v.shape[-1], len(v)))

    for i in tqdm.tqdm(range(nWat), desc = 'Calculating velocity ACF'):
        for j in range(v.shape[-1]):
            acf_matrix[i, j, :] = sm.tsa.acf(v[:, i, j], nlags=len(v)-1)

    acf = np.sum(acf_matrix, axis=1)
    acf_norm = acf / np.max(acf, axis=1, keepdims=True)

    vacf = np.mean(acf, axis=0)
    vacf_norm = np.mean(acf_norm, axis=0)

    dos = np.fft.fft(vacf)

    dt = t[1] - t[0]
    max_freq = 33.33 / dt # frequency in cm^-1
    freq = np.linspace(0, max_freq, len(dos))

    return vacf_norm, freq, dos

# function to output velocity files
def out_vel(vel_array, filename):
    print(f"writing output to file {filename}...")

    ns = vel_array.shape[0]
    nw = vel_array.shape[1]

    vel = vel_array.reshape(-1, 3)

    num = np.tile(np.arange(1, nw+1), ns)

    df = pd.DataFrame({'mol': num, 'vx': vel[:, 0], 'vy': vel[:, 1], 'vz': vel[:, 2]})

    df['mol'] = df['mol'].astype(int)
    df.to_csv(filename, sep=' ', index=False, header=False, float_format='%.4f')

    print("Complete!")

# function to output kinetic energy or temperature files
def output(array, filename):
    print(f"writing output to file {filename}...")
    
    n = array.shape[0]
    
    if n == 2:
        df = pd.DataFrame({'time':array[0], 'ke': array[1]})
    elif n == 3:
        df = pd.DataFrame({'time':array[0], 'ke_tr': array[1], 'ke_rot': array[2]})

    df.to_csv(filename, sep=' ', index=False, header=False, float_format='%.4f')
    print("Complete!")

############################
#    ARGUMENT PARSING
############################

# Command line arguments parsed with flags
desc = (
        "%(prog)s generates translational & rotational velocity,\n"
        "kinetic energy, temperature, velocity ACF, and power spectrum from GROMACS TRR trajectory.\n"
        "The input trajectory should be PBC corrected. For example,\n"
        "'gmx trjconv -f traj.trr -s topol.tpr -o traj_pbc.trr -pbc mol-ur compact'.\n"
        "The trajectory file must have velocity information."
        )

parser = argparse.ArgumentParser(description=desc, epilog="Copyright reserved by Dr. Saumyak Mukherjee")

inp  = parser.add_argument_group('Input arguments')
inpo = parser.add_argument_group('Input arguments (optional)')
oup  = parser.add_argument_group('Output arguments (optional)')
log  = parser.add_argument_group('Boolean arguments (optional)')

inp.add_argument('-top' , '--topology'   , type=str , help='topology file (.gro, .tpr, .pdb) [default: %(default)s]' , default='topol.gro'    , metavar = '', required=True)
inp.add_argument('-trj' , '--trajectory' , type=str , help='trajectory file (.trr) [default: %(default)s]'           , default='traj_pbc.trr' , metavar = '', required=True)
inp.add_argument('-sel' , '--selection'  , type=str , help='atomgroup selection [default: %(default)s]'              , default='"resname SOL"', metavar = '', required=True)

inpo.add_argument('-b'   , '--begin'      , type=int , help='beginning time step [default: %(default)s]', default=0 , metavar = '')
inpo.add_argument('-e'   , '--end'        , type=int , help='end time step [default: %(default)s]'      , default=-1, metavar = '')

oup.add_argument('-vtr' , '--velo_trans' , type=str , help='output translational velocity file (.dat) [default: %(default)s]', default='velo_trans.dat' , metavar = '') 
oup.add_argument('-vrot', '--velo_rot'   , type=str , help='output rotational velocity file (.dat) [default: %(default)s]'   , default='velo_ang.dat'   , metavar = '') 
oup.add_argument('-ke'  , '--kin_ener'   , type=str , help='output kinetic energy file (.dat) [default: %(default)s]'        , default='kin_ener.dat'   , metavar = '')
oup.add_argument('-temp', '--temperature', type=str , help='output temperature file (.dat) [default: %(default)s]'           , default='temperature.dat', metavar = '')
oup.add_argument('-ac'  , '--velacf'     , type=str , help='output velocity ACF file (.dat) [default: %(default)s]'          , default='vacf.dat'       , metavar = '')
oup.add_argument('-dos' , '--spectrum'   , type=str , help='output density of states file (.dat) [default: %(default)s]'     , default='dos.dat'        , metavar = '')

log.add_argument('-VT', '--VELTR'  , help='turn on translational velocity output'      , action='store_true')
log.add_argument('-VR', '--VELROT' , help='turn on rotational velocity output'         , action='store_true')
log.add_argument('-KT', '--KINTR'  , help='turn on translational kinetic energy output', action='store_true')
log.add_argument('-KR', '--KINROT' , help='turn on rotational kinetic energy output'   , action='store_true')
log.add_argument('-TT', '--TEMPTR' , help='turn on translational temperature output'   , action='store_true')
log.add_argument('-TR', '--TEMPROT', help='turn on rotational temperature output'      , action='store_true')
log.add_argument('-AT', '--VACFTR' , help='turn on translational vacf output'          , action='store_true')
log.add_argument('-AR', '--VACFROT', help='turn on rotational vacf output'             , action='store_true')
log.add_argument('-DT', '--DOSTR'  , help='turn on translational DoS output'           , action='store_true')
log.add_argument('-DR', '--DOSROT' , help='turn on rotational DoS output'              , action='store_true')

args = parser.parse_args()

top_file = args.topology
trj_file = args.trajectory
selatoms = args.selection
ts0      = args.begin
tsN      = args.end

veltr_file  = args.velo_trans
velrot_file = args.velo_rot
ke_file     = args.kin_ener
temp_file   = args.temperature
ac_file     = args.velacf
dos_file    = args.spectrum

veltr_bool  = args.VELTR
velrot_bool = args.VELROT
ketr_bool   = args.KINTR
kerot_bool  = args.KINROT
Ttr_bool    = args.TEMPTR
Trot_bool   = args.TEMPROT
vactr_bool  = args.VACFTR
vacrot_bool = args.VACFROT
dostr_bool  = args.DOSTR
dosrot_bool = args.DOSROT

velo_bools = [veltr_bool, velrot_bool, ketr_bool, kerot_bool, Ttr_bool, Trot_bool]
vacf_bools = [vactr_bool, vacrot_bool, dostr_bool, dosrot_bool]

# Check for the existence of input files
if not os.path.exists(top_file):
    print(f"Error: The {top_file} file does not exist in the current directory!")
    sys.exit(1)

if not os.path.exists(trj_file):
    print(f"Error: The {trj_file} file does not exist in the current directory!")
    sys.exit(1)

############################
#       MAIN PROGRAM
############################

# If nothing provided
if not any(velo_bools) and not any(vacf_bools):
    print('If you do not want any output then better not waste energy.')
    sys.exit()
else:
    # Call the velo function
    results = velo(top_file, trj_file, selatoms, ts0, tsN)

# Extract values
time    = np.array(results[0]) # ps
vel_tr  = np.array(results[1]) # m/s
vel_ang = np.array(results[2]) # /s
ke_tr   = np.array(results[3]) # J/mol
ke_rot  = np.array(results[4]) # J/mol
T_tr    = np.array(results[5]) # K
T_rot   = np.array(results[6]) # K

if vactr_bool or dostr_bool:
    vac_tr = vacf(vel_tr, time)

    vacf_tr = np.array(vac_tr[0]) # normalized
    freq    = np.array(vac_tr[1])[:len(vac_tr[1])//2]
    dos_tr  = np.array(vac_tr[2].real)[:len(vac_tr[2])//2]

if vacrot_bool or dosrot_bool:
    vac_rot = vacf(vel_ang, time)

    vacf_rot = np.array(vac_rot[0]) # normalized
    freq     = np.array(vac_rot[1])[:len(vac_rot[1])//2]
    dos_rot  = np.array(vac_rot[2].real)[:len(vac_rot[2])//2]

############################
#         OUTPUT
############################

# Velocities
if veltr_bool:
    out_vel(vel_tr, veltr_file)
elif velrot_bool:
    out_vel(vel_ang, velrot_file)

# Kinetic energies
if ketr_bool and kerot_bool:
    arr = np.array([time, ke_tr, ke_rot])
    output(arr, ke_file)
elif ketr_bool:
    arr = np.array([time, ke_tr])
    output(arr, ke_file)
elif kerot_bool:
    arr = np.array([time, ke_rot])
    output(arr, ke_file)

# Temperatures
if Ttr_bool and Trot_bool:
    arr = np.array([time, T_tr, T_rot])
    output(arr, temp_file)
elif Ttr_bool:
    arr = np.array([time, T_tr])
    output(arr, temp_file)
elif Trot_bool:
    arr = np.array([time, T_rot])
    output(arr, temp_file)

# vacf
if vactr_bool and vacrot_bool:
    arr = np.array([time[:len(time)//2], vacf_tr, vacf_rot])
    output(arr, ac_file)
elif vactr_bool:
    arr = np.array([time[:len(time)//2], vacf_tr])
    output(arr, ac_file)
elif vacrot_bool:
    arr = np.array([time[:len(time)//2], vacf_rot])
    output(arr, ac_file)

#dos
if dostr_bool and dosrot_bool:
    arr = np.array([freq, dos_tr, dos_rot])
    output(arr, dos_file)
elif dostr_bool:
    arr = np.array([freq, dos_tr])
    output(arr, dos_file)
elif dosrot_bool:
    arr = np.array([freq, dos_rot])
    output(arr, dos_file)
