
import pandas as pd
import os
import random
import math
import time
import pickle
import numpy as np
from numpy.random import uniform
#imports from pyrosetta
from mimetypes import init
from pyrosetta import *
from pyrosetta.teaching import *
#from IPython.display import Image
#Core Includes
from rosetta.core.kinematics import MoveMap
from rosetta.core.kinematics import FoldTree
from rosetta.core.pack.task import TaskFactory
from rosetta.core.pack.task import operation
from rosetta.core.simple_metrics import metrics
from rosetta.core.select import residue_selector as selections
from rosetta.core import select
from rosetta.core.select.movemap import *
from rosetta.protocols import minimization_packing as pack_min
from rosetta.protocols import relax as rel
from rosetta.protocols.antibody.residue_selector import CDRResidueSelector
from rosetta.protocols.antibody import *
from rosetta.protocols.loops import *
from rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.docking import setup_foldtree
from pyrosetta.rosetta.protocols import *
from rosetta.core.scoring.methods import EnergyMethodOptions
from pyrosetta import *
from pyrosetta.toolbox import *
import pyrosetta.rosetta.protocols.constraint_generator
import pyrosetta.rosetta.protocols
import csv
from pyrosetta.rosetta.protocols.simple_moves import SmallMover
from pyrosetta.rosetta.protocols.simple_moves import ShearMover
from pyrosetta.rosetta.protocols.minimization_packing import MinMover
from pyrosetta import standard_packer_task
import pyrosetta.rosetta.protocols.constraint_generator
from rosetta.core.kinematics import MoveMap
from rosetta.core.kinematics import FoldTree
from rosetta.core.pack.task import TaskFactory
from rosetta.core.pack.task import operation
from rosetta.core.simple_metrics import metrics
from rosetta.core.select import residue_selector as selections
from rosetta.core import select
from rosetta.core.select.movemap import *
from rosetta.protocols import minimization_packing as pack_min
from rosetta.protocols import relax as rel
from rosetta.protocols.antibody.residue_selector import CDRResidueSelector
from rosetta.protocols.antibody import *
from rosetta.protocols.loops import *
from rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.docking import setup_foldtree
from pyrosetta.rosetta.protocols import *
from rosetta.core.scoring.methods import EnergyMethodOptions
from pyrosetta.rosetta.protocols.docking import setup_foldtree
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover


def read_fasta(file_path):
    """
    Reads a FASTA file and converts it to a pandas DataFrame.
    Treats the entire header as the sequence ID and doesn't 
    attempt to parse the header format.
    
    Parameters:
    - file_path (str): Path to the FASTA file.
    
    Returns:
    - DataFrame: Contains SeqID and Sequence columns.
    """
    sequences = {'SeqID': [], 'Sequence': []}
    caracteres_invalidos = {'X', '*', '.', '~'}
    
    with open(file_path, 'r') as file:
        current_id = None
        current_sequence = ''
        
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence if exists
                if current_id and not any(char in current_sequence for char in caracteres_invalidos):
                    sequences['SeqID'].append(current_id)
                    sequences['Sequence'].append(current_sequence)
                
                # Get the full header as ID (removing the '>' character)
                current_id = line[1:]
                current_sequence = ''
            else:
                current_sequence += line
        
        # Add the last sequence
        if current_id and not any(char in current_sequence for char in caracteres_invalidos):
            sequences['SeqID'].append(current_id)
            sequences['Sequence'].append(current_sequence)
    
    # Convert to DataFrame
    return pd.DataFrame(sequences)

def read_pose(pdb):
    """Read and prepare protein structure."""
    pyrosetta.init(options="-constant_seed -jran 314")
    pose = pose_from_pdb(pdb)
    scorefxn = pyrosetta.create_score_function("ref2015_cart.wts")
    scorefxn(pose)
    pose = pack_relax(pose = pose, scorefxn = scorefxn, times_to_relax = 3)
    return pose, scorefxn

def mc(pose_ref, scorefxn):
    """Monte Carlo simulation for structure optimization."""
    en_size=1000 ## ensemble_size
    kT=1.0
    n_moves=2 ### number of shear mover movements
    anglemax_ss=1 ### anglemax of shearmove in Alpha helix and beta-sheets
    anglemax=1 ### anglemax of shearmove in coil regions

    structure = pose_ref.clone()
    score = []
    score_final = []
    
    # Set up movemap for neighbors    
    to_move = list(range(1, len(pose_ref.sequence())))    
    move_map_list=[]
    for posi in to_move:
        mut_posi = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
        mut_posi.set_index(posi)
        nbr_selector = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector()
        nbr_selector.set_focus_selector(mut_posi)
        nbr_selector.set_include_focus_in_subset(True)
        bla=pyrosetta.rosetta.core.select.get_residues_from_subset(nbr_selector.apply(pose_ref))
        for xres in bla:
            move_map_list.append(xres)

    movemap = MoveMap()
    for x in move_map_list:    
        movemap.set_bb(x,True)
        
    # Setup movers  
    minmover = MinMover()
    minmover.movemap(movemap)
    minmover.score_function(scorefxn)
    
    to_pack = standard_packer_task(structure)
    to_pack.restrict_to_repacking()    # prevents design, packing only
    to_pack.or_include_current(True)    # considers the original sidechains
    to_pack.temporarily_fix_everything()
    
    for xres in bla: 
        to_pack.temporarily_set_pack_residue(xres, True)
    
    packmover = PackRotamersMover(scorefxn, to_pack)
    shearmover = ShearMover(movemap, kT, n_moves) ## move phi e psi do mesmo residuo
    shearmover.angle_max('H', anglemax_ss) ##helices
    shearmover.angle_max('E', anglemax_ss) ## strands
    shearmover.angle_max('L', anglemax) ## loops
    
    combined_mover = SequenceMover()
    combined_mover.add_mover(shearmover)
    combined_mover.add_mover(packmover)

    # Run Monte Carlo
    before_pose = pose_ref.clone()
    for en in range(en_size):
        after_pose = before_pose.clone()
        combined_mover.apply(after_pose)
        score.append(scorefxn(after_pose))
        before_pose = decision(before_pose, after_pose, scorefxn)
        score_final.append(scorefxn(before_pose))

    return np.mean(score_final)

def decision(before_pose, after_pose, scorefxn):
    """Make Monte Carlo decision based on energy difference."""
    E = scorefxn(after_pose) - scorefxn(before_pose)
    if E < 0:
        return after_pose
    elif random.uniform(0, 1) >= math.exp(-E/1):
        return before_pose
    else:
        return after_pose

def pack_relax(pose, scorefxn, times_to_relax):
    """Perform relaxation protocol on the structure."""
    for i in range(1, times_to_relax + 1):
        tf = TaskFactory()
        tf.push_back(operation.InitializeFromCommandline())
        tf.push_back(operation.RestrictToRepacking())

        mmf = pyrosetta.rosetta.core.select.movemap.MoveMapFactory()
        mmf.all_bb(setting=True)
        mmf.all_bondangles(setting=True)
        mmf.all_bondlengths(setting=True)
        mmf.all_chi(setting=True)
        mmf.all_jumps(setting=True)
        mmf.set_cartesian(setting=True)
    
        fr = pyrosetta.rosetta.protocols.relax.FastRelax(scorefxn_in=scorefxn, standard_repeats=1)
        fr.cartesian(True)
        fr.set_task_factory(tf)
        fr.set_movemap_factory(mmf)
        fr.min_type("lbfgs_armijo_nonmonotone")
        fr.apply(pose)
    return pose

def mutate_repack(starting_pose, posi, amino, scorefxn):
    """Mutate and repack residues."""
    pose = starting_pose.clone()
    mut_posi = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
    mut_posi.set_index(posi)
   
    nbr_selector = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector()
    nbr_selector.set_focus_selector(mut_posi)
    nbr_selector.set_include_focus_in_subset(True)
   
    not_design = pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector(mut_posi)

    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.NoRepackDisulfides())

    prevent_repacking_rlt = pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT()
    prevent_subset_repacking = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(prevent_repacking_rlt, nbr_selector, True )
    tf.push_back(prevent_subset_repacking)

    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(pyrosetta.rosetta.core.pack.task.operation.RestrictToRepackingRLT(),not_design))

    aa_to_design = pyrosetta.rosetta.core.pack.task.operation.RestrictAbsentCanonicalAASRLT()
    aa_to_design.aas_to_keep(amino)
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(aa_to_design, mut_posi))

    packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn)
    packer.task_factory(tf)
    packer.apply(pose)
   
    return pose

def unbind(pose, partners, scorefxn):
    pose_dummy = pose.clone()
    pose_binded = pose.clone()
    STEP_SIZE = 100
    JUMP = 1
    docking.setup_foldtree(pose_dummy, partners, Vector1([-1,-1,-1]))
    trans_mover = rigid.RigidBodyTransMover(pose_dummy,JUMP)
    trans_mover.step_size(STEP_SIZE)
    trans_mover.apply(pose_dummy)
    pack_relax(pose_dummy, scorefxn, 1)
    return pose_binded , pose_dummy

def dG_v2_0(pose_Sep, pose_bind, scorefxn):
    """Calculate the binding energy difference between the unbound and bound states."""
    bound_score = mc(pose_bind, scorefxn)
    unbound_score = mc(pose_Sep, scorefxn)
    dG = bound_score - unbound_score
    return dG

def Dg_bind(pose, partners, scorefxn):
    """Calculate the binding energy difference for a given complex."""
    pose_dummy = pose.clone()
    unbinded_dummy = unbind(pose_dummy, partners, scorefxn)
    return (dG_v2_0(unbinded_dummy[1], unbinded_dummy[0], scorefxn))

def Get_residues_from_chain(pose, chain):
    """Get the sequence and residue numbers for a specific chain in a given pose."""
    residue_numbers = [residue for residue in range(1, pose.size() + 1) if pose.pdb_info().chain(residue) == chain]
    chain_sequence = ''.join([pose.residue(residue).name1() for residue in residue_numbers])
    return chain_sequence, residue_numbers

def Compare_sequences(before_seq, after_seq, indexes):
    """Compare two sequences and identify mutations. Print new mutations."""
    wt = before_seq
    mut = after_seq
   
    mutation = dict()
    for index, (res1, res2) in enumerate(zip(wt, mut)):
        if res1 != res2:
            mutation[indexes[index]] = res2
    return mutation

def model_sequence(pose, mutations, scorefxn):
    """Model a sequence on a given pose by applying mutations and repacking."""
    new_pose = pose.clone()
    to_mutate = mutations
   
    for index in to_mutate:
        new_pose = mutate_repack(starting_pose = new_pose, posi = index, amino = to_mutate[index], scorefxn = scorefxn)
    pack_relax(pose = new_pose, scorefxn = scorefxn, times_to_relax = 1)
    return new_pose

def Get_residues_from_pose(pose):
    """Get the sequence and residue numbers for a specific chain in a given pose."""
   
    residue_numbers = [residue for residue in range(1, pose.size() + 1)]
    sequence = ''.join([pose.residue(residue).name1() for residue in residue_numbers])

    return sequence, residue_numbers

def Energy_contribution(pose, by_term):
    """Calculate and analyze the energy contributions of different terms for each residue in a given protein pose."""

    listadosdicts = [fa_atr, fa_rep, fa_sol, fa_intra_rep, fa_intra_sol_xover4,
                lk_ball_wtd, fa_elec, hbond_sr_bb, hbond_lr_bb, hbond_bb_sc, hbond_sc, dslf_fa13,
                omega, fa_dun, p_aa_pp, yhh_planarity, ref, rama_prepro, cart_bonded]
   
    scorefxn = pyrosetta.create_score_function("ref2015_cart.wts")
    weights = scorefxn.weights()
   
    emopts = EnergyMethodOptions(scorefxn.energy_method_options())
    emopts.hbond_options().decompose_bb_hb_into_pair_energies(True)
    scorefxn.set_energy_method_options(emopts)
   
    scorefxn.score(pose)
   
    if by_term == True:
        dasd = {'Protein': [], 'Sequence': []}
        Residues = [residue.seqpos() for residue in pose]
        dasd['Protein'].append("WT")
        for posi in Residues:
            for i in range(len(listadosdicts)):
                term_key = '{}-%s'.format(posi) % listadosdicts[i]
                dasd[term_key] = []
                dasd[term_key].append(pose.energies().residue_total_energies(posi)[listadosdicts[i]])
        dasd['Sequence'].append(pose.sequence())
        dasd["dG_Fold"] = mc(pose, scorefxn)
        df2 = pd.DataFrame(dasd)

        weights_by_term = pd.DataFrame(index=range(1, len(listadosdicts)+1), columns=range(0, 2))
        weights_by_term.iloc[:, 0] = listadosdicts
        list_weights = [1, 0.55, 1, 0.005, 1, 1, 1, 1, 1, 1, 1, 1.25, 0.4, 0.7, 0.6, 0.625, 1, 0.45, 0.5]
        weights_by_term.iloc[:, 1] = list_weights
       
        for i in range(len(weights_by_term)):
            list_to_change = df2.filter(like=str(weights_by_term.iloc[i, 0])).columns
            df2[list_to_change] = df2[list_to_change] * weights_by_term.iloc[i, 1]
        
        return df2
    else:
        seq_size = len([x for x in pose.sequence()])
        Residues = [residue.seqpos() for residue in pose]
        df_byresidue = pd.DataFrame(index=range(1, 2), columns=range(1, seq_size+1))
       
        for i in range(1, len(df_byresidue.columns)+1):
            df_byresidue.iloc[0, i-1] = pose.energies().residue_total_energy(Residues[i-1])
        df_byresidue["dG_Fold"] = mc(pose, scorefxn)
        return df_byresidue
   
def Execute(pdb, dataframe):
    """Perform protein sequence modeling using PyRosetta based on input data."""
    start_time = time.time()  # Record the start time
    pose, scorefxn = read_pose(pdb)
    pose_init = pose.clone()
    wt_energies = Energy_contribution(pose_init, by_term = True)
    for i in range(0, len(dataframe.index)):
        sequence = dataframe.iloc[i,1]
        residues_from_chain, index = Get_residues_from_pose(pose = pose)
        mutations = Compare_sequences(before_seq = residues_from_chain, after_seq = sequence, indexes = index)

        new_pose = model_sequence(pose_init, mutations, scorefxn)
        muts_energies = Energy_contribution(new_pose, by_term = True)
        wt_energies = pd.concat([wt_energies, muts_energies])
        wt_energies.iloc[i+1,0] = dataframe.iloc[i,0]
        new_pose.dump_pdb(f"{dataframe.iloc[i,0]}.pdb")
        wt_energies.to_csv("temp_out.csv")
    
    wt_energies.to_csv("output.csv")
   
    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time  # Calculate the execution time
    with open("output.txt", "w") as f:
        f.write(f"Execution Time: {round(execution_time/60, 1)} minutes\n")
               
    return wt_energies

def split_df(df):
    # Calculate the number of columns
    num_columns = df.shape[1]
    
    # Divide the number of columns by 2 to find the split point
    split_point = num_columns // 2
    
    # Split the DataFrame into two matrices
    matrix1 = df.iloc[:, :split_point].to_numpy()
    matrix2 = df.iloc[:, split_point:].to_numpy()
    
    return matrix1, matrix2

def average_matrices(matrix1, matrix2):
    # Check if the matrices have the same dimensions
    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        return "The matrices have different dimensions and cannot be added."

    # Initialize the resulting matrix with zeros
    result = [[0 for _ in range(len(matrix1[0]))] for _ in range(len(matrix1))]

    # Calculate the average of the matrix elements
    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            result[i][j] = (matrix1[i][j] + matrix2[i][j]) / 2

    return result

def process_data(df_rosetta, df):
    values = df_rosetta.iloc[:, 2:].drop(['dG_Fold'], axis=1)

    m1, m2 = split_df(values)
    mean = pd.DataFrame(average_matrices(m1, m2))
    mean.columns = values.columns[0:1881]

    p1 = df.iloc[:, 1:17]
    p3 = df_rosetta.iloc[:, -1]

    sum_df = pd.concat([p1.reset_index(drop=True), 
                        mean.reset_index(drop=True), 
                        p3.reset_index(drop=True)], axis=1)
    return sum_df

def make_prediction(model_pkl, data):
    # Load the model
    with open(model_pkl, 'rb') as file:
        model = pickle.load(file)
    
    # Get the feature names expected by the model
    feature_names = model.feature_names_in_
    
    # Select row 1 from the DataFrame with columns in the correct order
    X_input = data.loc[1, feature_names].to_frame().T
    
    # Make the prediction
    prediction = model.predict(X_input)
    
    return prediction

def predict_hiv_resistance(model, fasta_path, drug, pdb, n_runs=100, seed=42, output_file="predictions.txt"):
    random.seed(seed)
    np.random.seed(seed)

    # Initialize list to store times
    times = []
    
    for i in range(n_runs):
        start_time = time.time()
        dataframe = read_fasta(fasta_path)
        data_rosetta = Execute(pdb, dataframe=dataframe)
        data = process_data(data_rosetta, dataframe)
        prediction = make_prediction(model, data)
        end_time = time.time()
        times.append(end_time - start_time)

    mean_time = np.mean(times)
    std_time = np.std(times)

    result = f"Prediction for drug {drug}: {prediction}\n"
    result += f"Number of runs: {n_runs}\n"
    result += f"Mean execution time: {mean_time:.4f} seconds\n"
    result += f"Standard deviation: {std_time:.4f} seconds\n"
    result += f"Seed used: {seed}\n"
    result += f"Individual run times: {', '.join([f'{t:.4f}' for t in times])}\n"
    result += "\nSequence Information:\n"
    for i, (seq_id, sequence) in enumerate(zip(dataframe['SeqID'], dataframe['Sequence'])):
        result += f"Sequence {i+1}: {seq_id}\n"
        result += f"Length: {len(sequence)}\n"
    
    print(result)  # Console output
    with open(output_file, 'a') as f:
        f.write(result + '\n')

# In-house test
fasta_path = 'seq_test.fasta'
pdb_path = '3oxc_edited.pdb'
predict_hiv_resistance(model='NFV7_best_logistic_regression_model.pkl', 
                          fasta_path=fasta_path,
                          drug='NFV', 
                          n_runs=100,
                          pdb=pdb_path, 
                          output_file='inhouse_rosetta_prediction.txt')

# Shen test
predict_hiv_resistance(model='NFV14_best_logistic_regression_model.pkl', 
                          fasta_path=fasta_path,
                          drug='NFV', 
                          n_runs=100,
                          pdb=pdb_path, 
                          output_file='inhouse_rosetta_prediction.txt')

# Steiner test
predict_hiv_resistance(model='nfv11_best_logistic_regression_model.pkl', 
                          fasta_path=fasta_path,
                          drug='NFV', 
                          n_runs=100,
                          pdb=pdb_path, 
                          output_file='inhouse_rosetta_prediction.txt')

