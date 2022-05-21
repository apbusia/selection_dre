import os
import argparse
import string
import numpy as np
import pandas as pd
# from Bio.Seq import Seq
from Bio import SeqIO
from tensorflow import keras
import data_prep
import pre_process
import modeling
import simulate_nnk_library as snl
from seqtools import SequenceTools
# import sys
# sys.path.append('..')


SEED = 7


def load_parents(file_path=None):
    if file_path is None:
        file_path = os.path.join(os.path.abspath(os.getcwd()), 'data/aav_parent_alignment.fasta')
        print('\tUsing default parents FASTA: {}'.format(file_path))
    else:
        print('\tUsing user-supplied parents FASTA: {}'.format(file_path))
    parent_dict = {}
    for record in SeqIO.parse(file_path, 'fasta'):
        parent_dict[record.id] = str(record.seq)
    return parent_dict


def crossovers_to_blocks(crossovers, max_length):
    """Turns crossover points, which are 1-based for readability, into (begin, end) 
    pairs of 0-based indices into the parent sequence."""
    # Based on getFragments in SCHEMA library.
    xovers = list(crossovers[:])
    if 1 not in xovers:
        xovers = [1] + xovers
    xovers.append(max_length+1)
    fragments = [(xovers[i]-1, xovers[i+1]-1) for i in range(len(xovers)-1)]
    return fragments


def to_dnary_string(n, base):
    """Returns a string representation of n in the given base."""
    # Inverse function to int(str, base) or long(str, base)
    # Ref: https://runestone.academy/runestone/books/published/pythonds/Recursion/pythondsConvertinganIntegertoaStringinAnyBase.html
    convertString = string.digits + string.ascii_letters
    if not 2 <= base <= 36:
        raise(ValueError, 'base must be between 2 and 36')
    if n < base:
        return convertString[n]
    else:
        return to_dnary_string(n // base, base) + convertString[n % base]


def make_chimera(idx, n_parents, n_blocks):
    # Based on makeChimera in SCHEMA library.
    # The next two lines turn chim_num into a chimera block pattern
    # (e.g., 0 -> '11111111', 1 -> '11111112', 2 -> '11111113'...)
    block_ids = to_dnary_string(idx, n_parents)
    chimera_blocks = ''.join(['1']*(n_blocks-len(block_ids))+['%d'%(int(x)+1,) for x in block_ids])
    return chimera_blocks


def get_chimera_sequence(chimera_blocks, blocks, parent_dict):
    """Converts a chimera block patterns, such as '11213312', into a protein sequence
    by assembling fragments from the parents."""
    # Based on getChimeraSequence in SCHEMA library.
    chimera = ''
    for i, c in enumerate(chimera_blocks):
        which_parent = 'AAV{}'.format(int(c))
        begin, end = blocks[i]
        chimera += parent_dict[which_parent][begin:end]
    return chimera


def generate_recombination_library(parent_dict, crossovers=None, n_crossovers=None, save_file=None):
    if crossovers is None and n_crossovers is None:
        # Crossovers from Ojala et al. [128, 284, 347, 417, 480, 589, 673] for consensus of length 738
        # Default crossovers are slightly adjusted for a consensus of length 751
        crossovers = [128, 287, 350, 423, 487, 604, 689]
        txt = 'default'
    elif crossovers is None:
        crossovers = np.sort(np.random.choice(len(parent_dict['Consensus']), n_crossovers, replace=False))
        txt = 'random'
    else:
        crossovers = np.sort(crossovers)
        txt = 'user-supplied'
    print('\tUsing {} crossover positions: {}'.format(txt, crossovers))
    
    variants = []
    n_parents = len(parent_dict) - 1  # Don't count consensus sequence as a parent.
    blocks = crossovers_to_blocks(crossovers, len(parent_dict['Consensus']))
    n_blocks = len(blocks)
    for i in range(n_parents ** n_blocks):
        block_ids = make_chimera(i, n_parents, n_blocks)
        seq = get_chimera_sequence(block_ids, blocks, parent_dict)
        variants.append({'parent_ids': block_ids, 'seq': seq})
    variants_df = pd.DataFrame(variants)
    return variants_df


def main(args):
    np.random.seed(SEED)
    
    # Generate list of variants.
    if args.load_file is not None:
        print('Loading library from file: {}'.format(args.load_file))
        library_df = pd.read_csv(args.load_file)
    else:
        print('Generating recombination library...')
        parent_dict = load_parents(args.parent_fasta)
        library_df = generate_recombination_library(parent_dict, args.crossovers, args.n_crossovers)
    
    if args.filter_duplicates:
        library_df = library_df[~library_df.duplicated(subset=['seq'])]    

    # Compute simulated fitness values for variants.
    if args.nk_sizes is not None:
        print('Simulating NK fitness values...')
        for k in args.nk_sizes:
            if k > 1:
                print('\tComputing NK model with size {} random neighborhoods'.format(k))
                library_df['nk_{}_fitness'.format(k)] = snl.get_random_nk_fitness(k, library_df['seq'])
            print('\tComputing NK model with size {} block neighborhoods'.format(k))
            library_df['block_{}_fitness'.format(k)] = snl.get_block_nk_fitness(k, library_df['seq'])
    if args.epistatic_terms is not None:
        print('Simulating linear epistatic fitness values...')
        for p in args.epistatic_terms:
            print('\tComputing epistatic model with {} epistatic terms'.format(p))
            library_df['epistatic_{}_fitness'.format(p)] = snl.get_epistatic_fitness(p, library_df['seq'])
    if args.ann_sizes is not None:
        if args.save_file is None:
            raise(ValueError, 'save_file required to simulate fitness values with neural network(s).')
        print('Simulating neural network fitness values...')
        # Reload large dataset in chunks to avoid memory issues.
        library_df.to_csv(args.save_file, index=False)
        library_df = pd.read_csv(args.save_file, chunksize=int(5e5))
        fitnesses = {}
        for l in args.ann_sizes:
            print('\tComputing fitness with {}-layer neural network'.format(l))
            fitnesses['ann_{}_fitness'.format(l)] = snl.get_random_ann_fitness(l, args.ann_units, library_df)
        # Restore and fill in the un-chunked dataset.
        library_df = pd.read_csv(args.save_file)
        for k, v in fitnesses.items():
            library_df[k] = v
        del fitnesses
    
    if args.filter_conserved:
        seqs = library_df['seq'].str.split('', expand=True)
        filtered_seqs = seqs.loc[:, (seqs != seqs.iloc[0]).any()]
        filtered_seqs = filtered_seqs.iloc[:,0].str.cat(others=filtered_seqs.iloc[:,1:])
        library_df['seq'] = filtered_seqs

    # Save simulated dataset to file.
    save_path = args.save_file
    if args.save_file is not None:
        print('Saving library of {:.2e} simulated variants to {}'.format(len(library_df), args.save_file))
        library_df.to_csv(args.save_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent_fasta', help='path to alignment FASTA of AAV parent sequences', type=str)
    parser.add_argument('--crossovers', help='list of crossover points for recombination library', nargs='+', type=int)
    parser.add_argument('--n_crossovers', help='number of desired crossovers to be chosen at random', type=int)
    parser.add_argument('--filter_duplicates', help='remove library variants with duplicate sequence', action='store_true')
    parser.add_argument('--filter_conserved', help='remove conserved positions in sequence', action='store_true')
    parser.add_argument('--save_file', help='output path to which to save generated counts', type=str)
    parser.add_argument('--nk_sizes', help='list of neighborhood sizes for NK models', nargs='+', type=int)
    parser.add_argument('--epistatic_terms', help='list of number of terms to include for linear epistatic models', nargs='+', type=int)
    parser.add_argument('--ann_sizes', help='list of layer counts for random neural network models', nargs='+', type=int)
    parser.add_argument('--ann_units', default=100, help='number of hidden units in each layer for random neural network models', type=int)
    parser.add_argument('--load_file', help='library file to load and append additional fitness(es)', type=str)
    args = parser.parse_args()
    main(args)
