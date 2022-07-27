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


def load_seed(file_path):
    print('\tUsing user-supplied seed FASTA: {}'.format(file_path))
    seed_seq = None
    for record in SeqIO.parse(file_path, 'fasta'):
        seed_seq = str(record.seq)
    return seed_seq.upper()


def index_encoding_to_string(index_seqs):
    """Converts integer sequences into string sequences."""
    str_seqs = []
    for seq in index_seqs:
        str_seqs.append(''.join([pre_process.AA_ORDER[i] for i in seq]))
    return str_seqs


def generate_mutagenesis_library(seed_seq, mutation_rate, n_seq):
    alphabet_size = len(pre_process.AA_ORDER)
    variants = np.repeat([seed_seq], n_seq, axis=0)
    mutations = np.random.choice(alphabet_size, size=variants.shape)
    mutations[0] = 0  # Keep wildtype/seed sequence in the library.
    variants = np.mod(variants + np.random.binomial(1, mutation_rate, size=variants.shape) * mutations, alphabet_size)
    variants = index_encoding_to_string(variants)
    variants_df = pd.DataFrame({'seq': variants})
    return variants_df


def main(args):
    np.random.seed(SEED)
    
    # Generate list of variants.
    if args.load_file is not None:
        print('Loading library from file: {}'.format(args.load_file))
        library_df = pd.read_csv(args.load_file)
    else:
        print('Generating random mutagenesis library...')
        if args.parent_seq is None:
            seed_seq = np.random.choice(len(pre_process.AA_ORDER), size=200)
        else: 
            seed_seq = data_prep.index_encode(load_seed(args.parent_seq))
        library_df = generate_mutagenesis_library(seed_seq, args.mutation_rate, args.n_variants)
    
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
    parser.add_argument('--parent_seq', help='path to FASTA of seed sequence; if none, a random seed of length 200 is used', type=str)
    parser.add_argument('--mutation_rate', default=0.005, help='mutation rate/probability of mutating each position', type=float)
    parser.add_argument('--n_variants', default=int(8.6e6), help='number of variants to generate', type=int)
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
