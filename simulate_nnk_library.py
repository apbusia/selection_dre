import os
import argparse
import numpy as np
import pandas as pd
from Bio.Seq import Seq
from tensorflow import keras
import data_prep
import pre_process
import modeling
from seqtools import SequenceTools


SEED = 7


def get_subsequence(positions, seq_series):
    subseq_series = None
    for i in positions:
        if subseq_series is None:
            subseq_series = seq_series.str.get(i)
        else:
            subseq_series += seq_series.str.get(i)
    return subseq_series


def get_subsequence_fitness(neighborhood, seq_series):
    subseq_series = get_subsequence(neighborhood, seq_series)
    unique_subseq = subseq_series.unique()
    subseq_fitness = dict(zip(unique_subseq, np.random.normal(size=len(unique_subseq))))
    return subseq_series.map(subseq_fitness)


def get_random_nk_fitness(k, seq_series):
    fitness = np.zeros(len(seq_series))
    seq_len = len(seq_series[0])
    seq_pos = np.arange(seq_len)
    np.random.shuffle(seq_pos)
    for i in seq_pos:
        neighbors = np.random.choice(seq_len, k, replace=False)
        if i not in neighbors:
            neighbors[-1] = i
        neighbors = np.sort(neighbors)
        fitness += get_subsequence_fitness(neighbors, seq_series)
    return fitness


def get_adjacent_nk_fitness(k, seq_series):
    fitness = np.zeros(len(seq_series))
    seq_len = len(seq_series[0])
    padded_pos = np.arange(seq_len + k)
    for i in range(seq_len):
        neighbors = padded_pos[i:i+k] % seq_len
        fitness += get_subsequence_fitness(neighbors, seq_series)
    return fitness


def get_block_nk_fitness(k, seq_series):
    fitness = np.zeros(len(seq_series))
    seq_len = len(seq_series[0])
    padded_pos = np.arange(seq_len + k)
    for i in range(0, seq_len, k):
        neighbors = padded_pos[i:i+k] % seq_len
        fitness += k * get_subsequence_fitness(neighbors, seq_series)
    return fitness


def get_epistatic_fitness(p, seq_series):
    fitness = np.zeros(len(seq_series))
    seq_len = len(seq_series[0])
    # Always include independent sites.
    for i in range(seq_len):
        fitness += get_subsequence_fitness([i], seq_series)
    # Include p% of epistatic terms, randomly sampled.
    # Poelwijk Fig 2 suggests epistatic order is a bell-curve centered on third order.
    n_terms = int(np.around(p * 2 ** seq_len))
    epistatic_orders, counts = np.unique(np.random.normal(loc=3, scale=1, size=n_terms).round(), return_counts=True)
    for i, order in enumerate(epistatic_orders):
        if order > 1 and order <= seq_len:
            neighborhoods = [np.random.choice(seq_len, int(order), replace=False) for _ in range(counts[i])]
            neighborhoods = np.unique(np.sort(neighborhoods), axis=0)
            for neighborhood in neighborhoods:
                fitness += get_subsequence_fitness(np.sort(neighborhood), seq_series)
    return fitness

    
def get_random_ann_fitness(n_layers, n_units, sequences, batch_size=1024):
    keras.backend.clear_session()
    n_samples = len(sequences)
    xgen = modeling.get_dataset(
        sequences, np.arange(n_samples), data_prep.index_encode, np.zeros((n_samples, 2)), None, None, batch_size=batch_size, shuffle=False)
    input_shape = tuple(xgen.element_spec[0].shape[1:])
    model = modeling.make_ann_model(input_shape, num_hid=n_layers, hid_size=n_units)
    return model.predict(xgen).reshape(n_samples)


def dna_integers_to_string(int_seq):
    return ''.join([pre_process.NUC_ORDER[i] for i in int_seq])


def sample_integer_sequences(probs, n_seq):
    seq_len, alphabet_size = probs.shape
    return np.array([np.random.choice(alphabet_size, n_seq, p=probs[i]) for i in range(seq_len)]).T


def generate_nnk_library(n_aa, n_variants):
    p = data_prep.get_nnk_p(n_aa=n_aa)
    seq = sample_integer_sequences(p, n_variants)
    seq = np.unique(seq, axis=0)
    while len(seq) < n_variants:
        new_seq = sample_integer_sequences(p, n_variants - len(seq))
        seq = np.concatenate([seq, new_seq])
        seq = np.unique(seq, axis=0)
    seq = np.apply_along_axis(dna_integers_to_string, 1, seq)
    return [str(Seq(nuc_seq).translate()) for nuc_seq in seq]


def main(args):
    np.random.seed(SEED)
    
    # Generate list of variants.
    if args.load_file is not None:
        print('Loading library from file: {}'.format(args.load_file))
        library_df = pd.read_csv(args.load_file)
    else:
        seq_len = args.seq_length
        n_variants = args.n_variants
        print('Generating length {} NNK library of size {}...'.format(seq_len, n_variants))
        library_df = pd.DataFrame({'seq': generate_nnk_library(seq_len, n_variants)})
    

    # Compute simulated fitness values for variants.
    if args.nk_sizes is not None:
        print('Simulating NK fitness values...')
        for k in args.nk_sizes:
            if k > 1:
                print('\tComputing NK model with size {} random neighborhoods'.format(k))
                library_df['nk_{}_fitness'.format(k)] = get_random_nk_fitness(k, library_df['seq'])
            print('\tComputing NK model with size {} block neighborhoods'.format(k))
            library_df['block_{}_fitness'.format(k)] = get_block_nk_fitness(k, library_df['seq'])
    if args.epistatic_props is not None:
        print('Simulating linear epistatic fitness values...')
        for p in args.epistatic_props:
            print('\tComputing epistatic model with {} proportion of epistatic terms'.format(p))
            library_df['epistatic_{}_fitness'.format(p)] = get_epistatic_fitness(p, library_df['seq'])
    if args.ann_sizes is not None:
        print('Simulating neural network fitness values...')
        for l in args.ann_sizes:
            print('\tComputing fitness with {}-layer neural network'.format(l))
            library_df['ann_{}_fitness'.format(l)] = get_random_ann_fitness(l, args.ann_units, library_df['seq'])
    
    # Save simulated dataset to file.
    save_path = args.save_file
    if args.save_file is not None:
        print('Saving library of {:.2e} simulated variants to {}'.format(len(library_df), args.save_file))
        library_df.to_csv(args.save_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_length', default=7, help='number of amino acids to generate per sequence', type=int)
    parser.add_argument('--n_variants', default=int(8e6), help='number of unique variants to generate', type=int)
    parser.add_argument('--save_file', help='output path to which to save generated counts', type=str)
    parser.add_argument('--nk_sizes', help='list of neighborhood sizes for NK models', nargs='+', type=int)
    parser.add_argument('--epistatic_props', help='list of proportions of terms to include for linear epistatic models', nargs='+', type=float)
    parser.add_argument('--ann_sizes', help='list of layer counts for random neural network models', nargs='+', type=int)
    parser.add_argument('--ann_units', default=100, help='number of hidden units in each layer for random neural network models', type=int)
    parser.add_argument('--load_file', help='library file to load and append additional fitness(es)', type=str)
    args = parser.parse_args()
    main(args)
