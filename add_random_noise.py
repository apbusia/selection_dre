import os
import argparse
import tempfile
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import pre_process
from data_prep import index_encode
from simulate_mutagenesis_library import index_encoding_to_string


SEED = 7


def generate_mutated_sequences(seed_seq, mutation_rate, n_seq):
    seed_seq = index_encode(seed_seq)
    alphabet_size = len(pre_process.AA_ORDER)
    variants = np.repeat([seed_seq], n_seq, axis=0)
    mutations = np.random.choice(alphabet_size, size=variants.shape)
    variants = np.mod(variants + np.random.binomial(1, mutation_rate, size=variants.shape) * mutations, alphabet_size)
    variants = index_encoding_to_string(variants)
    variants_df = pd.DataFrame({'seq': variants})
    return variants_df


def main(args):
    np.random.seed(SEED)
    tempfile.tempdir = os.getcwd()
    error_rate = args.error_rate
    
    print('Loading data from file: {}'.format(args.data_file))
    df = pd.read_csv(args.data_file)
    count_cols = {'pre': args.pre_column, 'post': args.post_column}
    
    print('Adding random noise (p={}) to {} library variants...'.format(error_rate, len(df)))
    n_folds = args.n_folds
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    print_freq = 1000
    
    for f, (train_idx, test_idx) in enumerate(kf.split(df)):
        print('\nFold {}...'.format(f))
        reads = {'pre': None, 'post': None}
        if args.save_file is not None:
            np.save('models/' + os.path.splitext(os.path.split(args.save_file)[1])[0] + '_fold{}_test_idx.npy'.format(f), test_idx)
        for v, i in enumerate(train_idx):
            if not v % print_freq:
                print('\tVariant {} of {}...'.format(v, len(train_idx)))
            for p in ['pre', 'post']:
                count = df[count_cols[p]].iloc[i]
                if count > 0:
                    noisy_seqs = generate_mutated_sequences(df['seq'].iloc[i], error_rate, count)
                    if reads[p] is None:
                        reads[p] = tempfile.NamedTemporaryFile()
                        noisy_seqs.to_csv(reads[p], index=False)
                    else:
                        noisy_seqs.to_csv(reads[p], mode='a', header=False, index=False)
        chunksize = int(1e6)
        counts = {'pre': None, 'post': None}
        for p in ['pre', 'post']:
            reads[p].seek(0)
            chunks = pd.read_csv(reads[p], chunksize=chunksize, usecols=['seq'])
            for chunk in chunks:
                chunk_counts = chunk.value_counts()
                if counts[p] is None:
                    counts[p] = chunk_counts
                else:
                    counts[p] = counts[p].add(chunk_counts, fill_value=0)
            reads[p].close()
            counts[p] = counts[p].reset_index()
            counts[p].columns = ['seq', '{}_count'.format(p)]
        counts_df = pd.merge(counts['pre'], counts['post'], how='outer', on='seq').fillna(0)

        if args.save_file is not None:
            savename, ext = os.path.splitext(args.save_file)
            if args.shuffle:
                counts_df = counts_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
            else:
                counts_df.sort_values('post_count', ascending=False, inplace=True)
            counts_df.to_csv(savename + '_fold{}'.format(f) + ext, index=False)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', help='path to CSV file containing library counts', type=str)
    parser.add_argument('--pre_column', default='pre_count', help='column name in data_file', type=str)
    parser.add_argument('--post_column', default='post_count', help='column name in data_file', type=str)
    parser.add_argument('--error_rate', default=0.01, help='(scalar) error probability', type=float)
    parser.add_argument('--n_folds', default=1, help="number of folds into which to split the dataset", type=int)
    parser.add_argument('--shuffle', help='save data in shuffled order', action='store_true')
    parser.add_argument('--save_file', help='output path to which to save generated counts', type=str)
    args = parser.parse_args()
    main(args)
