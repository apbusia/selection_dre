import os
import argparse
import numpy as np
import pandas as pd
from simulate_from_fitness import simulate_true_proportions


SEED = 7


def merge_pre_and_post_counts(pre_df, post_df, seq_column='seq', count_column='count'):
    pre_groups = pre_df.groupby(seq_column)
    pre_df = pre_groups.sum().reset_index()
    post_groups = post_df.groupby(seq_column)
    post_df = post_groups.sum().reset_index()
    merged_df = pd.merge(pre_df, post_df, how='outer', on=seq_column, suffixes=('_pre', '_post')).fillna(0)
    merged_df = merged_df.rename(columns={count_column + "_pre": 'pre_count', 
                                          count_column + "_post": 'post_count', 
                                          seq_column: 'seq'})
    merged_df = merged_df.loc[~merged_df['seq'].str.contains('X')]
    merged_df = merged_df.reset_index(drop=True)
    return merged_df


def main(args):
    np.random.seed(SEED)
    
    print('Loading data from file: {}'.format(args.data_file))
    df = pd.read_csv(args.data_file)
    
    if 'post_p' not in df.columns:
        print('Using {} as fitness'.format(args.fitness_column))
        df = df[['seq', args.fitness_column]]
        fitness = df[args.fitness_column].values
        if args.standardize:
            print('Standardizing fitness to have 0 mean and unit variance...')
            fitness = (fitness - np.mean(fitness)) / np.std(fitness)
    
        print('Computing \'true\' library proportions...')
        pre_p, post_p, true_enrichment = simulate_true_proportions(fitness, args.dirichlet_concentration)
        df['pre_p'] = pre_p
        df['post_p'] = post_p
        df['true_enrichment'] = true_enrichment
    
    print('Sampling variant counts ({:.1e} reads)...'.format(args.total_reads))
    counts = {'pre': np.zeros((args.n_replicates, len(df))).astype(int),
              'post': np.zeros((args.n_replicates, len(df))).astype(int)}
    for r in range(args.n_replicates):
        for p in ['pre', 'post']:
            counts[p][r] = np.random.multinomial(args.total_reads, df['{}_p'.format(p)].values)
    print('Sampling short reads ({} variants)...'.format(len(df)))
    print_freq = 1000
    replicates = {'pre': [None] * args.n_replicates,
                  'post': [None] * args.n_replicates}
    for i in range(len(df)):
        if not i % print_freq:
            print('\tVariant {}...'.format(i))
        for r in range(args.n_replicates):
            for p in ['pre', 'post']:
                count = counts[p][r, i]
                if count > 0:
                    reads = pd.DataFrame({
                        'start': np.random.randint(len(df['seq'][i]) - args.read_length + 1, size=count)
                    })
                    reads['end'] = reads['start'] + args.read_length
                    reads = pd.DataFrame(reads.apply(lambda x: df['seq'][i][x['start']:x['end']], 1), columns=['seq'])
                    reads = reads.value_counts()
                    if replicates[p][r] is None:
                        replicates[p][r] = reads
                    else:
                        replicates[p][r] = replicates[p][r].add(reads, fill_value=0)
    for r in range(args.n_replicates):
        df = merge_pre_and_post_counts(replicates['pre'][r], replicates['post'][r], count_column='0')
        if args.save_file is not None:
            save_file, ext = os.path.splitext(args.save_file)
            save_file = save_file + '_r{}'.format(r) + ext
            print('Saving replicate {} of simulated short reads to {}'.format(r, save_file))
            if args.shuffle:
                df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
            else:
                df.sort_values('post_count', ascending=False, inplace=True)
            df.to_csv(save_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', help='path to CSV file containing library fitnesses', type=str)
    parser.add_argument('--fitness_column', help='column name in data_file', type=str)
    parser.add_argument('--total_reads', default=int(4.6e7), help='total number of reads to generate', type=int)
    parser.add_argument('--read_length', default=100, help='length of reads to generate', type=int)
    parser.add_argument('--dirichlet_concentration', default=1, help='(scalar) concentration parameter for Dirichlet dist', type=float)
    parser.add_argument('--n_replicates', default=1, help='number of replicates of count data to generate', type=int)
    parser.add_argument('--shuffle', help='save data in shuffled order', action='store_true')
    parser.add_argument('--standardize', help='standardize fitness_column before simulating library proportions', action='store_true')
    parser.add_argument('--save_file', help='output path to which to save generated counts', type=str)
    args = parser.parse_args()
    main(args)
