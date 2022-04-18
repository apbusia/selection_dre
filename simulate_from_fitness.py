import argparse
import numpy as np
import pandas as pd
import data_prep


SEED = 7


def main(args):
    np.random.seed(SEED)
    
    print('Loading data from file: {}'.format(args.data_file))
    df = pd.read_csv(args.data_file)
    
    print('Using {} as fitness'.format(args.fitness_column))
    df = df[['seq', args.fitness_column]]
    fitness = df[args.fitness_column].values
    if args.normalize:
        fitness = (fitness - np.mean(fitness)) / np.std(fitness)
    
    print('Computing \'true\' library proportions...')
    # Draw initial library proportions from a Dirichlet.
    alpha = np.ones(len(df))
    p = np.random.dirichlet(args.dirichlet_concentration * alpha)
    df['pre_p'] = p
    # Compute post-selection proportions using initial proportions and fitness.
    p = np.exp(fitness) * p
    # If needed, rescale so post-selection proportions sum to 1 and recomputed 'true' fitness.
    p = p / np.sum(p)
    df['post_p'] = p
    df['true_enrichment'] = np.log(df['post_p'].values / df['pre_p'].values)
    
    print('Sampling sequence counts...')
    N_pre = args.total_reads
    N_post = args.total_reads
    for i in range(args.n_replicates):
        df['pre_count_{}'.format(i)] = np.random.multinomial(N_pre, df['pre_p'].values)
        df['post_count_{}'.format(i)] = np.random.multinomial(N_post, df['post_p'].values)
    
    # Save simulated count data to file.
    if args.save_file is not None:
        print('Saving updated library with simulated counts to {}'.format(args.save_file))
        if args.shuffle:
            df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
        else:
            df.sort_values('post_p', ascending=False, inplace=True)
        df.to_csv(args.save_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', help='path to CSV file containing library fitnesses', type=str)
    parser.add_argument('fitness_column', help='column name in data_file', type=str)
    parser.add_argument('--total_reads', default=int(1e7), help='total number of reads to generate', type=int)
    parser.add_argument('--dirichlet_concentration', default=1, help='(scalar) concentration parameter for Dirichlet dist', type=float)
    parser.add_argument('--n_replicates', default=1, help='number of replicates of count data to generate', type=int)
    parser.add_argument('--shuffle', help='save data in shuffled order', action='store_true')
    parser.add_argument('--normalize', help='normalize fitness_column before simulating library proportions', action='store_true')
    parser.add_argument('--save_file', help='output path to which to save generated counts', type=str)
    args = parser.parse_args()
    main(args)
