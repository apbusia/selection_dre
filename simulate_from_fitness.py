import argparse
import numpy as np
import pandas as pd
import data_prep


SEED = 7


def quantile_normalization(df):
    # https://stackoverflow.com/questions/37935920/quantile-normalization-on-pandas-dataframe
    ranks = df.rank(method='first').stack()
    rank_mean = df.stack().groupby(ranks).mean()
    # Add interpolated values in between ranks.
    finer_ranks = ((rank_mean.index+0.5).to_list() + rank_mean.index.to_list())
    rank_mean = rank_mean.reindex(finer_ranks).sort_index().interpolate()
    return df.rank(method='average').stack().map(rank_mean).unstack()


def simulate_true_proportions(fitness, dirichlet_concentration=1):
    # Draw initial library proportions from a Dirichlet.
    alpha = np.ones(len(fitness))
    pre_p = np.random.dirichlet(dirichlet_concentration * alpha)
    # Compute post-selection proportions using initial proportions and fitness.
    post_p = np.exp(fitness) * pre_p
    # If needed, rescale so post-selection proportions sum to 1 and recompute 'true' fitness.
    post_p = post_p / np.sum(post_p)
    true_enrichment = np.log(post_p / pre_p)
    return pre_p, post_p, true_enrichment


def main(args):
    np.random.seed(SEED)
    
    print('Loading data from file: {}'.format(args.data_file))
    df = pd.read_csv(args.data_file)
    
    print('Using {} as fitness'.format(args.fitness_column))
    df = df[['seq', args.fitness_column]]
    fitness = df[args.fitness_column].values
    if args.q_norm_file is not None:
        print('Applying quantile normalization using log-enrichments from {}'.format(args.q_norm_file))
        counts_df = pd.read_csv(args.q_norm_file)
        pre_counts = counts_df['count_pre'] + 1
        post_counts = counts_df['count_post'] + 1
        del counts_df
        enrich_scores = data_prep.calculate_enrichment_scores(pre_counts, post_counts, pre_counts.sum(), post_counts.sum())[:, 0]
        qn_df = pd.DataFrame(np.column_stack((fitness, enrich_scores)), columns = ['fitness','enrich_scores'])
        qn_df = quantile_normalization(qn_df)
        fitness = qn_df['fitness']
    if args.standardize:
        print('Standardizing fitness to have 0 mean and unit variance...')
        fitness = (fitness - np.mean(fitness)) / np.std(fitness)
    
    print('Computing \'true\' library proportions...')
    pre_p, post_p, true_enrichment = simulate_true_proportions(fitness, args.dirichlet_concentration)
    df['pre_p'] = pre_p
    df['post_p'] = post_p
    df['true_enrichment'] = true_enrichment
    
    N_pre = args.total_reads
    N_post = args.total_reads
    print('Sampling sequence counts ({:.1e} reads)...'.format(N_pre))
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
    parser.add_argument('--total_reads', default=int(4.6e7), help='total number of reads to generate', type=int)
    parser.add_argument('--dirichlet_concentration', default=1, help='(scalar) concentration parameter for Dirichlet dist', type=float)
    parser.add_argument('--n_replicates', default=1, help='number of replicates of count data to generate', type=int)
    parser.add_argument('--shuffle', help='save data in shuffled order', action='store_true')
    parser.add_argument('--q_norm_file', help='path to counts dataset to use to quantile normalize fitness_column', type=str)
    parser.add_argument('--standardize', help='standardize fitness_column before simulating library proportions', action='store_true')
    parser.add_argument('--save_file', help='output path to which to save generated counts', type=str)
    args = parser.parse_args()
    main(args)
