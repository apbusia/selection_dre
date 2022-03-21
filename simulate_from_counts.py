import argparse
import numpy as np
import pandas as pd
import data_prep


SEED = 7


def main(args):
    np.random.seed(SEED)
    
    print('Loading data from file: {} and {}'.format(args.pre_file, args.post_file))
    counts_df = data_prep.load_data(args.pre_file, args.post_file)
    count_pre, count_post = counts_df['count_pre'].values, counts_df['count_post'].values

    print('Computing \'true\' library proportions...')
    # Draw initial library proportions from a Dirichlet.
    alpha = count_pre + 1
    p = np.random.dirichlet(alpha)
    counts_df['pre_p'] = p
    
    # Draw post-selection library proportions from a Dirichlet.
    alpha = count_post + 1
    p = np.random.dirichlet(alpha)
    counts_df['post_p'] = p
    
    # Compute 'true' fitness.
    counts_df['true_enrichment'] = np.log(counts_df['post_p'].values / counts_df['pre_p'].values)
    
    print('Sampling sequence counts...')
    N_pre = args.total_reads if args.total_reads is not None else np.sum(count_pre)
    N_post = args.total_reads if args.total_reads is not None else np.sum(count_post)
    for i in range(args.n_replicates):
        counts_df['pre_count_{}'.format(i)] = np.random.multinomial(N_pre, counts_df['pre_p'].values)
        counts_df['post_count_{}'.format(i)] = np.random.multinomial(N_post, counts_df['post_p'].values)
    
    # Save simulated count data to file.
    if args.save_file is not None:
        print('Saving updated library with simulated counts to {}'.format(args.save_file))
        if args.shuffle:
            counts_df = counts_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
        else:
            counts_df.sort_values('post_p', ascending=False, inplace=True)
        counts_df.to_csv(args.save_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pre_file', help='path to CSV file containing "pre" counts dataset', type=str)
    parser.add_argument('post_file', help='path to CSV file containing "post" counts dataset', type=str)
    parser.add_argument('--total_reads', help='total number of reads to generate; default same as counts_file', type=int)
    parser.add_argument('--n_replicates', default=1, help='number of replicates of count data to generate', type=int)
    parser.add_argument('--shuffle', help='whether to save data in shuffled order', action='store_true')
    parser.add_argument('--save_file', help='output path to which to save generated counts', type=str)
    args = parser.parse_args()
    main(args)
