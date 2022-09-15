import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


SEED = 7


def main(args):
    savefile = args.save_file
    
    long_df = pd.read_csv(args.long_data_file)
    long_df = long_df[['seq', args.pre_column, args.post_column]]
    long_df = long_df.rename(columns={args.pre_column: 'pre_count',
                                      args.post_column: 'post_count',
                                      'seq': 'seq'})
    
    short_df = pd.read_csv(args.short_data_file)
    short_df = short_df[['seq', 'pre_count', 'post_count']]
    if args.reweight_short:
        short_df['pre_count'] = short_df['pre_count'] * long_df['pre_count'].sum() / short_df['pre_count'].sum()
        short_df['post_count'] = short_df['post_count'] * long_df['post_count'].sum() / short_df['post_count'].sum()
    

    combined_dfs = [None] * args.n_folds
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)
    
    # Divide long read data.
    for i, (train_idx, test_idx) in enumerate(kf.split(long_df)):
        if savefile is not None:
            np.save('models/' + os.path.splitext(os.path.split(savefile)[1])[0] + 'fold{}_test_idx.npy'.format(i), test_idx)
        if not args.retain_unsequenced:
            train_idx = train_idx[(long_df['pre_count'][train_idx] > 0) | (long_df['post_count'][train_idx] > 0)]
        combined_dfs[i] = long_df.iloc[train_idx]
    
    # Divide short read data and combine with long read data.
    for i, (train_idx, test_idx) in enumerate(kf.split(short_df)):
        combined_dfs[i] = combined_dfs[i].append(short_df.iloc[train_idx], ignore_index=True)
    
    # Pad sequences, if necessary.
    if args.pad_sequences:
        seq_len = long_df['seq'].str.len().max()
        for i in range(len(combined_dfs)):
            combined_dfs[i]['seq'] = combined_dfs[i]['seq'].str.pad(width=seq_len, fillchar='*', side='right')
    
    # Save combined datasets to file.
    if savefile is not None:
        savename, ext = os.path.splitext(savefile)
        for i, df in enumerate(combined_dfs):
            if args.shuffle:
                df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
            df.to_csv(savename + '_fold{}'.format(i) + ext, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('long_data_file', help='path to counts dataset of long reads', type=str)
    parser.add_argument('short_data_file', help='path to counts dataset of short reads', type=str)
    parser.add_argument('--pre_column', default='count_pre', help='column name in long read counts dataset', type=str)
    parser.add_argument('--post_column', default='count_post', help='column name in long read counts dataset', type=str)
    parser.add_argument("--pad_sequences", help="pad all sequences to the same length", action='store_true')
    parser.add_argument("--reweight_short", help="reweight short read data to equalize total reads", action='store_true')
    parser.add_argument("--n_folds", default=1, help="number of folds into which to split the (combined) dataset", type=int)
    parser.add_argument("--save_file", help="path to which to save combined dataset", type=str)
    parser.add_argument("--retain_unsequenced", help="retain seq with pre_count=post_count=0 in training data", action='store_true')
    parser.add_argument("--shuffle", help="whether to shuffle combined dataset(s) before saving", action='store_true')
    args = parser.parse_args()
    main(args)
