import os
import argparse
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import KFold
import pre_process


DATA_DIR = "../data/counts/"
SEED = 7


def calculate_enrichment_scores(n_pre, n_post, N_pre, N_post):
    """
    Calculates the mean and approximate variance
    of enrichment scores. The variance is that implied
    by the ratio of two binomials.
    """
    f_pre = n_pre / N_pre
    f_post = n_post / N_post
    mean_true = np.log(f_post/f_pre)
    approx_sig = (1/n_post) * (1-f_post) + (1/n_pre) * (1-f_pre)
    return np.array([mean_true, approx_sig]).T


def index_encode(seq):
    """
    Returns an integer vector of indices representing
    the input sequence.
    """
    l = len(seq)
    out = [pre_process.AA_IDX[s] for s in seq]
    return np.array(out)


def one_hot_encode(seq):
    """
    Returns a one-hot encoded matrix representing
    the input sequence.
    """
    l = len(seq)
    m = len(pre_process.AA_ORDER)
    out = np.zeros((l, m))
    for i in range(l):
        out[i, pre_process.AA_IDX[seq[i]]] = 1
    return out.flatten()


def index_encode_neighbors(seq, shift=False):
    """
    Returns an integer vector where each entry represents
    an index encoding of a particular pair of amino
    acid at a pair of neighboring positions.
    """
    l = len(seq)
    neighbors = [(i, i+1) for i in range(l - 1)]
    m = len(pre_process.AA_ORDER)
    idx = np.reshape(np.arange(m ** 2, dtype=int), (m, m))
    out = np.zeros(len(neighbors), dtype=int)
    for i, (c1, c2) in enumerate(neighbors):
        s1 = seq[c1]
        s2 = seq[c2]
        out[i] = idx[pre_process.AA_IDX[s1], pre_process.AA_IDX[s2]]
    # Shift indices to avoid overlap with indices encoding independent sites.
    if shift:
        out = out + m
    return out.flatten()


def encode_neighbors(seq):
    """
    Returns a binary matrix where each entry represents
    the presence or absence of a particular pair of amino
    acid at a pair of neighboring positions
    """
    l = len(seq)
    neighbors = [(i, i+1) for i in range(l - 1)]
    m = len(pre_process.AA_ORDER)
    out = np.zeros((len(neighbors), m, m))
    for i, (c1, c2) in enumerate(neighbors):
        s1 = seq[c1]
        s2 = seq[c2]
        out[i, pre_process.AA_IDX[s1], pre_process.AA_IDX[s2]] = 1
    return out.flatten()


def index_encode_pairwise(seq, shift=False):
    """
    Returns an integer vector where each entry represents
    an index encoding of a particular pair of amino
    acid at a pair of positions.
    """
    l = len(seq)
    combos = list(combinations(range(l), 2))
    m = len(pre_process.AA_ORDER)
    idx = np.reshape(np.arange(m ** 2, dtype=int), (m, m))
    out = np.zeros(len(combos), dtype=int)
    for i, (c1, c2) in enumerate(combos):
        s1 = seq[c1]
        s2 = seq[c2]
        out[i] = idx[pre_process.AA_IDX[s1], pre_process.AA_IDX[s2]]
    # Shift indices to avoid overlap with indices encoding independent sites.
    if shift:
        out = out + m
    return out.flatten()


def encode_pairwise(seq):
    """
    Returns a binary matrix where each entry represents
    the presence or absence of a particular pair of amino
    acid at a pair of positions
    """
    l = len(seq)
    combos = list(combinations(range(l), 2))
    m = len(pre_process.AA_ORDER)
    out = np.zeros((len(combos), m, m))
    for i, (c1, c2) in enumerate(combos):
        s1 = seq[c1]
        s2 = seq[c2]
        out[i, pre_process.AA_IDX[s1], pre_process.AA_IDX[s2]] = 1
    return out.flatten()


def index_encode_is_plus_pairwise(seq):
    """
    Combines index encodings of independent sites
    and pairwise terms into single integer vector.
    """
    indep_sites = index_encode(seq)
    pairwise = index_encode_pairwise(seq)
    both = np.concatenate((indep_sites, pairwise))
    return both


def encode_one_plus_pairwise(seq):
    """
    Combines the one-hot and pairwise encodings
    into a single binary vector
    """
    one_hot = one_hot_encode(seq)
    pairwise = encode_pairwise(seq)
    both = np.concatenate((one_hot, pairwise))
    return both


def index_encode_is_plus_neighbors(seq):
    """
    Combines index encodings of independent sites
    and neighbor terms into single integer vector.
    """
    indep_sites = index_encode(seq)
    neighbors = index_encode_neighbors(seq)
    both = np.concatenate((indep_sites, neighbors))
    return both


def encode_one_plus_neighbors(seq):
    """
    Combines the one-hot and neighbor encodings
    into a single binary vector
    """
    one_hot = one_hot_encode(seq)
    neighbors = encode_neighbors(seq)
    both = np.concatenate((one_hot, neighbors))
    return both


def get_example_encoding(encoding_function, nuc=False):
    """
    Returns an example of the possible sequence encodings,
    given the encoding function. Should be used for sizing
    purposes.
    """
    if nuc:
        seq = "".join(['A']*21)
    else:
        seq = "".join(['A']*7)
    return encoding_function(seq)


def load_data(pre_file, post_file, 
              seq_column='seq', count_column='count'):
    """Loads pre- and post- count databases and combine into one merged DataFrame"""
    pre_df = pd.read_csv(pre_file)[[seq_column, count_column]]
    post_df = pd.read_csv(post_file)[[seq_column, count_column]]
    pre_groups = pre_df.groupby(seq_column)
    pre_df = pre_groups.sum().reset_index()
    post_groups = post_df.groupby(seq_column)
    post_df = post_groups.sum().reset_index()
    merged_df = pd.merge(pre_df, post_df, how='outer', on=seq_column, suffixes=('_pre', '_post')).fillna(0)
    merged_df = merged_df.rename(columns={count_column + "_pre": 'count_pre', 
                                          count_column + "_post": 'count_post', 
                                          seq_column: 'seq'})
    merged_df = merged_df.loc[~merged_df['seq'].str.contains('X')]
    merged_df = merged_df.reset_index()
    return merged_df


def prepare_data(merged_df):
    """
    Converts merged DataFrame of count data into list of sequences
    and (enrichment score, variance) pairs.
    """
    n_pre = np.array(merged_df['count_pre'] + 1)
    n_post = np.array(merged_df['count_post'] + 1)
    N_pre = n_pre.sum()
    N_post = n_post.sum()

    enrich_scores = calculate_enrichment_scores(n_pre, n_post, N_pre, N_post)
    sequences = list(merged_df['seq'])
    return sequences, enrich_scores


def featurize_and_transform(sequences, enrich_scores, 
                            encoding_func=encode_one_plus_pairwise):
    """
    Encodes sequences given an encoding function and calculates sample weights
    from (enrichment score, variance) pairs.
    """
    d = get_example_encoding(encoding_func).shape[0]
    X = np.zeros((len(sequences), d), dtype=np.int8)
    for i in range(len(X)):
        X[i] = encoding_func(sequences[i])

    y = enrich_scores[:, 0]
    sample_weights = 1/(2*enrich_scores[:, 1])
    return X, y, sample_weights


def get_nnk_p(n_aa=7):
    """
    Get NNK nucleotide probabilities.
    """
    p_nnk = np.ones((3, 4))
    p_nnk[:2] *= 0.25
    p_nnk[2, pre_process.NUC_IDX['A']] = 0
    p_nnk[2, pre_process.NUC_IDX['C']] = 0
    p_nnk[2] *= 0.5
    p_nnk = np.tile(p_nnk.T, n_aa).T
    return p_nnk


def main(args):
    np.random.seed(SEED)
    
    savefile = args.save_name
    print('Preparing data in {}'.format(args.counts_file))
    data_df = pd.read_csv(args.counts_file)[['seq', args.pre_column, args.post_column]]
    
    # Compute and store sequence encodings.
    encodings = [('is', index_encode), ('neighbors', index_encode_is_plus_neighbors), ('pairwise', index_encode_is_plus_pairwise)]
    seqs = data_df['seq']
    for enc_name, enc_fn in encodings:
        print('Computing "{}" encoding...'.format(enc_name))
        feat = np.stack(seqs.apply(enc_fn))
        feat = pd.DataFrame(feat, columns=['{}_{}'.format(enc_name, i) for i in range(feat.shape[1])])
        data_df = pd.concat([data_df, feat], axis=1)
        del feat
    
    # Split data into specified number of folds.
    if args.n_folds == 1:
        folds = [(np.random.permutation(len(data_df)), None)] if args.shuffle else [(np.arange(len(data_df)), None)]
    else:
        folds = KFold(n_splits=args.n_folds, shuffle=args.shuffle, random_state=SEED).split(data_df)
    for i, (train_idx, test_idx) in enumerate(folds):
        print('Preparing fold {}...'.format(i+1))
        if savefile is not None and test_idx is not None:
            np.save('models/' + os.path.splitext(os.path.split(savefile)[1])[0] + '_fold{}_test_idx.npy'.format(i), test_idx)
        fold_df = data_df.iloc[train_idx]

        # Optionally removed unobserved sequences.
        if not args.retain_unsequenced:
            fold_df = fold_df[(fold_df[args.pre_column] > 0) | (fold_df[args.post_column] > 0)]
        
        # Compute and store enrichment scores.
        pre_counts = fold_df[args.pre_column] + 1
        post_counts = fold_df[args.post_column] + 1
        enrich_scores = calculate_enrichment_scores(pre_counts, post_counts, pre_counts.sum(), post_counts.sum())
        mean_en, var_en = enrich_scores[:, 0], enrich_scores[:, 1]
        if args.normalize:
            mu_en = np.mean(mean_en)
            sig_en = np.std(mean_en)
            mean_en = (mean_en - mu_en) / sig_en
            var_en = (np.sqrt(var_en) / sig_en)**2
        fold_df['enrichment'] = mean_en
        fold_df['enrichment_var'] = var_en

        # Optionally re-weight observed counts.
        if args.normalize:
            count_scale = max(np.amax(fold_df[args.pre_column]), np.amax(fold_df[args.post_column]))
            fold_df[args.pre_column] = fold_df[args.pre_column] / count_scale
            fold_df[args.post_column] = fold_df[args.post_column] / count_scale
    
        if savefile is not None:
            savename, ext = os.path.splitext(savefile)
            fold_df.to_csv(savename + '_fold{}'.format(i) + ext, index=False, compression='gzip')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('counts_file', help='path to counts dataset', type=str)
    parser.add_argument('--pre_column', default='count_pre', help='column name in counts dataset', type=str)
    parser.add_argument('--post_column', default='count_post', help='column name in counts dataset', type=str)
    parser.add_argument('--normalize', help="normalize counts", action='store_true')
    parser.add_argument('--n_folds', default=3, help='number of folds to split into; pass n_folds=1 to keep full data.', type=int)
    parser.add_argument('--retain_unsequenced', help='retain seq with pre_count=post_count=0 in training data', action='store_true')
    parser.add_argument('--shuffle', help='whether to shuffle combined dataset(s) before saving', action='store_true')
    parser.add_argument('--save_name', help='filename to which to save prepared data', type=str)
    args = parser.parse_args()
    main(args)