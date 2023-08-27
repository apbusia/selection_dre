import os
import argparse
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import data_prep
import evaluation_utils


def main(args):
    save_path = args.save_path
    if save_path is not None and os.path.exists(save_path):
        raise IOError('Saved file already exists at {}. Choose another save_path.'.format(save_path))
    
    data_path = args.data_path
    print('\nLoading data from {}'.format(data_path))
    df = pd.read_csv(data_path)
    if args.enrichment_column is not None:
        print('Using enrichment scores in {}.'.format(args.enrichment_column))
        enrich_scores = df[args.enrichment_column].values
    else:
        print('Computing enrichment scores from \'count_pre\' and \'count_post\'.')
        pre_counts = df['count_pre'].values + 1
        post_counts = df['count_post'].values + 1
        enrich_scores = data_prep.calculate_enrichment_scores(pre_counts, post_counts, pre_counts.sum(), post_counts.sum())
        enrich_scores = enrich_scores[:, 0]
    n_replicates = args.n_replicates
    count_cols = ['pre_count_{}'.format(n) for n in range(n_replicates)] + ['post_count_{}'.format(n) for n in range(n_replicates)]
    df = df[count_cols]
    count_sum = df.sum(axis=1)
    df = df.set_axis(['seq{}'.format(i) for i in range(len(df))])
    clinical_df = pd.DataFrame(data={'condition': ['pre'] * n_replicates + ['post'] * n_replicates}, index=count_cols)
    
    results = {}
    idx_paths = args.idx_paths
    n_folds = 1 if idx_paths is None else len(idx_paths)
    results['meta'] = {
        'data_path': data_path,
        'model_paths': ['DEseq2'] * n_folds,
    }
    if idx_paths is not None:
        idx_paths = list(idx_paths)
        results['meta']['idx_paths'] = idx_paths
    results['metrics'] = defaultdict(list)

    for i in range(n_folds):
        print('\nFold {}'.format(i))
        if idx_paths is not None:
            idx_path = idx_paths[i]
            print('\n\tUsing indices {}'.format(idx_path))
            test_idx = np.load(idx_path)
            # Invert test_idx to correspond to the observed sequences in the train set
            test_idx = np.delete(np.arange(len(df)), test_idx)
            mask = np.full(len(test_idx), False)
            # Remove unobserved sequencecs with 0 counts in both conditions
            mask = mask | (count_sum[test_idx] > 0)
            test_idx = test_idx[mask]
        else:
            test_idx = np.arange(len(df))[count_sum > 0]
            
        truth = enrich_scores[test_idx]

        fold_df = df.iloc[test_idx] #+ 1 # pseudo-count to avoid divide by zero.
        fold_df = fold_df.T
        dds = DeseqDataSet(counts=fold_df, clinical=clinical_df, design_factors='condition')
        dds.deseq2()
        preds = -1 * dds.varm['LFC'].condition_pre_vs_post.values
            
        metrics = evaluation_utils.get_eval_metrics(truth, preds)
        print('\n\tCurrent metrics:')
        evaluation_utils.print_eval_metrics(metrics)
        for k in metrics.keys():
            results['metrics'][k].append(metrics[k])
        print('\n\tComputing top-K metrics...')
        fracs = np.linspace(0, 1, args.num_fracs, endpoint=False)
        results['meta']['culled_fracs'] = fracs
        results['metrics']['culled_pearson'].append(
            evaluation_utils.calculate_culled_correlation(preds, truth, fracs))
        results['metrics']['culled_spearman'].append(
            evaluation_utils.calculate_culled_correlation(preds, truth, fracs, correlation_type='spearman'))
        results['metrics']['culled_mse'].append(
            evaluation_utils.calculate_culled_correlation(preds, truth, fracs, correlation_type='mse'))
        try:
            # NDCG expects positive groundtruth, so pass enrichment instead of log-enrichment.
            culled_ndcg = evaluation_utils.calculate_culled_ndcg(np.exp(preds), np.exp(truth), fracs)
            results['metrics']['culled_ndcg'].append(culled_ndcg)
        except:
            # NDCG fails when predictions are very large and np.exp(.) overflows.
            print('\n\tUnable to compute culled NDCG.')
            pass

    print('\nFinal evaluation metrics:')
    evaluation_utils.print_eval_metrics(results['metrics'])
        
    if save_path is not None:
        print('\nSaving results to {}'.format(save_path))
        np.save(save_path, results)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='path to counts dataset', type=str)
    parser.add_argument('--enrichment_column', help='column name in counts dataset', type=str)
    parser.add_argument('--idx_paths', help='path to file containing subset of test indices', nargs='+', type=str)
    parser.add_argument('--num_fracs', default=100, help='number of K for computing top K performance metrics', type=int)
    parser.add_argument('--save_path', help='path to which to save output', type=str)
    parser.add_argument('--n_replicates', default=3, help='number of replicates in counts dataset', type=int)
    args = parser.parse_args()
    main(args)
