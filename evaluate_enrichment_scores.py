import os
import argparse
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import data_prep
import evaluation_utils


def main(args):
    save_path = args.save_path
    if save_path is not None and os.path.exists(save_path):
        raise IOError('Saved file already exists at {}. Choose another save_path.'.format(save_path))
    
    data_path = args.data_path
    print('\nLoading data from {}'.format(data_path))
    df = pd.read_csv(data_path)
    n_seqs = len(df)
#     all_seqs = df['seq']
    
    pre_counts_list = [df[c] for c in args.pre_columns]
    post_counts_list = [df[c] for c in args.post_columns]
    enrich_scores_list = []
    for i, post_counts in enumerate(post_counts_list):
        # Shared pre-selection pool.
        if len(pre_counts_list) == 1:
            pre_counts = pre_counts_list[0] + 1
        # Different pre-selection pools.
        else:
            pre_counts = pre_counts_list[i] + 1
        post_counts = post_counts + 1
        enrich_scores = data_prep.calculate_enrichment_scores(pre_counts, post_counts, pre_counts.sum(), post_counts.sum())
        enrich_scores_list.append(enrich_scores[:, 0])
    if len(enrich_scores_list) > 1:
        print('Using average enrichment score')
        enrich_scores = np.mean(enrich_scores_list, axis=0)
    else:
        enrich_scores = enrich_scores_list[0]
    print('Using fitness scores in {}.'.format(args.fitness_column))
    fitness_scores = df[args.fitness_column].values
    del df
    
    results = {}
    results['meta'] = {
        'data_path': data_path,
        'model_paths': ['pre_{}'.format(c) for c in args.pre_columns] + ['post_'.format(c) for c in args.post_columns],
        'encodings': [None],
    }
    idx_paths = args.idx_paths
    if idx_paths is not None:
        results['meta']['idx_paths'] = list(idx_paths)
    else:
        idx_paths = [None]
    
    results['metrics'] = defaultdict(list)
    for i, idx_path in enumerate(idx_paths):
        truth = fitness_scores
        if idx_path is not None:
            print('\n\tUsing indices {}'.format(idx_path))
            test_idx = np.load(idx_path)
            if args.evaluate_train:
                print('\n\tInverting indices to evaluate on training examples')
                test_idx = np.delete(np.arange(n_seqs), test_idx)
            truth = fitness_scores[test_idx]
            preds = enrich_scores[test_idx]
        mask = np.full(len(truth), False)
        for pc in pre_counts_list + post_counts_list:
            mask = mask | (pc[test_idx] > 0)
        truth, preds = truth[mask], preds[mask]
#         else:
#             if args.output_idx == -1:
#                 preds = np.mean(preds, axis=0).flatten()
#             elif type(preds) == list:
#                 # Predictions for multi-output model.
#                 preds = preds[args.output_idx].flatten()
#             else:
#                 # Predictions for single output model.
#                 preds = preds.flatten()
               
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
    parser.add_argument('--pre_columns', default='count_pre', help='column name in counts dataset', type=str, nargs='+')
    parser.add_argument('--post_columns', default='count_post', help='column name in counts dataset', type=str, nargs='+')
    parser.add_argument('--fitness_column', help='column name in counts dataset', type=str)
#     parser.add_argument('--output_idx', default=0, help='index of model output to evaluate for multi-output models', type=int)
    parser.add_argument('--idx_paths', help='path to file containing subset of test indices', nargs='+', type=str)
    parser.add_argument('--evaluate_train', help='whether or not to "invert" test indices in idx_paths to evaluate performance on training examples', action='store_true')
    parser.add_argument('--num_fracs', default=100, help='number of K for computing top K performance metrics', type=int)
    parser.add_argument('--save_path', help='path to which to save output', type=str)
    args = parser.parse_args()
    main(args)
