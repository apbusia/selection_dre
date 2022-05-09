import os
import argparse
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import tensorflow as tf
from tensorflow import keras
from scipy.special import log_softmax
import data_prep
import modeling
import evaluation_utils
from run_models import disable_gpu


def get_model_type(model_path):
    if 'linear_classifier' in model_path:
        return 'logistic_linear'
    if 'linear' in model_path:
        return 'linear'
    if 'ann_classifier' in model_path:
        return 'logistic_ann'
    if 'ann' in model_path:
        return 'ann'
    raise ValueError('model type not identified from path: {}'.format(model_path))


def get_encoding_type(model_path):
    if 'neighbors' in model_path:
        return 'neighbors'
    if 'pairwise' in model_path:
        return 'pairwise'
    if 'is' in model_path:
        return 'is'
    raise ValueError('model\'s encoding type not identified from path: {}'.format(model_path))


def get_model_predictions(model, model_type, seqs, encoding, batch_size=256):
    n_samples = len(seqs)
    if model_type in ['linear', 'ann']:
        enrich_scores = np.zeros((n_samples, 2))
        classes = None
        counts = None
    elif model_type in ['logistic_linear', 'logistic_ann']:
        enrich_scores = None
        classes = np.zeros(n_samples)
        counts = np.ones(n_samples)
    xgen = modeling.get_dataset(
        seqs, np.arange(n_samples), encoding, enrich_scores, classes, counts, batch_size=batch_size, shuffle=False)
#     with tf.device('/cpu:0'):
#         predictions = model.predict(xgen)
    predictions = model.predict(xgen)
    return predictions


def logits_to_log_density_ratio(preds):
    preds = log_softmax(preds, axis=1) # Converts logits to log probabilities
    log_dr = preds[:, 1] - preds[:, 0] # Pre <-> 0, Post <-> 1
    return log_dr


def main(args):
    save_path = args.save_path
    if save_path is not None and os.path.exists(save_path):
        raise IOError('Saved file already exists at {}. Choose another save_path.'.format(save_path))
    
    data_path, model_paths = args.data_path, args.model_paths
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
    
    results = {}
    results['meta'] = {
        'data_path': data_path,
        'model_paths': [model_path for model_path in model_paths],
        'encodings': [get_encoding_type(model_path) for model_path in model_paths],
    }
    idx_paths = args.idx_paths
    if idx_paths is not None:
        results['meta']['idx_paths'] = list(idx_paths)
    
    results['metrics'] = defaultdict(list)
    for i, model_path in enumerate(model_paths):
        disable_gpu([0, 1])
        keras.backend.clear_session()
        
        print('\nUsing model {}'.format(model_path))
        model_type = get_model_type(model_path)
        if idx_paths is not None:
            idx_path = idx_paths[i]
            print('\n\tUsing indices {}'.format(idx_path))
            test_idx = np.load(idx_path)
            if 'logistic' in model_type:
                # Adjust logistic indices for fact that dataset was duplicated during training.
                test_idx = test_idx[test_idx < len(df)]
            truth = enrich_scores[test_idx]
            seqs = df['seq'].iloc[test_idx].reset_index(drop=True)
        else:
            truth = enrich_scores
            seqs = df['seq']
        
        print('\n\tComputing metrics')
        model = keras.models.load_model(model_path)
        encoding_type = get_encoding_type(model_path)
        if encoding_type == 'pairwise':
            encoding = data_prep.index_encode_is_plus_pairwise
        elif encoding_type == 'is':
            encoding = data_prep.index_encode
        elif encoding_type == 'neighbors':
            encoding = data_prep.index_encode_is_plus_neighbors

        print('\n\tGetting model predictions...')
        preds = get_model_predictions(model, model_type, seqs, encoding)
        del model
        if 'logistic' in model_type:
            preds = logits_to_log_density_ratio(preds)
        else:
            preds = preds.flatten()
        metrics = evaluation_utils.get_eval_metrics(truth, preds)
        print('\n\tCurrent metrics:')
        evaluation_utils.print_eval_metrics(metrics)
        for k in metrics.keys():
            results['metrics'][k].append(metrics[k])
        fracs = np.arange(0, 1, 0.01)
        results['meta']['culled_fracs'] = fracs
        culled_pearson = evaluation_utils.calculate_culled_correlation(preds, truth, fracs)
        results['metrics']['culled_pearson'].append(culled_pearson)
        culled_spearman = evaluation_utils.calculate_culled_correlation(preds, truth, fracs, correlation_type='spearman')
        results['metrics']['culled_spearman'].append(culled_spearman)
        # NDCG expects positive groundtruth, so pass enrichment instead of log-enrichment.
        culled_ndcg = evaluation_utils.calculate_culled_ndcg(np.exp(preds), np.exp(truth), fracs)
        results['metrics']['culled_ndcg'].append(culled_ndcg)

    print('\nFinal evaluation metrics:')
    evaluation_utils.print_eval_metrics(results['metrics'])
        
    if save_path is not None:
        print('\nSaving results to {}'.format(save_path))
        np.save(save_path, results)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='path to counts dataset', type=str)
    parser.add_argument('model_paths', help='path to trained Keras predictive model', nargs='+', type=str)
#     parser.add_argument('--pre_column', default='count_pre', help='column name in counts dataset', type=str)
#     parser.add_argument('--post_column', default='count_post', help='column name in counts dataset', type=str)
    parser.add_argument('--enrichment_column', help='column name in counts dataset', type=str)
    parser.add_argument('--idx_paths', help='path to file containing subset of test indices', nargs='+', type=str)
    parser.add_argument('--save_path', help='path to which to save output', type=str)
    args = parser.parse_args()
    main(args)
