import os
import argparse
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import tensorflow as tf
from tensorflow import keras
from scipy.special import log_softmax
from scipy.stats import t
import data_prep
import modeling
import evaluation_utils
import evaluate_models
from run_models import disable_gpu


def williams_t_test(r_jk, r_jh, r_kh, N, two_sided=False):
#     Steiger, J.H. (1980), Tests for comparing elements of a correlation matrix, Psychological Bulletin, 87, 245-251.
    det_R = 1 - r_jk ** 2 - r_jh ** 2 - r_kh ** 2 + 2 * r_jk * r_jh * r_kh
    bar_r = (r_jk + r_jh) / 2
    t2 = (r_jk - r_jh) * np.sqrt((N - 1) * (1 + r_kh) / (2 * (N - 1) * det_R / (N - 3) + bar_r ** 2 * (1 - r_kh) ** 3))
    p = 1 - t.cdf(np.abs(t2), N - 3)
    if two_sided:
        p *= 2
    return t2, p


def main(args):
    save_path = args.save_path
    if save_path is not None and os.path.exists(save_path):
        raise IOError('Saved file already exists at {}. Choose another save_path.'.format(save_path))
    
    if args.model_paths is None and args.pred_paths is None:
        raise argparse.ArgumentTypeError('Must provide either saved predictions or models to run')
    
    if args.pred_paths is not None:
        print('\nLoading predictions from {}'.format(args.pred_paths))
        drc_preds, ler_preds, truth_vals = args.pred_paths
        drc_preds = np.load(drc_preds)
        ler_preds = np.load(ler_preds)
        truth_vals = np.load(truth_vals)
    else:
        data_path, model_paths = args.data_path, args.model_paths
        print('\nLoading data from {}'.format(data_path))
        df = pd.read_csv(data_path)
        all_seqs = df['seq']
        if args.enrichment_column is not None:
            print('Using enrichment scores in {}.'.format(args.enrichment_column))
            enrich_scores = df[args.enrichment_column].values
        else:
            print('Computing enrichment scores from \'count_pre\' and \'count_post\'.')
            pre_counts = df['count_pre'].values + 1
            post_counts = df['count_post'].values + 1
            enrich_scores = data_prep.calculate_enrichment_scores(pre_counts, post_counts, pre_counts.sum(), post_counts.sum())
            enrich_scores = enrich_scores[:, 0]
        del df

        idx_paths = args.idx_paths
        drc_preds, ler_preds, truth_vals = [], [], []
        n_folds = len(model_paths) / 2
        for i, model_path in enumerate(model_paths):
            disable_gpu([0, 1, 2, 3])
            keras.backend.clear_session()

            print('\nUsing model {}'.format(model_path))
            model_type = evaluate_models.get_model_type(model_path)
            if idx_paths is not None:
                idx_path = idx_paths[i]
                print('\n\tUsing indices {}'.format(idx_path))
                test_idx = np.load(idx_path)
                if 'logistic' in model_type:
                    # Adjust logistic indices for fact that dataset was duplicated during training.
                    test_idx = test_idx[test_idx < len(all_seqs)]
                if args.evaluate_train:
                    print('\n\tInverting indices to evaluate on training examples')
                    test_idx = np.delete(np.arange(len(all_seqs)), test_idx)
                truth = enrich_scores[test_idx]
                seqs = all_seqs.iloc[test_idx].reset_index(drop=True)
            else:
                truth = enrich_scores
                seqs = all_seqs

            model = keras.models.load_model(model_path)
            encoding_type = evaluate_models.get_encoding_type(model_path)
            if encoding_type == 'pairwise':
                encoding = data_prep.index_encode_is_plus_pairwise
            elif encoding_type == 'is':
                encoding = data_prep.index_encode
            elif encoding_type == 'neighbors':
                encoding = data_prep.index_encode_is_plus_neighbors

            print('\n\tGetting model predictions...')
            preds = evaluate_models.get_model_predictions(model, model_type, seqs, encoding)
            del model
            if 'logistic' in model_type:
                if args.output_idx == -1:
                    preds = np.mean([evaluate_models.logits_to_log_density_ratio(preds, post_idx=i, pre_idx=i-1) for i in range(1, preds.shape[1])], axis=0)
                else:
                    # Add 1 to output_idx since pre is always index 0. in classification models.
                    preds = evaluate_models.logits_to_log_density_ratio(preds, post_idx=args.output_idx+1)
                drc_preds.append(preds)
            else:
                if args.output_idx == -1:
                    preds = np.mean(preds, axis=0).flatten()
                elif type(preds) == list:
                    # Predictions for multi-output model.
                    preds = preds[args.output_idx].flatten()
                else:
                    # Predictions for single output model.
                    preds = preds.flatten()
                ler_preds.append(preds)
            
            if i < n_folds:
                truth_vals.append(truth)
        max_len = np.amax([len(t) for t in truth_vals])
        drc_preds = np.array([np.pad(a, (0, max_len - len(a)), 'constant', constant_values=np.nan) for a in drc_preds])
        ler_preds = np.array([np.pad(a, (0, max_len - len(a)), 'constant', constant_values=np.nan) for a in ler_preds])
        truth_vals = np.array([np.pad(a, (0, max_len - len(a)), 'constant', constant_values=np.nan) for a in truth_vals])
            
        if save_path is not None:
            print('\nSaving predictions to {}'.format(save_path))
            fname = save_path
            np.save(fname + '_drc_preds.npy', drc_preds)
            np.save(fname + '_ler_preds.npy', ler_preds)
            np.save(fname + '_truth_vals.npy', truth_vals)
    
    # Use William's t-test, as proposed by Steiger:
    # Steiger, J.H. (1980), Tests for comparing elements of a correlation matrix, Psychological Bulletin, 87, 245-251.
    p_values = []
    cv_drc_r, cv_ler_r, cv_drc_ler_r, N = 0., 0., 0., 0
    for i, t in enumerate(truth_vals):
        t = t[~np.isnan(t)]
        drc = drc_preds[i][~np.isnan(drc_preds[i])]
        ler = ler_preds[i][~np.isnan(ler_preds[i])]
        N += len(t)
        drc_r = evaluation_utils.get_spearmanr(t, drc)
        cv_drc_r += drc_r
        ler_r = evaluation_utils.get_spearmanr(t, ler)
        cv_ler_r += ler_r
        drc_ler_r = evaluation_utils.get_spearmanr(drc, ler)
        cv_drc_ler_r += drc_ler_r
        test_stat, p_val = williams_t_test(drc_r, ler_r, drc_ler_r, len(t), two_sided=args.two_sided)
        p_values.append(p_val)
    N = int(N / len(p_values))
    cv_drc_r /= len(p_values)
    cv_ler_r /= len(p_values)
    cv_drc_ler_r /= len(p_values)
    print('DRC-fitness spearman: {}, LER-fitness spearman: {}, DRC-LER Spearman {}'.format(cv_drc_r, cv_ler_r, cv_drc_ler_r))
    test_stat, p_val = williams_t_test(cv_drc_r, cv_ler_r, cv_drc_ler_r, N, two_sided=args.two_sided)
    p_values.append(p_val)
    if save_path is not None:
        print('\nSaving p-values to {}'.format(save_path))
        fname = save_path
        np.save(fname + '_p_values.npy', np.array(p_values))
    print('Williams t-test: test statistic={}, p-value={}'.format(test_stat, p_val))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='path to counts dataset', type=str)
    parser.add_argument('--pred_paths', help='path to saved model predictions', nargs='+', type=str)
    parser.add_argument('--model_paths', help='path to trained Keras predictive model', nargs='+', type=str)
    parser.add_argument('--enrichment_column', help='column name in counts dataset', type=str)
    parser.add_argument('--output_idx', default=0, help='index of model output to evaluate for multi-output models', type=int)
    parser.add_argument('--idx_paths', help='path to file containing subset of test indices', nargs='+', type=str)
    parser.add_argument('--evaluate_train', help='whether or not to "invert" test indices in idx_paths to evaluate performance on training examples', action='store_true')
    parser.add_argument('--save_path', help='path to which to save output', type=str)
    parser.add_argument('--two_sided', help='whether or not to run a two-sided test', action='store_true')
    args = parser.parse_args()
    main(args)
