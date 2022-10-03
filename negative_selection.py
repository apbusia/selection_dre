import os
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from matplotlib import rcParams
from tensorflow import keras
from scipy.special import log_softmax
import data_prep
import modeling
from run_models import disable_gpu
from evaluate_models import get_encoding_type, get_model_type, get_model_predictions, logits_to_log_density_ratio


def set_rc_params():
    plt.style.use('seaborn-deep')
    rcParams['figure.dpi'] = 200
    rcParams['savefig.dpi'] = 300
    rcParams['lines.linewidth'] = 1.0
    rcParams['axes.grid'] = True
    rcParams['axes.spines.right'] = False
    rcParams['axes.spines.top'] = False
    rcParams['grid.color'] = 'gray'
    rcParams['grid.alpha'] = 0.2
    rcParams['axes.linewidth'] = 0.5
    rcParams['mathtext.fontset'] = 'cm'
    rcParams['font.family'] = 'STIXGeneral'
    return


def get_model_name(model_path):
    if 'linear' in model_path:
        return 'Linear, {}'.format(get_encoding_type(model_path))
    if 'ann' in model_path:
        n_units = re.search('(?<=ann_)(classifier_)?(\d+)x(\d+)', model_path).group(3)
        if n_units == '1000':
            n_units = '999'
        return 'NN, {}'.format(n_units)
    if 'cnn' in model_path:
        match = re.search('(?<=cnn_)(classifier_)?(\d+)x(\d+)x(\d+)', model_path)
        n_layers = match.group(2)
        win_size = match.group(3)
        n_units = match.group(4)
        return 'CNN, {}x{}x{}'.format(n_layers, win_size, n_units)


def get_method(model_path):
    if 'classifier' in model_path:
        return 'DRC'
    return 'LER'


def get_color_palette(include_cnn=False):
    linear_palette = sns.color_palette('crest', n_colors=3)
    ann_palette = sns.color_palette('flare', n_colors=4)
    palette = linear_palette + ann_palette
    order = [
        'Linear, IS', 'Linear, Neighbors', 'Linear, Pairwise', 'NN, 100', 'NN, 200', 'NN, 500', 'NN, 999']
    if include_cnn:
        cnn_palette = sns.color_palette('pink_r', n_colors=4)
        palette = palette + cnn_palette
        order = order + ['CNN, 2x5x100', 'CNN, 4x5x100', 'CNN, 8x5x100', 'CNN, 16x5x100']
    return palette, order


def make_negative_selection_plot(results_df, truth_cols, save_file):
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    palette, order = get_color_palette(True)
    sns.scatterplot(data=results_df, x=truth_cols[0], y=truth_cols[1], hue='model', style='method', palette=palette, hue_order=order, alpha=0.8, s=10, ax=ax, linewidth=0, edgecolor='none')
    ax.set_xlabel('Groundtruth Fitness 1')
    ax.set_ylabel('Groundtruth Fitness 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(save_file, dpi=300, transparent=False, bbox_inches='tight', facecolor='white')
    plt.close()
    return


def main(args):
    save_path = args.save_path
    data_path, model_paths = args.data_path, args.model_paths
    print('\nLoading data from {}'.format(data_path))
    df = pd.read_csv(data_path)
    truth = args.truth_columns if args.truth_columns is not None else [
        'true_enrichment_{}'.format(i) for i in [args.pos_output_idx, args.neg_output_idx]]
    print('Using groundtruth in:', truth)
    idx_paths = args.idx_paths
    
    results, plot_df = None, None
    for i, model_path in enumerate(model_paths):
        disable_gpu([0, 1])
        keras.backend.clear_session()
        
        print('\nUsing model {}'.format(model_path))
        model_type = get_model_type(model_path)
        cur_df = df.copy()
        cur_df['fold'] = [i] * len(cur_df)
        cur_df['model'] = [get_model_name(model_path)] * len(cur_df)
        cur_df['method'] = [get_method(model_path)] * len(cur_df)
        if idx_paths is not None:
            idx_path = idx_paths[i]
            print('\n\tUsing indices {}'.format(idx_path))
            test_idx = np.load(idx_path)
            cur_df = cur_df.iloc[test_idx]
            seqs = df['seq'].iloc[test_idx].reset_index(drop=True)
        else:
            seqs = df['seq']
        
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
            # Add 1 to output indices since pre is always index 0 in classification models.
            if args.run_mcnemar:
                cur_df['pos_pred'] = logits_to_log_density_ratio(preds, post_idx=args.pos_output_idx + 1, pre_idx=0)
                cur_df['neg_pred'] = logits_to_log_density_ratio(preds, post_idx=args.neg_output_idx + 1, pre_idx=0)
            preds = logits_to_log_density_ratio(preds, post_idx=args.pos_output_idx + 1, pre_idx=args.neg_output_idx + 1)
        else:
            if args.run_mcnemar:
                cur_df['pos_pred'] = preds[args.pos_output_idx].flatten()
                cur_df['neg_pred'] = preds[args.neg_output_idx].flatten()
            preds = preds[args.pos_output_idx].flatten() - preds[args.neg_output_idx].flatten()
        cur_df['pred'] = preds
        cur_df = cur_df.sort_values('pred', ascending=False)
        if results is None:
            results = cur_df
        else:
            results = results.append(cur_df)
        
        print('\n\tDetermining top selective variants...')
        if plot_df is None:
            plot_df = cur_df.head(n=args.n_top)
        else:
            plot_df = plot_df.append(cur_df.head(n=args.n_top))
    results.reset_index(inplace=True, drop=True)
    plot_df.reset_index(inplace=True, drop=True)

    if save_path is not None:
        print('\nSaving results to {}'.format(save_path))
        results.to_csv(save_path)
        print('\nPlotting selected variants...')
        set_rc_params()
        plot_savefile = os.path.splitext(save_path)[0] + '_plot.png'
        make_negative_selection_plot(plot_df, truth, plot_savefile)
    
    if args.run_mcnemar:
        pos_threshold = np.quantile(results[truth[0]].values, 0.75)
        neg_threshold = np.quantile(results[truth[1]].values, 0.25)
        print('\nRunning McNemar\'s test on model predictions...')
        method_preds = []
        for method in ['DRC', 'LER']:
#             cur_results = results.loc[(results['fold'] == i) & (results['method'] == method)]
            cur_results = results.loc[results['method'] == method]
            correct = (
                1 * np.logical_and(cur_results[truth[0]].values > pos_threshold, cur_results[truth[1]].values < neg_threshold) == 
                1 * np.logical_and(cur_results['pos_pred'].values > pos_threshold, cur_results['neg_pred'].values < neg_threshold))
            print('\t{} Accuracy = {:.3f}'.format(method, np.mean(correct)))
            method_preds.append(correct)
        _, contingency_table = stats.contingency.crosstab(method_preds[0], method_preds[1])
        mcnemar = (abs(contingency_table[1, 0] - contingency_table[0, 1]) - 1)**2 / (
            contingency_table[1, 0] + contingency_table[0, 1])
        p_value = 1 - stats.chi2.cdf(mcnemar, 1)
        print('\tMcNemar\'s test statistic = {:.3f}, p-value = {:.3f}'.format(mcnemar, p_value))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='path to counts dataset', type=str)
    parser.add_argument('model_paths', help='paths to trained Keras predictive models', nargs='+', type=str)
    parser.add_argument('--pos_output_idx', default=0, help='index of model output for which to positively select', type=int)
    parser.add_argument('--neg_output_idx', default=1, help='index of model output for which to negatively select', type=int)
    parser.add_argument('--truth_columns', help='column names of groundtruth in counts dataset', type=str, nargs='+')
    parser.add_argument('--idx_paths', help='paths to files containing subset of test indices', nargs='+', type=str)
    parser.add_argument('--n_top', default=10, help='number of top selected variants to save/plot', type=int)
    parser.add_argument('--run_mcnemar', help='whether to run McNemars test', action='store_true')
    parser.add_argument('--save_path', help='path to which to save output', type=str)
    args = parser.parse_args()
    main(args)
