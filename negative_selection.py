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
from matplotlib.lines import Line2D
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
        'Linear, IS', 'Linear, Neighbors', 'Linear, Pairwise', 'NN, 100', 'NN, 200', 'NN, 500', 'NN, 1000']
    if include_cnn:
        cnn_palette = sns.color_palette('pink_r', n_colors=4)
        palette = palette + cnn_palette
        order = order + ['CNN, 2x5x100', 'CNN, 4x5x100', 'CNN, 8x5x100', 'CNN, 16x5x100']
    return palette, order


def make_negative_selection_plot(results_df, truth_cols, save_file):
    palette, order = get_color_palette(True)    
    
    # Scatterplot of ground truth fitnesses
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    style_order = ['LER', 'DRC']
    palette = sns.color_palette('colorblind', n_colors=3)[:2]
    sns.scatterplot(data=results_df, x=truth_cols[0], y=truth_cols[1], hue='method', style='method', palette=palette, hue_order=style_order, style_order=style_order, alpha=0.75, s=10, ax=ax, linewidth=0, edgecolor='none', legend=False)
    ax.set_aspect('equal', adjustable='datalim')
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    mean_df = results_df[['model', 'method', truth_cols[0], truth_cols[1]]].groupby(['model', 'method']).mean().reset_index()
    mean_df['method'] = mean_df['method'] + ' Average'
    style_order = [s + ' Average' for s in style_order]
    sns.scatterplot(data=mean_df, x=truth_cols[0], y=truth_cols[1], facecolor='none', style='method', style_order=style_order, edgecolor='k', s=10, ax=ax, linewidth=0.5, legend=False)
    corner_coord = (results_df[truth_cols[0]].max(), results_df[truth_cols[1]].min())
    ax.scatter(corner_coord[0], corner_coord[1], s=10, marker='*', edgecolor='k', facecolor='none', linewidth=0.5)
    drc_coord = np.concatenate([mean_df[truth_cols[0]].values[mean_df['method'] == 'DRC Average'], mean_df[truth_cols[1]].values[mean_df['method'] == 'DRC Average']])
    radius = np.linalg.norm(np.array(drc_coord) - np.array(corner_coord))
    circle = plt.Circle(corner_coord, radius=radius, color='k', fill=False, linewidth=0.5, linestyle='dashed')
    plt.plot([drc_coord[0], corner_coord[0]], [drc_coord[1], corner_coord[1]], color=palette[style_order.index('DRC Average')], linestyle='dashed', linewidth=0.5)
    ler_coord = np.concatenate([mean_df[truth_cols[0]].values[mean_df['method'] == 'LER Average'], mean_df[truth_cols[1]].values[mean_df['method'] == 'LER Average']])
    plt.plot([ler_coord[0], corner_coord[0]], [ler_coord[1], corner_coord[1]], color=palette[style_order.index('LER Average')], linestyle='dashed', linewidth=0.5)
    ax.add_patch(circle)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r'Positive fitness $\quad \longrightarrow$')
    ax.set_ylabel(r"$\longleftarrow \quad$ Negative fitness")
    legend_elements = [
        Line2D([0], [0], marker='o', label='wLER', color=palette[style_order.index('LER Average')], markersize=5, linewidth=0, linestyle=''),
        Line2D([0], [0], marker='o', label='wLER average', markerfacecolor='none', markeredgecolor='k', markersize=5, linewidth=0, linestyle=''),
        Line2D([0], [0], marker='X', label='MBE', color=palette[style_order.index('DRC Average')], markersize=5, linewidth=0, linestyle=''),
        Line2D([0], [0], marker='X', label='MBE average', markerfacecolor='none', markeredgecolor='k', markersize=5, linewidth=0, linestyle=''),
        Line2D([0], [0], marker='*', label='Theoretical ideal', markerfacecolor='none', markeredgecolor='k', markersize=5, linewidth=0, linestyle=''),
    ]
    ax.legend(handles=legend_elements, fontsize=10, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(save_file, dpi=300, transparent=False, bbox_inches='tight', facecolor='white')
    plt.close()
    return


def make_objective_paired_plot(results_df, truth_cols, save_file):
    palette, order = get_color_palette(True) 
    
    # Paired plot of negative selection objective
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    results_df['truth'] = results_df[truth_cols[0]] - results_df[truth_cols[1]]
    style_order = ['LER', 'DRC']
    sns.scatterplot(data=results_df, x='pred', y='truth', hue='model', style='method', palette=palette, hue_order=order, style_order=style_order, alpha=0.75, s=10, ax=ax, linewidth=0, edgecolor='none', legend=False)
    mean_df = results_df[['model', 'method', 'pred', 'truth']].groupby(['model', 'method']).mean().reset_index()
    sns.scatterplot(data=mean_df, x='pred', y='truth', style='method', facecolor='none', edgecolor='k', style_order=style_order, s=10, ax=ax, linewidth=0.5, legend=False) #edgecolor=palette[order.index(mean_df['model'].iloc[0])]
    for v in mean_df['truth'].values:
        ax.axhline(v, linestyle='dashed', color='k', linewidth=0.5)
    ax.set_ylabel(r'Ground truth $\Delta \quad \longrightarrow$')
    ax.set_xlabel(r'Predicted $\Delta$')
    plt.savefig(save_file, dpi=300, transparent=False, bbox_inches='tight', facecolor='white')
    plt.close()
    return


def make_objective_boxplot(results_df, truth_cols, save_file):
    palette, order = get_color_palette(True) 
    fig, ax = plt.subplots(1, 1, figsize=(1.5, 2))
    results_df['truth'] = results_df[truth_cols[0]] - results_df[truth_cols[1]]
    style_dict = {'LER': 'o', 'DRC': 'x'}
    sns.boxplot(data=results_df, x='method', y='truth', hue='model', palette=palette, hue_order=order, dodge=False, width=0.4, linewidth=0.5, showcaps=False, ax=ax)
    ax.set_ylabel(r'Ground truth $\Delta \quad \longrightarrow$')
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
    print('Using ground truth in:', truth)
    df = df[['seq'] + truth]
    idx_paths = args.idx_paths
    
    results, plot_df = None, None
    for i, model_path in enumerate(model_paths):
        disable_gpu([0, 1, 2, 3])
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
#         print('\nSaving results to {}'.format(save_path))
#         results.to_csv(save_path)
        print('\nPlotting selected variants...')
        set_rc_params()
        plot_savefile = os.path.splitext(save_path)[0] + '_plot.png'
        make_negative_selection_plot(plot_df, truth, plot_savefile)
        plot_savefile = os.path.splitext(save_path)[0] + '_plot_paired.png'
        make_objective_paired_plot(plot_df, truth, plot_savefile)
        plot_savefile = os.path.splitext(save_path)[0] + '_boxplot.png'
        make_objective_boxplot(plot_df, truth, plot_savefile)
    
    if args.run_mcnemar:
        q = 0.99
        print('\nRunning McNemar\'s test on model predictions...')
        correct = {'DRC': [], 'LER': []}
        for i in results['fold'].unique():
            cur_results = results.loc[results['fold'] == i]
            method = cur_results['method'].iloc[0]
            t = cur_results[truth[args.pos_output_idx]].values - cur_results[truth[args.neg_output_idx]].values
            t = t > np.quantile(t, q)
            p = cur_results['pred'].values
            p = p > np.quantile(p, q)
            correct[method].append(1 * t == 1 * p)
        for method in ['DRC', 'LER']:
            print('\t{} Accuracy = {:.5f}'.format(method, np.mean(np.concatenate(correct[method]))))
        _, contingency_table = stats.contingency.crosstab(np.concatenate(correct['DRC']), np.concatenate(correct['LER']))
        mcnemar = (abs(contingency_table[1, 0] - contingency_table[0, 1]) - 1)**2 / (
            contingency_table[1, 0] + contingency_table[0, 1])
        p_value = 1 - stats.chi2.cdf(mcnemar, 1)
        print('\tMcNemar\'s test statistic = {:.3f}, p-value = {:.3f}'.format(mcnemar, p_value))
        if save_path is not None:
            mcnemar_savefile = os.path.splitext(save_path)[0] + '_mcnemar.npz'
            np.savez(mcnemar_savefile, contingency_table=contingency_table, test_statistic=mcnemar, p_value=p_value)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='path to counts dataset', type=str)
    parser.add_argument('model_paths', help='paths to trained Keras predictive models', nargs='+', type=str)
    parser.add_argument('--pos_output_idx', default=0, help='index of model output for which to positively select', type=int)
    parser.add_argument('--neg_output_idx', default=1, help='index of model output for which to negatively select', type=int)
    parser.add_argument('--truth_columns', help='column names of ground truth in counts dataset', type=str, nargs='+')
    parser.add_argument('--idx_paths', help='paths to files containing subset of test indices', nargs='+', type=str)
    parser.add_argument('--n_top', default=10, help='number of top selected variants to save/plot', type=int)
    parser.add_argument('--run_mcnemar', help='whether to run McNemars test', action='store_true')
    parser.add_argument('--save_path', help='path to which to save output', type=str)
    args = parser.parse_args()
    main(args)
