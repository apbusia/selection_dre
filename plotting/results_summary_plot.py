import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from matplotlib.lines import Line2D


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


def get_task(model_path):
    if 'classifier' in model_path:
        return 'DRC'
    return 'LER'


def get_encoding_type(model_path):
    if 'pairwise' in model_path:
        return 'Pairwise'
    if 'neighbors' in model_path:
        return 'Neighbors'
    if 'is' in model_path:
        return 'IS'


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
    palette = palette + sns.color_palette('gray', n_colors=1)
    order = order + ['Observed LE']
    return palette, order


def compile_results_dataframe(result_files_dict):
    results_df = None
    for dataset_name, file_list in result_files_dict.items():
        for r_file in file_list:
            cur_results = np.load(r_file, allow_pickle=True).item()
            if 'culled_ndcg' in cur_results['metrics']:
                del cur_results['metrics']['culled_ndcg']
            cur_df = pd.DataFrame(cur_results['metrics'])
            if 'observed_enrichment' in r_file:
                cur_df['model'] = ['Observed LE'] * len(cur_df)
                cur_df['task'] = ['Observed LE'] * len(cur_df)
            else:
                cur_df['model'] = pd.Series(cur_results['meta']['model_paths']).apply(get_model_name)
                cur_df['task'] = pd.Series(cur_results['meta']['model_paths']).apply(get_task)
            cur_df['fracs'] = [cur_results['meta']['culled_fracs']] * len(cur_df)
            cur_df['dataset'] = [dataset_name] * len(cur_df)
            if results_df is None:
                results_df = cur_df
            else:
                results_df = results_df.append(cur_df)
    results_df.reset_index(inplace=True)
    return results_df


def make_results_stripplot(results_df, metric, out_dir, out_tag, include_cnn=False, best_only=False):
    results_df = results_df[['dataset', 'task', 'model', metric]]
    results_df = results_df.groupby(['dataset', 'task', 'model']).mean().reset_index()
    if best_only:
        idx = results_df.groupby(['dataset', 'task'])[metric].transform(max) == results_df[metric]
        results_df = results_df[idx]
    
    if 'train' in out_tag:
        dataset_order = [r'AAV1-9 recombination ($4.6 \times 10^5$ long)',
                         r'AAV1-9 recombination ($4.6 \times 10^4$ long)',
                         r'AAV1-9 recombination ($4.6 \times 10^3$ long)',
                         r'avGFP mutagenesis ($4.6 \times 10^5$ long)',
                         r'100-mer peptide ($4.6 \times 10^7$ short)',
                         r'50-mer peptide ($4.6 \times 10^7$ short)',
                         r'7-mer peptide ($4.6 \times 10^7$ short)']
    else:
        dataset_order = [r'AAV1-9 recombination ($4.6 \times 10^5$ long)',
                         r'noisy AAV1-9 recombination ($4.6 \times 10^5$ long)',
                         r'AAV1-9 recombination ($4.6 \times 10^4$ long)',
                         r'AAV1-9 recombination ($4.6 \times 10^3$ long)',
                         r'AAV1-9 recombination ($4.6 \times 10^7$ short)',
                         r'AAV1-9 recombination ($4.6 \times 10^3$ long $+ 4.6 \times 10^7$ short)',
                         r'avGFP mutagenesis ($4.6 \times 10^5$ long)',
                         r'avGFP mutagenesis ($4.6 \times 10^7$ short)',
                         r'100-mer peptide ($4.6 \times 10^7$ short)',
                         r'50-mer peptide ($4.6 \times 10^7$ short)',
                         r'7-mer peptide ($4.6 \times 10^7$ short)',
                         r'noisy 7-mer peptide ($4.6 \times 10^7$ short)']
    
    f, ax = plt.subplots(1, 1, figsize=(4, 4 * len(dataset_order) / 12))
    palette, hue_order = get_color_palette(include_cnn)
    markers = ['o', 'X', 'd']
    marker_order = ['LER', 'DRC', 'Observed LE']
    for i, t in enumerate(results_df['task'].unique()): #['LER', 'Observed LE', 'DRC']
        legend = True if i == 0 else False
        task_df = results_df[results_df['task'] == t]
        sns.stripplot(data=task_df, y='dataset', x=metric, order=dataset_order,
                      hue='model', hue_order=hue_order, palette=palette,
                      marker=markers[marker_order.index(t)], size=5,
                      dodge=False, edgecolor='none',  ax=ax) #legend=legend,
#     ax.set_xlabel('')
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='right', fontsize=8)
    ax.set_ylabel('')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
    tag = 'MSE'
    if metric == 'pearson_r':
        tag = 'Pearson'
    if metric == 'spearman_r':
        tag = 'Spearman'
#     ax.set_ylabel(tag)
    ax.set_xlabel(tag, fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    handles = handles[:len(hue_order)]
    additional_handles = [
        Line2D([0], [0], marker='o', label='LER', color='k', markersize=5, linewidth=0, linestyle=''),
        Line2D([0], [0], marker='X', label='DRC', color='k', markersize=5, linewidth=0, linestyle=''),
        Line2D([0], [0], marker='d', label='Observed LE', color='k', markersize=5, linewidth=0, linestyle=''),
    ]
    ax.legend(handles=handles + additional_handles, fontsize=10, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)#.get_texts()[-1].set_text('Error Bars')
    plt.savefig(os.path.join(out_dir, '{}_results_summary_plot.png'.format(out_tag)), dpi=300, transparent=False, bbox_inches='tight', facecolor='white')
    plt.close()

def make_results_barplot(results_df, metric, out_dir, out_tag, include_cnn=False):
    results_df = results_df[['dataset', 'task', 'model', metric]]
    results_df = results_df.groupby(['dataset', 'task', 'model']).mean().reset_index()
    idx = results_df.groupby(['dataset', 'task'])[metric].transform(max) == results_df[metric]
    results_df = results_df[idx]
    
    if 'train' in out_tag:
        dataset_order = [r'AAV1-9 recombination ($4.6 \times 10^5$ long)',
                         r'AAV1-9 recombination ($4.6 \times 10^4$ long)',
                         r'AAV1-9 recombination ($4.6 \times 10^3$ long)',
                         r'avGFP mutagenesis ($4.6 \times 10^5$ long)',
                         r'100-mer peptide ($4.6 \times 10^7$ short)',
                         r'50-mer peptide ($4.6 \times 10^7$ short)',
                         r'7-mer peptide ($4.6 \times 10^7$ short)']
    else:
        dataset_order = [r'AAV1-9 recombination ($4.6 \times 10^5$ long)',
                         r'noisy AAV1-9 recombination ($4.6 \times 10^5$ long)',
                         r'AAV1-9 recombination ($4.6 \times 10^4$ long)',
                         r'AAV1-9 recombination ($4.6 \times 10^3$ long)',
                         r'AAV1-9 recombination ($4.6 \times 10^7$ short)',
                         r'AAV1-9 recombination ($4.6 \times 10^3$ long $+ 4.6 \times 10^7$ short)',
                         r'avGFP mutagenesis ($4.6 \times 10^5$ long)',
                         r'avGFP mutagenesis ($4.6 \times 10^7$ short)',
                         r'100-mer peptide ($4.6 \times 10^7$ short)',
                         r'50-mer peptide ($4.6 \times 10^7$ short)',
                         r'7-mer peptide ($4.6 \times 10^7$ short)',
                         r'noisy 7-mer peptide ($4.6 \times 10^7$ short)']
    
    f, ax = plt.subplots(1, 1, figsize=(4, 4 * len(dataset_order) / 12))
    palette = sns.color_palette('tab20', n_colors=3)
    hue_order = ['Observed LE', 'LER', 'DRC']
    sns.barplot(data=results_df, y=metric, x='dataset', hue='task', order=dataset_order, hue_order=hue_order, palette=palette, ax=ax)
    ax.set_ylabel('')
    tag = 'MSE'
    if metric == 'pearson_r':
        tag = 'Pearson'
    if metric == 'spearman_r':
        tag = 'Spearman'
    ax.set_xlabel(tag)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(os.path.join(out_dir, '{}_results_summary_barplot.png'.format(out_tag)), dpi=300, transparent=False, bbox_inches='tight', facecolor='white')
    plt.close()


def main(args):
    set_rc_params()
    results_dir = '../outputs' if args.results_dir is None else args.results_dir
    out_dir = '../outputs' if args.out_dir is None else args.out_dir
    out_tag = args.out_description
    corr_type = args.correlation
    include_cnn = args.include_cnn
    
    train_result_files_dict = {
        r'7-mer peptide ($4.6 \times 10^7$ short)': [
            os.path.join(results_dir, 'observed_enrichment_sim_nnk_7_epistatic_140_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_2x1000_weighted_sim_nnk_7_epistatic_140_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_2x100_weighted_sim_nnk_7_epistatic_140_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_2x200_weighted_sim_nnk_7_epistatic_140_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_2x500_weighted_sim_nnk_7_epistatic_140_db_train_results.npy'),
            os.path.join(results_dir, 'is_linear_weighted_sim_nnk_7_epistatic_140_db_train_results.npy'),
            os.path.join(results_dir, 'neighbors_linear_weighted_sim_nnk_7_epistatic_140_db_train_results.npy'),
            os.path.join(results_dir, 'pairwise_linear_weighted_sim_nnk_7_epistatic_140_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x1000_sim_nnk_7_epistatic_140_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x100_sim_nnk_7_epistatic_140_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x200_sim_nnk_7_epistatic_140_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x500_sim_nnk_7_epistatic_140_db_train_results.npy'),
            os.path.join(results_dir, 'is_linear_classifier_sim_nnk_7_epistatic_140_db_train_results.npy'),
            os.path.join(results_dir, 'neighbors_linear_classifier_sim_nnk_7_epistatic_140_db_train_results.npy'),
            os.path.join(results_dir, 'pairwise_linear_classifier_sim_nnk_7_epistatic_140_db_train_results.npy'),
            
        ],
        r'50-mer peptide ($4.6 \times 10^7$ short)': [
            os.path.join(results_dir, 'observed_enrichment_sim_nnk_50_epistatic_1000_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x1000_sim_nnk_50_epistatic_1000_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x100_sim_nnk_50_epistatic_1000_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x200_sim_nnk_50_epistatic_1000_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x500_sim_nnk_50_epistatic_1000_db_train_results.npy'),
            os.path.join(results_dir, 'is_linear_classifier_sim_nnk_50_epistatic_1000_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_2x1000_weighted_sim_nnk_50_epistatic_1000_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_2x100_weighted_sim_nnk_50_epistatic_1000_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_2x200_weighted_sim_nnk_50_epistatic_1000_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_2x500_weighted_sim_nnk_50_epistatic_1000_db_train_results.npy'),
            os.path.join(results_dir, 'is_linear_weighted_sim_nnk_50_epistatic_1000_db_train_results.npy'),
        ],
        r'100-mer peptide ($4.6 \times 10^7$ short)': [
            os.path.join(results_dir, 'observed_enrichment_sim_nnk_100_epistatic_2000_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_2x1000_weighted_sim_nnk_100_epistatic_2000_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_2x100_weighted_sim_nnk_100_epistatic_2000_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_2x200_weighted_sim_nnk_100_epistatic_2000_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_2x500_weighted_sim_nnk_100_epistatic_2000_db_train_results.npy'),
            os.path.join(results_dir, 'is_linear_weighted_sim_nnk_100_epistatic_2000_db_train_results.npy'),
            os.path.join(results_dir, 'neighbors_linear_weighted_sim_nnk_100_epistatic_2000_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x1000_sim_nnk_100_epistatic_2000_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x100_sim_nnk_100_epistatic_2000_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x200_sim_nnk_100_epistatic_2000_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x500_sim_nnk_100_epistatic_2000_db_train_results.npy'),
            os.path.join(results_dir, 'is_linear_classifier_sim_nnk_100_epistatic_2000_db_train_results.npy'),
            os.path.join(results_dir, 'neighbors_linear_classifier_sim_nnk_100_epistatic_2000_db_train_results.npy'),
        ],
        r'avGFP mutagenesis ($4.6 \times 10^5$ long)': [
            os.path.join(results_dir, 'observed_enrichment_sim_gfp_mut0.1_460K_epistatic_4760_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_2x1000_weighted_sim_gfp_mut0.1_460K_epistatic_4760_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_2x100_weighted_sim_gfp_mut0.1_460K_epistatic_4760_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_2x200_weighted_sim_gfp_mut0.1_460K_epistatic_4760_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_2x500_weighted_sim_gfp_mut0.1_460K_epistatic_4760_db_train_results.npy'),
            os.path.join(results_dir, 'is_cnn_16x5x100_weighted_sim_gfp_mut0.1_460K_epistatic_4760_db_train_results.npy'),
            os.path.join(results_dir, 'is_cnn_2x5x100_weighted_sim_gfp_mut0.1_460K_epistatic_4760_db_train_results.npy'),
            os.path.join(results_dir, 'is_cnn_4x5x100_weighted_sim_gfp_mut0.1_460K_epistatic_4760_db_train_results.npy'),
            os.path.join(results_dir, 'is_cnn_8x5x100_weighted_sim_gfp_mut0.1_460K_epistatic_4760_db_train_results.npy'),
            os.path.join(results_dir, 'is_linear_weighted_sim_gfp_mut0.1_460K_epistatic_4760_db_train_results.npy'),
            os.path.join(results_dir, 'neighbors_linear_weighted_sim_gfp_mut0.1_460K_epistatic_4760_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x1000_sim_gfp_mut0.1_460K_epistatic_4760_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x100_sim_gfp_mut0.1_460K_epistatic_4760_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x200_sim_gfp_mut0.1_460K_epistatic_4760_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x500_sim_gfp_mut0.1_460K_epistatic_4760_db_train_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_16x5x100_sim_gfp_mut0.1_460K_epistatic_4760_db_train_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_2x5x100_sim_gfp_mut0.1_460K_epistatic_4760_db_train_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_4x5x100_sim_gfp_mut0.1_460K_epistatic_4760_db_train_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_8x5x100_sim_gfp_mut0.1_460K_epistatic_4760_db_train_results.npy'),
            os.path.join(results_dir, 'is_linear_classifier_sim_gfp_mut0.1_460K_epistatic_4760_db_train_results.npy'),
            os.path.join(results_dir, 'neighbors_linear_classifier_sim_gfp_mut0.1_460K_epistatic_4760_db_train_results.npy'),
        ],
        r'AAV1-9 recombination ($4.6 \times 10^5$ long)': [
            os.path.join(results_dir, 'observed_enrichment_sim_recomb_7_460K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x1000_sim_recomb_7_460K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x100_sim_recomb_7_460K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x200_sim_recomb_7_460K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x500_sim_recomb_7_460K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_16x5x100_sim_recomb_7_460K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_2x5x100_sim_recomb_7_460K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_4x5x100_sim_recomb_7_460K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_8x5x100_sim_recomb_7_460K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_linear_classifier_sim_recomb_7_460K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'neighbors_linear_classifier_sim_recomb_7_460K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_2x1000_weighted_sim_recomb_7_460K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_2x100_weighted_sim_recomb_7_460K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_2x200_weighted_sim_recomb_7_460K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_2x500_weighted_sim_recomb_7_460K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_cnn_16x5x100_weighted_sim_recomb_7_460K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_cnn_2x5x100_weighted_sim_recomb_7_460K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_cnn_4x5x100_weighted_sim_recomb_7_460K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_cnn_8x5x100_weighted_sim_recomb_7_460K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_linear_weighted_sim_recomb_7_460K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'neighbors_linear_weighted_sim_recomb_7_460K_epistatic_15020_db_train_results.npy'),
        ],
        r'AAV1-9 recombination ($4.6 \times 10^4$ long)': [
            os.path.join(results_dir, 'observed_enrichment_sim_recomb_7_46K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x1000_sim_recomb_7_46K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x100_sim_recomb_7_46K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x200_sim_recomb_7_46K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x500_sim_recomb_7_46K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_16x5x100_sim_recomb_7_46K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_2x5x100_sim_recomb_7_46K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_4x5x100_sim_recomb_7_46K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_8x5x100_sim_recomb_7_46K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_linear_classifier_sim_recomb_7_46K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'neighbors_linear_classifier_sim_recomb_7_46K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_2x1000_weighted_sim_recomb_7_46K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_2x100_weighted_sim_recomb_7_46K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_2x200_weighted_sim_recomb_7_46K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_2x500_weighted_sim_recomb_7_46K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_cnn_16x5x100_weighted_sim_recomb_7_46K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_cnn_2x5x100_weighted_sim_recomb_7_46K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_cnn_4x5x100_weighted_sim_recomb_7_46K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_cnn_8x5x100_weighted_sim_recomb_7_46K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_linear_weighted_sim_recomb_7_46K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'neighbors_linear_weighted_sim_recomb_7_46K_epistatic_15020_db_train_results.npy'),
        ],
        r'AAV1-9 recombination ($4.6 \times 10^3$ long)': [
            os.path.join(results_dir, 'observed_enrichment_sim_recomb_7_4K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x1000_sim_recomb_7_4K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x100_sim_recomb_7_4K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x200_sim_recomb_7_4K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x500_sim_recomb_7_4K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_16x5x100_sim_recomb_7_4K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_2x5x100_sim_recomb_7_4K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_4x5x100_sim_recomb_7_4K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_8x5x100_sim_recomb_7_4K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_linear_classifier_sim_recomb_7_4K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'neighbors_linear_classifier_sim_recomb_7_4K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_2x1000_weighted_sim_recomb_7_4K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_2x100_weighted_sim_recomb_7_4K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_2x200_weighted_sim_recomb_7_4K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_ann_2x500_weighted_sim_recomb_7_4K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_cnn_16x5x100_weighted_sim_recomb_7_4K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_cnn_2x5x100_weighted_sim_recomb_7_4K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_cnn_4x5x100_weighted_sim_recomb_7_4K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_cnn_8x5x100_weighted_sim_recomb_7_4K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'is_linear_weighted_sim_recomb_7_4K_epistatic_15020_db_train_results.npy'),
            os.path.join(results_dir, 'neighbors_linear_weighted_sim_recomb_7_4K_epistatic_15020_db_train_results.npy'),
        ],
    }
    
    val_result_files_dict = {
        r'7-mer peptide ($4.6 \times 10^7$ short)': [
            os.path.join(results_dir, 'is_ann_2x1000_weighted_sim_nnk_7_epistatic_140_db_results.npy'),
            os.path.join(results_dir, 'is_ann_2x100_weighted_sim_nnk_7_epistatic_140_db_results.npy'),
            os.path.join(results_dir, 'is_ann_2x200_weighted_sim_nnk_7_epistatic_140_db_results.npy'),
            os.path.join(results_dir, 'is_ann_2x500_weighted_sim_nnk_7_epistatic_140_db_results.npy'),
            os.path.join(results_dir, 'is_linear_weighted_sim_nnk_7_epistatic_140_db_results.npy'),
            os.path.join(results_dir, 'neighbors_linear_weighted_sim_nnk_7_epistatic_140_db_results.npy'),
            os.path.join(results_dir, 'pairwise_linear_weighted_sim_nnk_7_epistatic_140_db_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x1000_sim_nnk_7_epistatic_140_db_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x100_sim_nnk_7_epistatic_140_db_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x200_sim_nnk_7_epistatic_140_db_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x500_sim_nnk_7_epistatic_140_db_results.npy'),
            os.path.join(results_dir, 'is_linear_classifier_sim_nnk_7_epistatic_140_db_results.npy'),
            os.path.join(results_dir, 'neighbors_linear_classifier_sim_nnk_7_epistatic_140_db_results.npy'),
            os.path.join(results_dir, 'pairwise_linear_classifier_sim_nnk_7_epistatic_140_db_results.npy'),
        ],
        r'noisy 7-mer peptide ($4.6 \times 10^7$ short)': [
            os.path.join(results_dir, 'is_ann_2x1000_weighted_sim_nnk_7_epistatic_140_noisy_db_results.npy'),
            os.path.join(results_dir, 'is_ann_2x100_weighted_sim_nnk_7_epistatic_140_noisy_db_results.npy'),
            os.path.join(results_dir, 'is_ann_2x200_weighted_sim_nnk_7_epistatic_140_noisy_db_results.npy'),
            os.path.join(results_dir, 'is_ann_2x500_weighted_sim_nnk_7_epistatic_140_noisy_db_results.npy'),
            os.path.join(results_dir, 'is_linear_weighted_sim_nnk_7_epistatic_140_noisy_db_results.npy'),
            os.path.join(results_dir, 'neighbors_linear_weighted_sim_nnk_7_epistatic_140_noisy_db_results.npy'),
            os.path.join(results_dir, 'pairwise_linear_weighted_sim_nnk_7_epistatic_140_noisy_db_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x1000_sim_nnk_7_epistatic_140_noisy_db_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x100_sim_nnk_7_epistatic_140_noisy_db_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x200_sim_nnk_7_epistatic_140_noisy_db_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x500_sim_nnk_7_epistatic_140_noisy_db_results.npy'),
            os.path.join(results_dir, 'is_linear_classifier_sim_nnk_7_epistatic_140_noisy_db_results.npy'),
            os.path.join(results_dir, 'neighbors_linear_classifier_sim_nnk_7_epistatic_140_noisy_db_results.npy'),
            os.path.join(results_dir, 'pairwise_linear_classifier_sim_nnk_7_epistatic_140_noisy_db_results.npy'),
        ],
        r'50-mer peptide ($4.6 \times 10^7$ short)': [
            os.path.join(results_dir, 'is_ann_classifier_2x1000_sim_nnk_50_epistatic_1000_db_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x100_sim_nnk_50_epistatic_1000_db_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x200_sim_nnk_50_epistatic_1000_db_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x500_sim_nnk_50_epistatic_1000_db_results.npy'),
            os.path.join(results_dir, 'is_linear_classifier_sim_nnk_50_epistatic_1000_db_results.npy'),
            os.path.join(results_dir, 'neighbors_linear_classifier_sim_nnk_50_epistatic_1000_db_results.npy'),
            os.path.join(results_dir, 'pairwise_linear_classifier_sim_nnk_50_epistatic_1000_db_results.npy'),
            os.path.join(results_dir, 'is_ann_2x1000_weighted_sim_nnk_50_epistatic_1000_db_results.npy'),
            os.path.join(results_dir, 'is_ann_2x100_weighted_sim_nnk_50_epistatic_1000_db_results.npy'),
            os.path.join(results_dir, 'is_ann_2x200_weighted_sim_nnk_50_epistatic_1000_db_results.npy'),
            os.path.join(results_dir, 'is_ann_2x500_weighted_sim_nnk_50_epistatic_1000_db_results.npy'),
            os.path.join(results_dir, 'is_linear_weighted_sim_nnk_50_epistatic_1000_db_results.npy'),
            os.path.join(results_dir, 'neighbors_linear_weighted_sim_nnk_50_epistatic_1000_db_results.npy'),
            os.path.join(results_dir, 'pairwise_linear_weighted_sim_nnk_50_epistatic_1000_db_results.npy'),
        ],
        r'100-mer peptide ($4.6 \times 10^7$ short)': [
            os.path.join(results_dir, 'is_ann_2x1000_weighted_sim_nnk_100_epistatic_2000_db_results.npy'),
            os.path.join(results_dir, 'is_ann_2x100_weighted_sim_nnk_100_epistatic_2000_db_results.npy'),
            os.path.join(results_dir, 'is_ann_2x200_weighted_sim_nnk_100_epistatic_2000_db_results.npy'),
            os.path.join(results_dir, 'is_ann_2x500_weighted_sim_nnk_100_epistatic_2000_db_results.npy'),
            os.path.join(results_dir, 'is_linear_weighted_sim_nnk_100_epistatic_2000_db_results.npy'),
            os.path.join(results_dir, 'neighbors_linear_weighted_sim_nnk_100_epistatic_2000_db_results.npy'),
            os.path.join(results_dir, 'pairwise_linear_weighted_sim_nnk_100_epistatic_2000_db_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x1000_sim_nnk_100_epistatic_2000_db_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x100_sim_nnk_100_epistatic_2000_db_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x200_sim_nnk_100_epistatic_2000_db_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x500_sim_nnk_100_epistatic_2000_db_results.npy'),
            os.path.join(results_dir, 'is_linear_classifier_sim_nnk_100_epistatic_2000_db_results.npy'),
            os.path.join(results_dir, 'neighbors_linear_classifier_sim_nnk_100_epistatic_2000_db_results.npy'),
            os.path.join(results_dir, 'pairwise_linear_classifier_sim_nnk_100_epistatic_2000_db_results.npy'),
        ],
        r'avGFP mutagenesis ($4.6 \times 10^5$ long)': [
            os.path.join(results_dir, 'is_ann_2x1000_weighted_sim_gfp_mut0.1_460K_epistatic_4760_db_results.npy'),
            os.path.join(results_dir, 'is_ann_2x100_weighted_sim_gfp_mut0.1_460K_epistatic_4760_db_results.npy'),
            os.path.join(results_dir, 'is_ann_2x200_weighted_sim_gfp_mut0.1_460K_epistatic_4760_db_results.npy'),
            os.path.join(results_dir, 'is_ann_2x500_weighted_sim_gfp_mut0.1_460K_epistatic_4760_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_16x5x100_weighted_sim_gfp_mut0.1_460K_epistatic_4760_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_2x5x100_weighted_sim_gfp_mut0.1_460K_epistatic_4760_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_4x5x100_weighted_sim_gfp_mut0.1_460K_epistatic_4760_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_8x5x100_weighted_sim_gfp_mut0.1_460K_epistatic_4760_db_results.npy'),
            os.path.join(results_dir, 'is_linear_weighted_sim_gfp_mut0.1_460K_epistatic_4760_db_results.npy'),
            os.path.join(results_dir, 'neighbors_linear_weighted_sim_gfp_mut0.1_460K_epistatic_4760_db_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x1000_sim_gfp_mut0.1_460K_epistatic_4760_db_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x100_sim_gfp_mut0.1_460K_epistatic_4760_db_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x200_sim_gfp_mut0.1_460K_epistatic_4760_db_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x500_sim_gfp_mut0.1_460K_epistatic_4760_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_16x5x100_sim_gfp_mut0.1_460K_epistatic_4760_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_2x5x100_sim_gfp_mut0.1_460K_epistatic_4760_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_4x5x100_sim_gfp_mut0.1_460K_epistatic_4760_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_8x5x100_sim_gfp_mut0.1_460K_epistatic_4760_db_results.npy'),
            os.path.join(results_dir, 'is_linear_classifier_sim_gfp_mut0.1_460K_epistatic_4760_db_results.npy'),
            os.path.join(results_dir, 'neighbors_linear_classifier_sim_gfp_mut0.1_460K_epistatic_4760_db_results.npy'),
        ],
        r'avGFP mutagenesis ($4.6 \times 10^7$ short)': [
            os.path.join(results_dir, 'is_cnn_16x5x100_weighted_sim_gfp_mut0.1_epistatic_4760_short_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_2x5x100_weighted_sim_gfp_mut0.1_epistatic_4760_short_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_4x5x100_weighted_sim_gfp_mut0.1_epistatic_4760_short_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_8x5x100_weighted_sim_gfp_mut0.1_epistatic_4760_short_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_16x5x100_sim_gfp_mut0.1_epistatic_4760_short_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_2x5x100_sim_gfp_mut0.1_epistatic_4760_short_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_4x5x100_sim_gfp_mut0.1_epistatic_4760_short_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_8x5x100_sim_gfp_mut0.1_epistatic_4760_short_db_results.npy'),
        ],
        r'AAV1-9 recombination ($4.6 \times 10^5$ long)': [
            os.path.join(results_dir, 'is_ann_2x1000_weighted_sim_recomb_7_460K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_ann_2x100_weighted_sim_recomb_7_460K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_ann_2x200_weighted_sim_recomb_7_460K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_ann_2x500_weighted_sim_recomb_7_460K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_16x5x100_weighted_sim_recomb_7_460K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_2x5x100_weighted_sim_recomb_7_460K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_4x5x100_weighted_sim_recomb_7_460K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_8x5x100_weighted_sim_recomb_7_460K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_linear_weighted_sim_recomb_7_460K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'neighbors_linear_weighted_sim_recomb_7_460K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x1000_sim_recomb_7_460K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x100_sim_recomb_7_460K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x200_sim_recomb_7_460K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x500_sim_recomb_7_460K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_16x5x100_sim_recomb_7_460K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_2x5x100_sim_recomb_7_460K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_4x5x100_sim_recomb_7_460K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_8x5x100_sim_recomb_7_460K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_linear_classifier_sim_recomb_7_460K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'neighbors_linear_classifier_sim_recomb_7_460K_epistatic_15020_db_results.npy'),
        ],
        r'noisy AAV1-9 recombination ($4.6 \times 10^5$ long)': [
            os.path.join(results_dir, 'is_ann_2x1000_weighted_sim_recomb_7_epistatic_15020_noisy_db_results.npy'),
            os.path.join(results_dir, 'is_ann_2x100_weighted_sim_recomb_7_epistatic_15020_noisy_db_results.npy'),
            os.path.join(results_dir, 'is_ann_2x200_weighted_sim_recomb_7_epistatic_15020_noisy_db_results.npy'),
            os.path.join(results_dir, 'is_ann_2x500_weighted_sim_recomb_7_epistatic_15020_noisy_db_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x1000_sim_recomb_7_epistatic_15020_noisy_db_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x100_sim_recomb_7_epistatic_15020_noisy_db_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x200_sim_recomb_7_epistatic_15020_noisy_db_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x500_sim_recomb_7_epistatic_15020_noisy_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_16x5x100_weighted_sim_recomb_7_epistatic_15020_noisy_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_2x5x100_weighted_sim_recomb_7_epistatic_15020_noisy_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_4x5x100_weighted_sim_recomb_7_epistatic_15020_noisy_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_8x5x100_weighted_sim_recomb_7_epistatic_15020_noisy_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_16x5x100_sim_recomb_7_epistatic_15020_noisy_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_2x5x100_sim_recomb_7_epistatic_15020_noisy_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_4x5x100_sim_recomb_7_epistatic_15020_noisy_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_8x5x100_sim_recomb_7_epistatic_15020_noisy_db_results.npy'),
            os.path.join(results_dir, 'is_linear_classifier_sim_recomb_7_epistatic_15020_noisy_db_results.npy'),
            os.path.join(results_dir, 'is_linear_weighted_sim_recomb_7_epistatic_15020_noisy_db_results.npy'),
            os.path.join(results_dir, 'neighbors_linear_classifier_sim_recomb_7_epistatic_15020_noisy_db_results.npy'),
            os.path.join(results_dir, 'neighbors_linear_weighted_sim_recomb_7_epistatic_15020_noisy_db_results.npy'),
        ],
        r'AAV1-9 recombination ($4.6 \times 10^4$ long)': [
            os.path.join(results_dir, 'is_ann_classifier_2x1000_sim_recomb_7_46K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x100_sim_recomb_7_46K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x200_sim_recomb_7_46K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x500_sim_recomb_7_46K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_16x5x100_sim_recomb_7_46K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_2x5x100_sim_recomb_7_46K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_4x5x100_sim_recomb_7_46K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_8x5x100_sim_recomb_7_46K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_linear_classifier_sim_recomb_7_46K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'neighbors_linear_classifier_sim_recomb_7_46K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_ann_2x1000_weighted_sim_recomb_7_46K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_ann_2x100_weighted_sim_recomb_7_46K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_ann_2x200_weighted_sim_recomb_7_46K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_ann_2x500_weighted_sim_recomb_7_46K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_16x5x100_weighted_sim_recomb_7_46K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_2x5x100_weighted_sim_recomb_7_46K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_4x5x100_weighted_sim_recomb_7_46K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_8x5x100_weighted_sim_recomb_7_46K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_linear_weighted_sim_recomb_7_46K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'neighbors_linear_weighted_sim_recomb_7_46K_epistatic_15020_db_results.npy'),
        ],
        r'AAV1-9 recombination ($4.6 \times 10^3$ long)': [
            os.path.join(results_dir, 'is_ann_classifier_2x1000_sim_recomb_7_4K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x100_sim_recomb_7_4K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x200_sim_recomb_7_4K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_ann_classifier_2x500_sim_recomb_7_4K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_16x5x100_sim_recomb_7_4K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_2x5x100_sim_recomb_7_4K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_4x5x100_sim_recomb_7_4K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_8x5x100_sim_recomb_7_4K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_linear_classifier_sim_recomb_7_4K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'neighbors_linear_classifier_sim_recomb_7_4K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_ann_2x1000_weighted_sim_recomb_7_4K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_ann_2x100_weighted_sim_recomb_7_4K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_ann_2x200_weighted_sim_recomb_7_4K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_ann_2x500_weighted_sim_recomb_7_4K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_16x5x100_weighted_sim_recomb_7_4K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_2x5x100_weighted_sim_recomb_7_4K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_4x5x100_weighted_sim_recomb_7_4K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_8x5x100_weighted_sim_recomb_7_4K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'is_linear_weighted_sim_recomb_7_4K_epistatic_15020_db_results.npy'),
            os.path.join(results_dir, 'neighbors_linear_weighted_sim_recomb_7_4K_epistatic_15020_db_results.npy'),
        ],
        r'AAV1-9 recombination ($4.6 \times 10^7$ short)': [
            os.path.join(results_dir, 'is_cnn_classifier_16x5x100_sim_recomb_7_epistatic_15020_short_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_2x5x100_sim_recomb_7_epistatic_15020_short_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_4x5x100_sim_recomb_7_epistatic_15020_short_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_8x5x100_sim_recomb_7_epistatic_15020_short_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_16x5x100_weighted_sim_recomb_7_epistatic_15020_short_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_2x5x100_weighted_sim_recomb_7_epistatic_15020_short_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_4x5x100_weighted_sim_recomb_7_epistatic_15020_short_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_8x5x100_weighted_sim_recomb_7_epistatic_15020_short_db_results.npy'),
        ],
        r'AAV1-9 recombination ($4.6 \times 10^3$ long $+ 4.6 \times 10^7$ short)': [
            os.path.join(results_dir, 'is_cnn_classifier_16x5x100_sim_recomb_7_epistatic_15020_long_4K_short_46M_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_2x5x100_sim_recomb_7_epistatic_15020_long_4K_short_46M_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_4x5x100_sim_recomb_7_epistatic_15020_long_4K_short_46M_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_classifier_8x5x100_sim_recomb_7_epistatic_15020_long_4K_short_46M_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_16x5x100_weighted_sim_recomb_7_epistatic_15020_long_4K_short_46M_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_2x5x100_weighted_sim_recomb_7_epistatic_15020_long_4K_short_46M_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_4x5x100_weighted_sim_recomb_7_epistatic_15020_long_4K_short_46M_db_results.npy'),
            os.path.join(results_dir, 'is_cnn_8x5x100_weighted_sim_recomb_7_epistatic_15020_long_4K_short_46M_db_results.npy'),
        ],
    }
    
    results_df = compile_results_dataframe(train_result_files_dict)
    make_results_stripplot(results_df, args.correlation,
                           args.out_dir, 'train' + args.out_description,
                           include_cnn=args.include_cnn, best_only=args.best_only)
    make_results_barplot(results_df, args.correlation,
                         args.out_dir, 'train' + args.out_description,
                         include_cnn=args.include_cnn)
    
    results_df = compile_results_dataframe(val_result_files_dict)
    make_results_stripplot(results_df, args.correlation,
                           args.out_dir, 'val' + args.out_description,
                           include_cnn=args.include_cnn, best_only=args.best_only)
    make_results_barplot(results_df, args.correlation,
                         args.out_dir, 'val' + args.out_description,
                         include_cnn=args.include_cnn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', help='directory from which to load saved results files', type=str)
    parser.add_argument('--out_dir', help='directory to which to save the generated plots', type=str)
    parser.add_argument('--out_description', default='', help='descriptive tag to add to output filenames', type=str)
    parser.add_argument('--correlation', default='spearman_r', help='type of correlation to plot', type=str)
    parser.add_argument('--include_cnn', help='include convolutional architectures in plots', action='store_true')
    parser.add_argument('--best_only', help='plot only best-performing instance for each method', action='store_true')
    args = parser.parse_args()
    main(args)