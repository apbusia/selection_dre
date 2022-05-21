import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams


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


def get_task(model_path):
    if 'classifier' in model_path:
        return 'classification'
    return 'regression'


def get_encoding_type(model_path):
    if 'pairwise' in model_path:
        return 'Pairwise'
    if 'neighbors' in model_path:
        return 'Neighbors'
    if 'is' in model_path:
        return 'IS'


def get_color_palette():
    linear_palette = sns.color_palette('crest', n_colors=3)
    ann_palette = sns.color_palette('flare', n_colors=4)
    order = [
        'Linear, IS', 'Linear, Neighbors', 'Linear, Pairwise', 'NN, 100', 'NN, 200', 'NN, 500', 'NN, 1000']
    return linear_palette + ann_palette, order


def make_culled_correlation_plot(results_df, out_dir, out_tag, corr_type):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    palette, order = get_color_palette()
    reg_df, class_df = None, None
    for _, row in results_df.iterrows():
        cur_df = {'fracs': row['fracs'],  # np.arange(0, 1, 0.01),
                  'culled_{}'.format(corr_type): row['culled_{}'.format(corr_type)],
                  'model': [row['model']] * len(row['culled_{}'.format(corr_type)])}
        cur_df = pd.DataFrame(cur_df)
        if row['task'] == 'regression':
            if reg_df is None:
                reg_df = cur_df
            else:
                reg_df = reg_df.append(cur_df)
        elif row['task'] == 'classification':
            if class_df is None:
                class_df = cur_df
            else:
                class_df = class_df.append(cur_df)
    reg_df.reset_index(inplace=True)
    class_df.reset_index(inplace=True)
    sns.lineplot(data=reg_df, x='fracs', y='culled_{}'.format(corr_type), hue='model', palette=palette, hue_order=order, err_style='band', ax=axes[0], legend=False)
    sns.lineplot(data=class_df, x='fracs', y='culled_{}'.format(corr_type), hue='model', palette=palette, hue_order=order, err_style='band', ax=axes[1])
    axes[0].set_title('Regression Models')
    axes[1].set_title('Classification Models')
    for ax in axes:
        ax.set_xlabel('Fraction of top test sequences')
        ax.set_ylabel('{} correlation'.format(corr_type.capitalize()))
        fracs2 = [f for i, f in enumerate(np.arange(0, 1, 0.01)) if i % 10 == 0]
        ax.set_xticks(fracs2)
        ax.set_xticklabels(["%.1f" % (1-f) for f in fracs2])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(os.path.join(out_dir, '{}_culled_correlation_plot.png'.format(out_tag)), dpi=300, transparent=False, bbox_inches='tight', facecolor='white')
    plt.close()


def make_correlation_barplots(results_df, out_dir, out_tag):
    f, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    palette, order = get_color_palette()
    sns.barplot(data=results_df, x='task', y='pearson_r', hue='model', palette=palette, hue_order=order, ax=axes[0])
    axes[0].legend_.remove()
    sns.barplot(data=results_df, x='task', y='spearman_r', hue='model', palette=palette, hue_order=order, ax=axes[1])
    axes[0].set_title('Pearson Correlation')
    axes[1].set_title('Spearman Correlation')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(os.path.join(out_dir, '{}_model_comparison_barplot.png'.format(out_tag)), dpi=300, transparent=False, bbox_inches='tight', facecolor='white')
    plt.close()


def make_culled_correlation_paired_plot(results_df, out_dir, out_tag, corr_type):
    fig, ax = plt.subplots(1, 1, figsize=(4, 2))
    palette, order = get_color_palette()
    reg_df, class_df = None, None
    for _, row in results_df.iterrows():
        cur_df = {'fracs': row['fracs'],  # np.arange(0, 1, 0.01),
                  '{}_culled_{}'.format(row['task'], corr_type): row['culled_{}'.format(corr_type)],
                  'model': [row['model']] * len(row['culled_{}'.format(corr_type)])}
        cur_df = pd.DataFrame(cur_df)
        if row['task'] == 'regression':
            if reg_df is None:
                reg_df = cur_df
            else:
                reg_df = reg_df.append(cur_df)
        elif row['task'] == 'classification':
            if class_df is None:
                class_df = cur_df
            else:
                class_df = class_df.append(cur_df)
    reg_df = reg_df.groupby(['model','fracs']).mean()
    reg_df.reset_index(inplace=True)
    class_df = class_df.groupby(['model','fracs']).mean()
    class_df.reset_index(inplace=True)
    plot_df = pd.merge(reg_df, class_df, how='outer')
    plot_df['fracs'] = 1 - plot_df['fracs']  # For consistency with x-axis of culled correlation line plots above.
    sns.scatterplot(data=plot_df, x='regression_culled_{}'.format(corr_type), y='classification_culled_{}'.format(corr_type), hue='model', size='fracs', sizes=(5, 100), palette=palette, hue_order=order, alpha=0.8, ax=ax, linewidth=0, edgecolor='none')
    diag_x = np.linspace(min(plot_df['regression_culled_{}'.format(corr_type)].min(), plot_df['classification_culled_{}'.format(corr_type)].min()),
                         min(plot_df['regression_culled_{}'.format(corr_type)].max(), plot_df['classification_culled_{}'.format(corr_type)].max()),
                         10)
    ax.plot(diag_x, diag_x, color='k', linestyle='dashed', linewidth=1)
    ax.set_xlabel('Regression {}'.format(corr_type.capitalize()))
    ax.set_ylabel('Classification {}'.format(corr_type.capitalize()))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(os.path.join(out_dir, '{}_culled_{}_paired_plot.png'.format(out_tag, corr_type)), dpi=300, transparent=False, bbox_inches='tight', facecolor='white')
    plt.close()


def main(args):
    set_rc_params()
    out_dir = '../outputs' if args.out_dir is None else args.out_dir
    out_tag = args.out_description
    corr_type = args.correlation
    
    results_df = None
    results_files = args.results_files
    for r_file in results_files:
        cur_results = np.load(r_file, allow_pickle=True).item()
        cur_df = pd.DataFrame(cur_results['metrics'])
        cur_df['model'] = pd.Series(cur_results['meta']['model_paths']).apply(get_model_name)
        cur_df['task'] = pd.Series(cur_results['meta']['model_paths']).apply(get_task)
        cur_df['fracs'] = [cur_results['meta']['culled_fracs']] * len(cur_df)
        if results_df is None:
            results_df = cur_df
        else:
            results_df = results_df.append(cur_df)
    results_df.reset_index(inplace=True)
    
    make_culled_correlation_plot(results_df, out_dir, out_tag, corr_type)
    make_correlation_barplots(results_df, out_dir, out_tag)
    if corr_type == 'ndcg':
        make_culled_correlation_paired_plot(results_df, out_dir, out_tag, 'ndcg')
    else:
        make_culled_correlation_paired_plot(results_df, out_dir, out_tag, corr_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_files', help='path(s) to file(s) containing model results', nargs='+', type=str)
    parser.add_argument('--out_dir', help='directory to which to save the generated plots', type=str)
    parser.add_argument('--out_description', default='dre', help='descriptive tag to add to output filenames', type=str)
    parser.add_argument('--correlation', default='pearson', help='type of correlation to plot', type=str)
    args = parser.parse_args()
    main(args)