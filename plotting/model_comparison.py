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
    if 'cnn' in model_path:
        match = re.search('(?<=cnn_)(classifier_)?(\d+)x(\d+)x(\d+)', model_path)
        n_layers = match.group(2)
        win_size = match.group(3)
        n_units = match.group(4)
        return 'CNN, {}x{}x{}'.format(n_layers, win_size, n_units)


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


def make_culled_correlation_plot(results_df, out_dir, out_tag, corr_type, include_cnn=False):
    fig, axes = plt.subplots(1, 2, figsize=(4, 2))
    palette, order = get_color_palette(include_cnn)
    reg_df, class_df = None, None
    for _, row in results_df.iterrows():
        cur_df = {'fracs': row['fracs'],  # np.arange(0, 1, 0.01),
                  'culled_{}'.format(corr_type): np.nan_to_num(row['culled_{}'.format(corr_type)]),
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
    sns.lineplot(data=class_df, x='fracs', y='culled_{}'.format(corr_type), hue='model', palette=palette, hue_order=order, err_style='band', ax=axes[1], legend=False)
    bottom_lim = min(0, axes[0].get_ylim()[0], axes[1].get_ylim()[0])
    top_lim = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    for ax, task in zip(axes, ['LER', 'DRC']):
        ax.set_xlabel('Fraction of top test sequences')
        ax.set_ylabel('{} {}'.format(task, corr_type.capitalize()))
        fracs2 = [f for i, f in enumerate(np.arange(0, 1, 0.01)) if i % 20 == 0]
        ax.set_xticks(fracs2)
        ax.set_xticklabels(["%.1f" % (1-f) for f in fracs2])
        ax.set_ylim(bottom_lim, top_lim)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.tight_layout(pad=0.5)
    plt.savefig(os.path.join(out_dir, '{}_culled_correlation_plot.png'.format(out_tag)), dpi=300, transparent=False, bbox_inches='tight', facecolor='white')
    plt.close()


def make_correlation_barplots(results_df, out_dir, out_tag, include_cnn=False):
    f, axes = plt.subplots(1, 3, figsize=(9, 3))
    palette, order = get_color_palette(include_cnn)
    sns.barplot(data=results_df, x='task', y='pearson_r', hue='model', palette=palette, hue_order=order, ax=axes[0])
    axes[0].legend_.remove()
    axes[0].set_ylim(-1, 1)
    axes[0].set_title('Pearson Correlation')
    sns.barplot(data=results_df, x='task', y='spearman_r', hue='model', palette=palette, hue_order=order, ax=axes[1])
    axes[1].legend_.remove()
    axes[1].set_ylim(-1, 1)
    axes[1].set_title('Spearman Correlation')
    sns.barplot(data=results_df, x='task', y='mse', hue='model', palette=palette, hue_order=order, ax=axes[2])
    axes[2].set_title('Mean Squared Error')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(os.path.join(out_dir, '{}_model_comparison_barplot.png'.format(out_tag)), dpi=300, transparent=False, bbox_inches='tight', facecolor='white')
    plt.close()


def make_correlation_paired_plots(results_df, out_dir, out_tag, include_cnn=False):
    f, axes = plt.subplots(1, 3, figsize=(9, 3))
    palette, order = get_color_palette(include_cnn)
    metrics = ['pearson_r', 'spearman_r', 'mse']
    reg_df, class_df = None, None
    for _, row in results_df.iterrows():
        cur_df = {'{}_{}'.format(row['task'], metric): row[metric] for metric in metrics}
        cur_df['model'] = [row['model']] #* len(row[metric])
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
    reg_gb = reg_df.groupby(['model'])
    reg_df = pd.merge(reg_gb.mean().reset_index(), reg_gb.std().reset_index(), on='model', how='inner', suffixes=('', '_std'))
    class_gb = class_df.groupby(['model'])
    class_df = pd.merge(class_gb.mean().reset_index(), class_gb.std().reset_index(), on='model', how='inner', suffixes=('', '_std'))
    plot_df = pd.merge(reg_df, class_df, on='model', how='inner')
    for i, metric in enumerate(metrics):
        ax = axes[i]
        sns.scatterplot(data=plot_df, x='regression_{}'.format(metric), y='classification_{}'.format(metric), hue='model', palette=palette, hue_order=order, alpha=0.8, ax=ax, linewidth=0, edgecolor='none')
        ax.errorbar(data=plot_df, x='regression_{}'.format(metric), y='classification_{}'.format(metric), yerr='classification_{}_std'.format(metric), xerr='regression_{}_std'.format(metric), fmt='none', ecolor='k')
        diag_x = np.linspace(min(plot_df['regression_{}'.format(metric)].min(), plot_df['classification_{}'.format(metric)].min()),
                             min(plot_df['regression_{}'.format(metric)].max(), plot_df['classification_{}'.format(metric)].max()),
                             10)
        ax.plot(diag_x, diag_x, color='k', linestyle='dashed', linewidth=1)
        tag = 'MSE'
        if metric == 'pearson_r':
            tag = 'Pearson'
        if metric == 'spearman_r':
            tag = 'Spearman'
        ax.set_xlabel('LER {}'.format(tag))
        ax.set_ylabel('DRC {}'.format(tag))
        if i < len(axes) - 1:
            ax.legend_.remove()
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.).get_texts()[-1].set_text('Error Bars')
    plt.savefig(os.path.join(out_dir, '{}_model_comparison_paired_plot.png'.format(out_tag)), dpi=300, transparent=False, bbox_inches='tight', facecolor='white')
    plt.close()


def make_culled_correlation_paired_plot(results_df, out_dir, out_tag, corr_type, include_cnn=False):
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    palette, order = get_color_palette(include_cnn)
    reg_df, class_df = None, None
    for _, row in results_df.iterrows():
        cur_df = {'fracs': row['fracs'],  # np.arange(0, 1, 0.01),
                  '{}_culled_{}'.format(row['task'], corr_type): np.nan_to_num(row['culled_{}'.format(corr_type)]),
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
    sns.scatterplot(data=plot_df, x='regression_culled_{}'.format(corr_type), y='classification_culled_{}'.format(corr_type), hue='model', size='fracs', sizes=(5, 25), palette=palette, hue_order=order, alpha=0.7, ax=ax, linewidth=0, edgecolor='none')
    if corr_type in ['pearson', 'spearman']:
        bottom_lim = min(0, ax.get_ylim()[0], ax.get_xlim()[0])
        top_lim = max(1, ax.get_ylim()[1], ax.get_xlim()[1])
    else:
        bottom_lim = min(
            plot_df['regression_culled_{}'.format(corr_type)].min(), plot_df['classification_culled_{}'.format(corr_type)].min())
        top_lim = max(
            plot_df['regression_culled_{}'.format(corr_type)].max(), plot_df['classification_culled_{}'.format(corr_type)].max())
    ax_buffer = (top_lim - bottom_lim) * 0.075
    bottom_lim, top_lim = bottom_lim - ax_buffer, top_lim + ax_buffer
    ax.set_ylim(bottom_lim, top_lim)
    ax.set_xlim(bottom_lim, top_lim)
    ax.set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])
    diag_x = np.linspace(bottom_lim, top_lim, 10)
    ax.plot(diag_x, diag_x, color='k', linestyle='dashed', linewidth=1)
    ax.set_xlabel('LER {}'.format(corr_type.capitalize()))
    ax.set_ylabel('DRC {}'.format(corr_type.capitalize()))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(os.path.join(out_dir, '{}_culled_{}_paired_plot.png'.format(out_tag, corr_type)), dpi=300, transparent=False, bbox_inches='tight', facecolor='white')
    plt.close()


def main(args):
    set_rc_params()
    out_dir = '../outputs' if args.out_dir is None else args.out_dir
    out_tag = args.out_description
    corr_type = args.correlation
    include_cnn = args.include_cnn
    
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
    
    make_culled_correlation_plot(results_df, out_dir, out_tag, corr_type, include_cnn)
    make_correlation_barplots(results_df, out_dir, out_tag, include_cnn)
    make_correlation_paired_plots(results_df, out_dir, out_tag, include_cnn)
    if corr_type == 'ndcg':
        make_culled_correlation_paired_plot(results_df, out_dir, out_tag, 'ndcg', include_cnn)
    else:
        make_culled_correlation_paired_plot(results_df, out_dir, out_tag, corr_type, include_cnn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_files', help='path(s) to file(s) containing model results', nargs='+', type=str)
    parser.add_argument('--out_dir', help='directory to which to save the generated plots', type=str)
    parser.add_argument('--out_description', default='dre', help='descriptive tag to add to output filenames', type=str)
    parser.add_argument('--correlation', default='pearson', help='type of correlation to plot', type=str)
    parser.add_argument('--include_cnn', help='include convolutional architectures in plots', action='store_true')
    args = parser.parse_args()
    main(args)