import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model_comparison import set_rc_params


def main(args):
    set_rc_params()
    out_dir = '../outputs' if args.out_dir is None else args.out_dir
    out_tag = args.out_description
    
    df = pd.read_csv(args.data_file)
    
    fig, axes = plt.subplots(3, 1, figsize=(3, 9))
    palette = sns.color_palette('crest', n_colors=1) + sns.color_palette('flare', n_colors=1)
    
#     # Fitness distribution
#     ax = axes[0]
#     sns.histplot(data=df, x=args.fitness_column, stat='density', kde=True, color=palette[0], ax=ax)
    
    # Enrichment distribution
    ax = axes[0]
    sns.histplot(data=df, x=args.enrichment_column, stat='density', color=palette[0], ax=ax)
    ax.set_xlabel('Groundtruth Log-enrichment')
    
    ax = axes[1]
    pre_counts = df[args.pre_column] + 1
    post_counts = df[args.post_column] + 1
    df['observed'] = np.log((post_counts / np.sum(post_counts)) / (pre_counts / np.sum(pre_counts)))
#     sns.histplot(data=df[['groundtruth', 'observed']], stat='density', kde=True, common_norm=False, palette=palette, ax=ax)
    sns.histplot(data=df, x='observed', stat='density', color=palette[0], ax=ax)
    ax.set_xlabel('Observed Log-enrichment')
    
    # Count distributions
    ax = axes[2]
    df[args.pre_column] = df[args.pre_column] + 1 * (df[args.pre_column] == 0)  # Add pseudocount for log-scale
    df[args.post_column] = df[args.post_column] + 1 * (df[args.post_column] == 0) # Add pseudocount for log-scale
    df = df.rename(columns={args.pre_column: 'pre', args.post_column: 'post'})
    sns.histplot(data=df[['pre', 'post']], stat='proportion', multiple='dodge', log_scale=True, common_norm=False, palette=palette, ax=ax)
#     sns.histplot(data=df[['pre', 'post']], stat='proportion', element='step', cumulative=True, fill=False, common_norm=False, palette=palette, ax=ax)
#     ax.set_ylim(0, 1)
    ax.set_xlabel('Sequencing Count')

#     m = df['post'].max()
#     g = sns.JointGrid(data=df, x='pre', y='post', height=3) #, xlim=(0, m), ylim=(0, m), marginal_ticks=True)
#     # g.ax_joint.set(xscale='log', yscale='log')
# #     cax = g.figure.add_axes([.15, .55, .02, .2])
#     g.plot_joint(sns.histplot, cmap=sns.light_palette(palette[0], as_cmap=True)) #, cbar=True, cbar_ax=cax)
#     g.plot_marginals(sns.histplot, element='step', stat='proportion', color=palette[0])
#     g.set_axis_labels('Pre Count', 'Post Count')

#     g = sns.JointGrid(data=df, x='groundtruth', y='observed', height=3) #, xlim=(0, m), ylim=(0, m), marginal_ticks=True)
#     # g.ax_joint.set(xscale='log', yscale='log')
# #     cax = g.figure.add_axes([.15, .55, .02, .2])
#     g.plot_joint(sns.histplot, cmap=sns.light_palette(palette[0], as_cmap=True)) #, cbar=True, cbar_ax=cax)
#     g.plot_marginals(sns.histplot, element='step', stat='density', color=palette[0])
#     g.set_axis_labels('Groundtruth Log-enrichment', 'Observed Log-enrichment')
    
    plt.savefig(os.path.join(out_dir, '{}_data_visualization.png'.format(out_tag)), dpi=300, transparent=False, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', help='path to CSV file containing library fitnesses', type=str)
    parser.add_argument('fitness_column', help='column name in data_file', type=str)
    parser.add_argument('--enrichment_column', default='true_enrichment', help='column name in data_file', type=str)
    parser.add_argument('--pre_column', default='pre_count_0', help='column name in data_file', type=str)
    parser.add_argument('--post_column', default='post_count_0', help='column name in data_file', type=str)
    parser.add_argument('--out_dir', help='directory to which to save the generated plots', type=str)
    parser.add_argument('--out_description', default='dre', help='descriptive tag to add to output filenames', type=str)
    args = parser.parse_args()
    main(args)
