import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model_comparison import set_rc_params


def main(args):
    set_rc_params()
    out_dir = '../outputs' if args.out_dir is None else args.out_dir
    out_tag = args.out_description
    
    results_path = args.results_path
    print('\nLoading data from {}'.format(results_path))
    df = pd.read_csv(results_path)
    
    # Only plot results using our models.
    df = df.loc[(df['model'].isin(["NN, 100", "Linear, IS"])) & (df['method'].isin(['DRC', 'LER']))]
    
    colors = sns.light_palette(sns.color_palette('flare')[0], n_colors=3, reverse=True)
    fig, ax = plt.subplots(figsize=(2 , 6))
    fig, ax = plt.subplots(figsize=(6 , 2))

    df['dataset_labels'] = df['dataset'] + '\n(n=' + df['n_measurements'].astype(str) + ')'
#     sns.barplot(data=df, x='spearman', y='dataset_labels', hue='method', palette=colors, saturation=0.75, ax=ax)
#     ax.set_ylabel('', fontsize=10)
#     ax.set_xlabel('Spearman', fontsize=10)
    sns.barplot(data=df, y='spearman', x='dataset_labels', hue='method', palette=colors, saturation=0.75, ax=ax)
    ax.set_ylabel('Spearman', fontsize=10)
    ax.set_xlabel('', fontsize=10)
    
    ax.get_legend().set_title('')
    plt.legend(bbox_to_anchor=(1, 1.05), loc=10, borderaxespad=0.)

    plt.savefig(os.path.join(out_dir, '{}_experimental_results_barplot.png'.format(out_tag)), dpi=300, transparent=False, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_path', help='path to CSV of experimental results', type=str)
    parser.add_argument('--out_dir', help='directory to which to save the generated plots', type=str)
    parser.add_argument('--out_description', default='dre', help='descriptive tag to add to output filenames', type=str)
    args = parser.parse_args()
    main(args)