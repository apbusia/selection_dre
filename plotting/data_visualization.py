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
    
    df = pd.read_csv(args.data_file)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    
    # Fitness distribution
    ax = axes[0]
    sns.histplot(data=df, x=args.fitness_column, kde=True, ax=ax)
    
    # Enrichment distribution
    ax = axes[1]
    sns.histplot(data=df, x=args.enrichment_column, kde=True, ax=ax)
    
    # Pre count distribution
    ax = axes[2]
    df[args.pre_column] = df[args.pre_column] + 1  # Add pseudocount for log-scale
    sns.histplot(data=df, x=args.pre_column, log_scale=True, ax=ax)
    
    # Post count distribution
    ax = axes[3]
    df[args.post_column] = df[args.post_column] + 1  # Add pseudocount for log-scale
    sns.histplot(data=df, x=args.post_column, log_scale=True, ax=ax)
    
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
