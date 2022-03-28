import os
import sys
sys.path.append('..')
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
import data_prep
from run_models import disable_gpu
from evaluate_models import get_model_type, get_encoding_type, get_model_predictions, logits_to_log_density_ratio
from model_comparison import set_rc_params, get_model_name, get_task
from evaluation_utils import get_pearsonr, get_spearmanr


def sci_notation(n, sig_fig=2):
    if n > 9999:
        fmt_str = '{0:.{1:d}e}'.format(n, sig_fig)
        n, exp = fmt_str.split('e')
        # return n + ' x 10^' + str(int(exp))
        return r'${n:s} \times 10^{{{e:d}}}$'.format(n=n, e=int(exp))
    return str(n)


def get_text_label(p, t):
    if t < 2e4:
        label = '({:0.2f}, {}, True Min)'
    elif t > 1e12:
        label = '({:0.2f}, {}, True Max)'
    else:
        label = '({:0.2f}, {})'
    if p < 2:
        ha = 'left'
        xytext=(12, 0)
    elif p > 4:
        ha = 'right'
        xytext=(-12, -24)
    else:
        ha = 'right'
        xytext=(-24, -24)
    return label, xytext, ha


def main(args):
    set_rc_params()
    out_dir = '../outputs' if args.out_dir is None else args.out_dir
    out_tag = args.out_description
    
    data_path, model_paths = args.data_path, args.model_paths
    print('\nLoading data from {}'.format(data_path))
    df = pd.read_csv(data_path)
    seqs, titers = df['seq'], df['titer'].values
    
    colors = sns.color_palette('flare', n_colors=5)
    fig, axes = plt.subplots(1, len(model_paths), figsize=(5 * len(model_paths), 4))
    
    for i, model_path in enumerate(model_paths):
        disable_gpu(1)
        keras.backend.clear_session()
        
        print('\nUsing model {}'.format(model_path))
        model = keras.models.load_model(model_path)
        encoding_type = get_encoding_type(model_path)
        if encoding_type == 'pairwise':
            encoding = data_prep.encode_one_plus_pairwise
        elif encoding_type == 'is':
            encoding = data_prep.one_hot_encode
        elif encoding_type == 'neighbors':
            encoding = data_prep.encode_one_plus_neighbors
        print('\n\tGetting model predictions...')
        model_type = get_model_type(model_path)
        preds = get_model_predictions(model, model_type, seqs, encoding)
        del model
        if 'logistic' in model_type:
            preds = logits_to_log_density_ratio(preds)
        else:
            preds = preds.flatten()
        
        ax = axes[i]
        ax.scatter(preds, titers, s=20, c=colors)
        ax.set_yscale('log')
        ax.set_ylabel(r'Viral Genome (vg/$\mu$L)', fontsize=14)
        ax.set_xlabel('Predicted Log Enrichment', fontsize=14)
        ax.set_title('{} {} Model vs. Titer'.format(get_model_name(model_path), get_task(model_path)), fontsize=14)
        
        for p, t in zip(preds, titers):
            label, xytext, ha = get_text_label(p, t)
            ax.annotate(label.format(p, sci_notation(t)), (p, t), textcoords='offset pixels', xytext=xytext, fontsize=10, ha=ha)
        
        pearson = get_pearsonr(titers, preds)
        spearman = get_spearmanr(titers, preds)
        plt.figtext((0.5 + i) / (len(model_paths)), -0.1, 'Pearson = {:0.2f},\nSpearman={:0.2f}'.format(pearson, spearman), ha='center', fontsize=10, bbox={'facecolor': 'gray', 'alpha': 0.5, 'pad': 5})
    
    plt.savefig(os.path.join(out_dir, '{}_titer_comparison_plot.png'.format(out_tag)), dpi=300, transparent=False, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='path to (seq, titer) dataset', type=str)
    parser.add_argument('model_paths', help='path to trained Keras predictive model', nargs='+', type=str)
    parser.add_argument('--out_dir', help='directory to which to save the generated plots', type=str)
    parser.add_argument('--out_description', default='dre', help='descriptive tag to add to output filenames', type=str)
    args = parser.parse_args()
    main(args)