import os
import sys
sys.path.append('..')
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tensorflow import keras
from matplotlib.lines import Line2D
import data_prep
from run_models import disable_gpu
from evaluate_models import get_model_type, get_encoding_type, get_model_predictions, logits_to_log_density_ratio
from model_comparison import set_rc_params, get_model_name, get_task


def sci_notation(n, sig_fig=2):
    if n > 9999:
        fmt_str = '{0:.{1:d}e}'.format(n, sig_fig)
        n, exp = fmt_str.split('e')
        # return n + ' x 10^' + str(int(exp))
        return r'${n:s} \times 10^{{{e:d}}}$'.format(n=n, e=int(exp))
    return str(n)


def get_text_label(p, t, min_t=None, max_t=None):
    if min_t is not None and t <= min_t:
        label = '({:0.2f}, {}, True Min)'
    elif max_t is not None and t >= max_t:
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
    log_titers = np.log10(titers)
    
    if 'source' in df.columns:
        unique_sources = pd.unique(df['source'])
        int_source = list(df['source'].astype('category').cat.codes)
        colors = sns.color_palette('Accent', n_colors=len(unique_sources))
        legend_elements = [
            Line2D([0], [0], marker='o', label=source, color=color, markersize=10, linewidth=0, linestyle='')
            for source, color in zip(unique_sources, colors)]
        colors = [colors[i] for i in int_source]
    else:
        colors = sns.color_palette('flare', n_colors=len(seqs))
        legend_elements = None
    fig, axes = plt.subplots(1, len(model_paths), figsize=(2 * len(model_paths), 2), sharey=True)
    min_pred, max_pred = np.inf, -np.inf
    
    for i, model_path in enumerate(model_paths):
        disable_gpu([0, 1])
        keras.backend.clear_session()
        
        print('\nUsing model {}'.format(model_path))
        model = keras.models.load_model(model_path)
        encoding_type = get_encoding_type(model_path)
        if encoding_type == 'pairwise':
            encoding = data_prep.index_encode_is_plus_pairwise
        elif encoding_type == 'is':
            encoding = data_prep.index_encode
        elif encoding_type == 'neighbors':
            encoding = data_prep.index_encode_is_plus_neighbors
        print('\n\tGetting model predictions...')
        model_type = get_model_type(model_path)
        preds = get_model_predictions(model, model_type, seqs, encoding)
        del model
        if 'logistic' in model_type:
            preds = logits_to_log_density_ratio(preds)
        else:
            preds = preds.flatten()
        min_pred = min(np.amin(preds), min_pred)
        max_pred = max(np.amax(preds), max_pred)
        
        ax = axes[i]
        ax.scatter(preds, log_titers, s=20, c=colors)
        sns.regplot(x=preds, y=log_titers, scatter=False, ci=None, color='gray', line_kws={'lw': 1, 'ls':'--'}, ax=ax)
        if i == 0:
            ax.set_ylabel(r'Titer (Viral Genome/$\mu$L)', fontsize=10)
        train_type = 'LER' if get_task(model_path) == 'regression' else 'DRC'
        ax.set_xlabel('{}-trained {} Prediction'.format(train_type, get_model_name(model_path)), fontsize=9)
        
        if args.annotate_points:
            min_t, max_t = np.amin(titers), np.amax(titers)
            for p, t, lt in zip(preds, titers, log_titers):
                label, xytext, ha = get_text_label(p, t, min_t, max_t)
                ax.annotate(label.format(p, sci_notation(t)), (p, lt), textcoords='offset pixels', xytext=xytext, fontsize=6, ha=ha)
        
        pearson, pearson_p = stats.pearsonr(log_titers, preds)
        spearman, spearman_p = stats.spearmanr(log_titers, preds)
#         plt.figtext((0.5 + i) / (len(model_paths)), -0.1, 'Pearson = {:0.2f},\nSpearman={:0.2f}'.format(pearson, spearman), ha='center', fontsize=10, bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'pad': 5})
        ax.annotate('Pearson = {:0.2f}, p={:0.2f}\nSpearman={:0.2f}, p={:0.2f}'.format(pearson, pearson_p, spearman, spearman_p), (-1, 5), fontsize=8, bbox={'facecolor': 'lightgray', 'alpha': 0.2, 'pad': 2})
                    
    ep = 0.25
    for ax in axes:
        ax.set_xlim(min_pred - ep, max_pred + ep)
    if legend_elements is not None:
        fig.legend(handles=legend_elements, loc='center right', fontsize=10)
    plt.savefig(os.path.join(out_dir, '{}_titer_comparison_plot.png'.format(out_tag)), dpi=300, transparent=False, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='path to (seq, titer) dataset', type=str)
    parser.add_argument('model_paths', help='path to trained Keras predictive model', nargs='+', type=str)
    parser.add_argument('--annotate_points', help='whether to annotate each point plotted', action='store_true')
    parser.add_argument('--out_dir', help='directory to which to save the generated plots', type=str)
    parser.add_argument('--out_description', default='dre', help='descriptive tag to add to output filenames', type=str)
    args = parser.parse_args()
    main(args)