import os
import sys
sys.path.append('..')
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn import metrics
from tensorflow import keras
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
    
    data_path, model_path = args.data_path, args.model_path
    print('\nLoading data from {}'.format(data_path))
    df = pd.read_csv(data_path)
    seqs, targets = df[args.seq_column], df[args.target_column].values
    
    target_col_to_axis_label = {
        'Tm': r'Tm ($\degree$C)',
        'deltaGunf thermal': r'$\Delta G_{unf}$ (kcal/mol)',
        'deltaGunf chemical': r'$\Delta G_{unf}$ (kcal/mol)',
        'IC50_muM': r'$IC_{50} (\mu M)$',
        'neg_IC50_muM': r'$-IC_{50} (\mu M)$',
        'delta_ln_ka_literature': r'$\Delta ln(K_{A})$',
        'log_10_kcat_km': r'$\log_{10}(k_{cat} / K_m)$',
        'T_50': r'$T_{50}$',
    }
#     colors = sns.color_palette('flare', n_colors=len(seqs))
#     colors = [colors[i] for i in np.argsort(targets)]
    colors = sns.color_palette('flare')[0]
    fig, ax = plt.subplots(figsize=(2 , 2))
    min_pred, max_pred = np.inf, -np.inf
    
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
        if args.output_idx == -1:
            preds = np.mean([logits_to_log_density_ratio(preds, post_idx=i, pre_idx=i-1) for i in range(1, preds.shape[1])], axis=0)
        else:
            preds = logits_to_log_density_ratio(preds, post_idx=args.output_idx+1)
    else:
        if args.output_idx == -1:
            preds = np.mean(preds, axis=0).flatten()
        elif type(preds) == list:
            preds = preds[args.output_idx].flatten()
        else:
            preds = preds.flatten()
    min_pred = min(np.amin(preds), min_pred)
    max_pred = max(np.amax(preds), max_pred)
    
    ax.scatter(preds, targets, s=10, c=colors, alpha=0.75, edgecolor='none')
    sns.regplot(x=preds, y=targets, scatter=False, ci=None, color='gray', line_kws={'lw': 0.5, 'ls':'--'}, ax=ax)
    ax.set_ylabel(target_col_to_axis_label[args.target_column], fontsize=10)
    train_type = 'LER' if get_task(model_path) == 'regression' else 'DRC'
    ax.set_xlabel('{}-trained "{}" Prediction'.format(train_type, get_model_name(model_path)), fontsize=9)

    pearson, pearson_p = stats.pearsonr(targets, preds)
    spearman, spearman_p = stats.spearmanr(targets, preds)
    ax.annotate('Pearson={:0.2f}, p={:0.3f}\nSpearman={:0.2f}, p={:0.3f}'.format(np.nan_to_num(pearson), pearson_p, np.nan_to_num(spearman), spearman_p), (0.25, 0.45), xycoords='axes fraction', fontsize=8, bbox={'facecolor': 'lightgray', 'alpha': 0.2, 'pad': 1})
    method = 'DRC' if 'classifier' in model_path else 'LER'
    r2 = metrics.r2_score(targets, preds)
    print('Target,Method,Spearman,Spearman p-value,Pearson,Pearson p-value,r2,n')
    print('{},{},{},{},{},{},{},{}'.format(target_col_to_axis_label[args.target_column], method, spearman, spearman_p, pearson, pearson_p, r2, len(df)))

    ep = 0.25
    ax.set_xlim(min_pred - ep, max_pred + ep)
    plt.savefig(os.path.join(out_dir, '{}_prediction_paired_plot.png'.format(out_tag)), dpi=300, transparent=False, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='path to dataset', type=str)
    parser.add_argument('model_path', help='path to trained Keras predictive model', type=str)
    parser.add_argument('--seq_column', default='seq', help='column name in dataset containing sequences', type=str)
    parser.add_argument('--target_column', help='column name in dataset containing target values', type=str)
    parser.add_argument('--output_idx', default=0, help='index of model output to evaluate for multi-output models', type=int)
    parser.add_argument('--out_dir', help='directory to which to save the generated plots', type=str)
    parser.add_argument('--out_description', default='dre', help='descriptive tag to add to output filenames', type=str)
    args = parser.parse_args()
    main(args)