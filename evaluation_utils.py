# Based on structure-guided-regression codebase

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import stats


def get_eval_metrics(truth, predictions, use_classification_metrics=False, sample_weight=None):
    if use_classification_metrics:
        return {
        'zero_one': metrics.zero_one_loss(truth, predictions, sample_weight=sample_weight),
        'precision': metrics.precision_score(truth, predictions, average='micro', sample_weight=sample_weight),
        'recall': metrics.recall_score(truth, predictions, average='micro', sample_weight=sample_weight),
        }
    return {
        'mse': metrics.mean_squared_error(truth, predictions, sample_weight=sample_weight),
        'r2': metrics.r2_score(truth, predictions, sample_weight=sample_weight),
        'var_explained': metrics.explained_variance_score(truth, predictions, sample_weight=sample_weight),
        'pearson_r': get_pearsonr(truth, predictions),
        'spearman_r': get_spearmanr(truth, predictions),
        'ndcg': metrics.ndcg_score(np.expand_dims(np.exp(truth), axis=0), np.expand_dims(predictions, axis=0)),
    }


def print_eval_metrics(metric_dict):
    if 'mse' in metric_dict:
        key_to_text = [
            ('mse', 'MSE'),
            ('r2', 'R2 Score'),
            ('var_explained', 'Variance Explained Score'),
            ('pearson_r', 'Pearson correlation coefficient'),
            ('spearman_r', 'Spearman correlation coefficient'),
            ('ndcg', 'Normalized Dicounted Cumulative Gain'),
        ]
    else:
        key_to_text = [
            ('zero_one', 'Misclassification Rate'),
            ('precision', 'Precision'),
            ('recall', 'Recall'),
        ]
    
    for key, text in key_to_text:
        metric = np.array(metric_dict[key])
        if metric.size:
            print('\t{} = {:.4f} +/- {:.4f}'.format(
                text, metric.mean(), metric.std()))
        else:
            print('\t{} = {:.4f}'.format(text, metric))
    print()


def print_cross_validation_metrics(fold_metrics, folds):
    metrics_to_print = [
        ('test_neg_mean_squared_error', 'Test MSE'),
        ('train_r2', 'Train R2 Score'),
        ('train_explained_variance', 'Train Variance Explained Score'),
        ('train_pearson', 'Train Pearson correlation coefficient'),
        ('train_spearman', 'Train Spearman correlation coefficient'),
    ]
    print('\n{}-fold Cross-Validation metrics:'.format(folds))
    for metric_tag, metric_name in metrics_to_print:
        mean_metric =  fold_metrics[metric_tag].mean()
        if 'neg' in metric_tag:
            mean_metric *= -1
        print('\t{} = {:.4f} +/- {:.4f}'.format(
            metric_name,
            mean_metric,
            fold_metrics[metric_tag].std()))


def get_pearsonr(truth, predictions):
    return stats.pearsonr(truth, predictions)[0]


def get_spearmanr(truth, predictions):
    return stats.spearmanr(truth, predictions)[0]


def get_scoring_metrics():
    """Defines cross-validation scoring metrics."""
    # Add Pearson and Spearman correlations as sklearn metrics.
    return {
        'pearson': metrics.make_scorer(get_pearsonr),
        'spearman': metrics.make_scorer(get_spearmanr),
        'r2': 'r2',
        'explained_variance': 'explained_variance',
        'neg_mean_squared_error': 'neg_mean_squared_error',
    }


def make_paired_plot(x, y, xlabel, ylabel, title=None):
    # Make a paired plot with marginal histograms.
    g = sns.jointplot(x=x, y=y, color='m',
                      joint_kws=dict(linewidth=0, edgecolor='none', alpha=0.1, s=25),
                      marginal_kws=dict(kde=True))
    g.ax_joint.plot(y, y, color='k', linestyle=':')
    g.set_axis_labels(xlabel=xlabel, ylabel=ylabel)
    if title:
        plt.suptitle(title)
        g.fig.subplots_adjust(top=0.95) # Reduce plot to make room


def plot_errors(predictions, y, error_type):
    if error_type not in ['squared', 'absolute']:
        raise ValueError('{} not a valid error type.'.format(error_type))
    if error_type == 'squared':
        errors = np.square(predictions - y)
        error_name = 'Squared Error'
    elif error_type == 'absolute':
        errors = np.abs(predictions - y)
        error_name = 'Absolute Error'
    
    print('{} Summary Statistics:'.format(error_name))
    print('\tMean {} +/- std = {} +/- {}'.format(
        error_name, np.mean(errors), np.std(errors)))
    print('\tMin {} = {}'.format(error_name, np.amin(errors)))
    for p in np.array([10, 25, 50, 75, 90]):
        print('\t{}th Percentile ={}'.format(p, np.percentile(errors, p)))
    print('\tMax {} = {}'.format(error_name, np.amax(errors)))

    # Plot distribution of errors.
    plt.subplots(figsize=(10,5))
    sns.distplot(errors, color="m", axlabel=error_name)

    # Plot true label versus error.
    plt.subplots(figsize=(10,5))
    sns.scatterplot(y, errors,
                    color='m', edgecolor='m', alpha=0.25, s=25)
    plt.xlabel('y')
    plt.ylabel(error_name)

    # Box plot of the errors.
    plt.subplots(figsize=(20,5))
    sns.boxplot(errors)
    
    
def make_errorbar_plot(xticks, vals, std_errs, title, figsize=(15,5), ylim=None):
    with sns.axes_style("whitegrid"):
        plt.subplots(figsize=figsize)

        plt.errorbar(
            x=np.arange(len(xticks)),
            y=vals,
            yerr=std_errs,
            fmt='o',
            color='black',
            ecolor='lightgray',
            elinewidth=3,
            capsize=0)

        plt.xticks(ticks=np.arange(len(xticks)), labels=xticks)
        plt.grid(axis='x', linewidth=0.0)
        
        if ylim is not None:
            plt.ylim(ylim)

        plt.title(title)
        sns.despine()


def calculate_culled_correlation(ypred, ytest, fracs, correlation_type='pearson'):
    """
    Calculates the correlation between predictions and true fitness values
    among subsets of test data. In particular, test data is succesively culled to only
    include the largest true fitness values.
    """
    if correlation_type not in ['pearson', 'spearman', 'mse']:
        raise NotImplementedError('culled correlations not implemented for correlation_type={}'.format(correlation_type))
    if correlation_type == 'mse':
        get_correlation = metrics.mean_squared_error
    elif correlation_type == 'pearson':
        get_correlation = get_pearsonr
    else:
        get_correlation = get_spearmanr
    corrs = []
    n_test = len(ypred)
    y_test = ytest[:n_test]
#     sorted_test_idx = np.argsort(ypred)
    sorted_test_idx = np.argsort(y_test)
    for i in range(len(fracs)):
        num_frac = int(n_test * fracs[i])
        idx = sorted_test_idx[num_frac:]
        ypred_frac = ypred[idx]
        ytest_frac = ytest[idx]
        c = get_correlation(ytest_frac, ypred_frac)
        corrs.append(c)
    return corrs


def calculate_culled_ndcg(ypred, ytest, fracs):
    """
    Calculates Normalized Discounted Cumulative Gain between predictions and true fitness values
    among subsets of test data. In particular, test data is succesively culled to only
    include the largest true fitness values.
    """
    ndcgs = []
    n_test = len(ypred)
    y_test = ytest[:n_test]
    sorted_test_idx = np.argsort(y_test)
    ypred = np.expand_dims(ypred, axis=0)
    ytest = np.expand_dims(ytest, axis=0)
    for i in range(len(fracs)):
        num_frac = int(n_test * fracs[i])
        top_k = n_test - num_frac
        ndcg = metrics.ndcg_score(ytest, ypred, k=top_k)
        ndcgs.append(ndcg)
    return ndcgs