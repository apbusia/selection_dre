import argparse
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import tensorflow as tf
from tensorflow import keras
from wandb.keras import WandbCallback
from sklearn.model_selection import KFold, train_test_split
import modeling
import data_prep
import evaluation_utils


SEED = 7


def get_savefile(args):
    savefile = 'models/'
    if args.model_type == 'linear':
        savefile += '%s_linear' % (args.encoding)
    elif args.model_type == 'ann':
        savefile += '%s_ann_%sx%s' % (args.encoding, args.n_hidden, args.hidden_size)
    elif args.model_type == 'cnn':
        savefile += '%s_cnn_%sx%sx%s' % (args.encoding, args.n_hidden, args.window_size, args.hidden_size)
    elif args.model_type == 'logistic_linear':
        savefile += '%s_linear_classifier' % (args.encoding)
    elif args.model_type == 'logistic_ann':
        savefile += '%s_ann_classifier_%sx%s' % (args.encoding, args.n_hidden, args.hidden_size)
    elif args.model_type == 'logistic_cnn':
        savefile += '%s_cnn_classifier_%sx%sx%s' % (args.encoding, args.n_hidden, args.window_size, args.hidden_size)
    if args.normalize:
        savefile += '_normalized'
    if args.weighted_loss:
        savefile += '_weighted'
    if args.description is not None:
        savefile += '_' + args.description
    return savefile


def disable_gpu(ind_list):
    # Convert integer argument for backwards compatibility.
    if type(ind_list) == int:
        ind_list = [ind_list]
    physical_devices = tf.config.list_physical_devices('GPU')
    n_physical_devices = len(physical_devices)
    ind_list.sort(reverse=True)
    try:
        # Disable GPU
        for ind in ind_list:
            del physical_devices[ind]
        tf.config.set_visible_devices(physical_devices, 'GPU')
        logical_devices = tf.config.list_logical_devices('GPU')
        # Logical device was not created for specified GPU
        assert len(logical_devices) == n_physical_devices - len(ind_list)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        print('Unable to disable GPU ', ind_list)
        pass


def run_training(seqs, pre_counts, post_counts, encoding, model_type, normalize=None,
                 lr=None, n_hidden=None, hidden_size=None, alpha=None, weighted_loss=None,
                 window_size=None, residual_channels=None, skip_channels=None,
                 train_idx=None, test_idx=None, val_idx=None, epochs=None, batch_size=None,
                 early_stopping=None, wandb=False, return_metrics=True, gradient_clip=None, savefile=None):
    eval_batch_size = 1024
    if encoding == 'pairwise':
        encoding = data_prep.index_encode_is_plus_pairwise
    elif encoding == 'is':
        encoding = data_prep.index_encode
    elif encoding == 'neighbors':
        encoding = data_prep.index_encode_is_plus_neighbors
    flatten = False if 'cnn' in model_type else True
    
    if model_type in ['linear', 'ann', 'cnn']:
        pre_counts = pre_counts + 1
        post_counts = post_counts + 1
        enrich_scores = data_prep.calculate_enrichment_scores(pre_counts, post_counts, pre_counts.sum(), post_counts.sum())
        if normalize:
            mean_en, var_en = enrich_scores[:, 0], enrich_scores[:, 1]
            mu_en = np.mean(mean_en)
            sig_en = np.std(mean_en)
            mean_en = (mean_en - mu_en) / sig_en
            var_en = (np.sqrt(var_en) / sig_en)**2
            enrich_scores[:, 0] = mean_en
            enrich_scores[:, 1] = var_en
        # If weighted_loss is True, then loss is Gaussian with variance according to log-ratio of two Binomials.
        # See Katz, 1978 for details on this variance.
        enrich_scores[:, 1] = 1. if not weighted_loss else 1. / (2 * enrich_scores[:, 1])
        classes = None
        counts = None
        monitor = 'val_mean_squared_error'
    elif model_type in ['logistic_linear', 'logistic_ann', 'logistic_cnn']:
        enrich_scores = None
        train_idx = np.random.permutation(np.concatenate([train_idx, np.array(train_idx) + len(seqs)]))
        if val_idx is not None:
            val_idx = np.concatenate([val_idx, np.array(val_idx) + len(seqs)])
        if test_idx is not None:
            test_idx = np.concatenate([test_idx, np.array(test_idx) + len(seqs)])
        counts = np.concatenate([pre_counts, post_counts])
#         counts = counts + 1  # Use same pseudo-count scheme as for regression models for consistency.
        if normalize:
            # Reweight counts to avoid exploding gradients but preserve relative amounts in each pool.
            counts = counts / np.amax(counts)
        classes = np.array([0] * len(seqs) + [1] * len(seqs), int)
        seqs = pd.concat([seqs, seqs.copy()], ignore_index=True)
        monitor = 'val_weighted_sparse_categorical_crossentropy'
    
    disable_gpu(0)
    keras.backend.clear_session()
    
    train_gen = modeling.get_dataset(seqs, train_idx, encoding, enrich_scores, classes, counts, batch_size=batch_size, seed=SEED, flatten=flatten)
    callbacks, val_gen, test_gen = None, None, None
    if wandb:
        callbacks = [WandbCallback()]
    if early_stopping:
        if val_idx is None:
            val_idx = test_idx
        es_callback = keras.callbacks.EarlyStopping(monitor=monitor, min_delta=0, patience=10, restore_best_weights=True)
        callbacks = [es_callback] if callbacks is None else callbacks + [es_callback]
        val_batch_size = min(eval_batch_size, len(val_idx))
        val_gen = modeling.get_dataset(seqs, val_idx, encoding, enrich_scores, classes, counts, batch_size=val_batch_size, shuffle=False, flatten=flatten)
    
    input_shape = tuple(train_gen.element_spec[0].shape[1:])
    if model_type == 'linear':
        model = modeling.make_linear_model(input_shape, lr=lr, l2_reg=alpha, gradient_clip=gradient_clip)
    elif model_type == 'ann':
        model = modeling.make_ann_model(input_shape, num_hid=n_hidden, hid_size=hidden_size, lr=lr, l2_reg=alpha, gradient_clip=gradient_clip)
    elif model_type == 'cnn':
        model = modeling.make_cnn_model(input_shape, num_hid=n_hidden, hid_size=hidden_size, win_size=window_size, residual_channels=residual_channels, skip_channels=skip_channels, lr=lr, l2_reg=alpha, gradient_clip=gradient_clip)
    elif model_type == 'logistic_linear':
        model = modeling.make_linear_classifier(input_shape, lr=lr, l2_reg=alpha, gradient_clip=gradient_clip)
    elif model_type == 'logistic_ann':
        model = modeling.make_ann_classifier(input_shape, num_hid=n_hidden, hid_size=hidden_size, lr=lr, l2_reg=alpha, gradient_clip=gradient_clip)
    elif model_type == 'logistic_cnn':
        model = modeling.make_cnn_classifier(input_shape, num_hid=n_hidden, hid_size=hidden_size, win_size=window_size, residual_channels=residual_channels, skip_channels=skip_channels, lr=lr, l2_reg=alpha, gradient_clip=gradient_clip)
    
    print('\nStarting training...')
    history_callback = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)
    loss_history = np.array(history_callback.history['loss'])
    if savefile is not None:
        model_savefile = savefile + '_model'
        model.save(model_savefile)
        loss_savefile = savefile + '_loss_history.npy'
        np.save(loss_savefile, loss_history)
    
    train_metrics, test_metrics = None, None
    if return_metrics:
        print('\nStarting evaluation...')
        with tf.device('/cpu:0'):
            train_gen = modeling.get_dataset(seqs, train_idx, encoding, enrich_scores, classes, counts, batch_size=eval_batch_size, shuffle=False, flatten=flatten)
            pred = model.predict(train_gen)
            use_classification_metrics = False
            if 'logistic' in model_type:
                pred = np.argmax(pred, axis=1)
                truth = classes[train_idx]
                use_classification_metrics = True
                if np.amax(counts) == 1:
                    counts = counts * max(np.amax(post_counts), np.amax(pre_counts))
                sample_weight = counts[train_idx]
            else:
                pred = pred.flatten()
                truth = enrich_scores[train_idx][:,0]
                sample_weight = None
        train_metrics = evaluation_utils.get_eval_metrics(truth, pred, use_classification_metrics, sample_weight)

        if test_idx is not None:
            with tf.device('/cpu:0'):
                test_batch_size = min(eval_batch_size, len(test_idx))
                test_gen = modeling.get_dataset(seqs, test_idx, encoding, enrich_scores, classes, counts, batch_size=test_batch_size, shuffle=False, flatten=flatten)
                pred = model.predict(test_gen)
                if 'logistic' in model_type:
                    if savefile is not None:
                        np.save(savefile + '_test_logits.npy', pred)
                    pred = np.argmax(pred, axis=1)
                    truth = classes[test_idx]
                    sample_weight = counts[test_idx]
                else:
                    pred = pred.flatten()
                    truth = enrich_scores[test_idx][:,0]
            test_metrics = evaluation_utils.get_eval_metrics(truth, pred, use_classification_metrics, sample_weight)
            if savefile is not None:
                test_pred_savefile = savefile + '_test_pred.npy'
                np.save(test_pred_savefile, pred)
                np.save(savefile + '_test_idx.npy', test_idx)
    
    del model
    return train_metrics, test_metrics


def main(args):
    np.random.seed(SEED)
    
    savefile = get_savefile(args)
    
    data_df = pd.read_csv(args.data_file)
    seqs = data_df['seq']
    pre_counts = data_df[args.pre_column]
    post_counts = data_df[args.post_column]
    
    if args.n_folds == 1:
        print('\nRunning on full dataset...')
        n_samples = len(seqs)
        train_metrics, _ = run_training(
            seqs, pre_counts, post_counts, args.encoding, args.model_type, normalize=args.normalize,
            lr=args.learning_rate, n_hidden=args.n_hidden, hidden_size=args.hidden_size, alpha=args.alpha,
            window_size=args.window_size, residual_channels=args.residual_channels, skip_channels=args.skip_channels,
            train_idx=np.arange(n_samples), epochs=args.epochs, batch_size=args.batch_size, weighted_loss=args.weighted_loss, gradient_clip=args.gradient_clip, savefile=savefile)
        print('\nTrain metrics:')
        evaluation_utils.print_eval_metrics(train_metrics)
        with open(savefile + '_train_metrics.pkl', 'wb') as f:
            pickle.dump(train_metrics, f)
    else:
        kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)
        train_metrics, test_metrics = defaultdict(list), defaultdict(list)
        for i, (train_idx, test_idx) in enumerate(kf.split(seqs)):
            print('\nRunning on fold {}...'.format(i+1))
            train_idx, val_idx = train_test_split(train_idx, test_size=0.2, shuffle=False, random_state=SEED)
            cur_train_metrics, cur_test_metrics = run_training(
                seqs, pre_counts, post_counts, args.encoding, args.model_type, normalize=args.normalize,
                lr=args.learning_rate, n_hidden=args.n_hidden, hidden_size=args.hidden_size, alpha=args.alpha,
                window_size=args.window_size, residual_channels=args.residual_channels, skip_channels=args.skip_channels,
                train_idx=train_idx, test_idx=test_idx, val_idx=val_idx, epochs=args.epochs, batch_size=args.batch_size,
                early_stopping=args.early_stopping, weighted_loss=args.weighted_loss, gradient_clip=args.gradient_clip,
                savefile=savefile + '_fold{}'.format(i))
            for k in cur_train_metrics.keys():
                train_metrics[k].append(cur_train_metrics[k])
                test_metrics[k].append(cur_test_metrics[k])

        print('\nCV train metrics:')
        evaluation_utils.print_eval_metrics(train_metrics)
        with open(savefile + '_cv_train_metrics.pkl', 'wb') as f:
            pickle.dump(train_metrics, f)

        print('\nCV test metrics:')
        evaluation_utils.print_eval_metrics(test_metrics)
        with open(savefile + '_cv_test_metrics.pkl', 'wb') as f:
            pickle.dump(test_metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', help='path to counts dataset', type=str)
    parser.add_argument('model_type', help='type of model: "linear","ann","cnn","logistic_linear","logistic_ann","logistic_cnn"', type=str)
    parser.add_argument('--pre_column', default='count_pre', help='column name in counts dataset', type=str)
    parser.add_argument('--post_column', default='count_post', help='column name in counts dataset', type=str)
    parser.add_argument("--encoding", default='is', help="sequence encoding to use: 'is', 'neighbors', or 'pairwise'")
    parser.add_argument("--n_hidden", default=3, help="number of hidden layers in nn model", type=int)
    parser.add_argument("--hidden_size", default=100, help="size of hidden layers in nn model", type=int)
    parser.add_argument("--window_size", default=8, help="size of 1D convolution window in cnn model", type=int)
    parser.add_argument("--residual_channels", default=16, help="number of filters in cnn residual connections", type=int)
    parser.add_argument("--skip_channels", default=16, help="number of filters in cnn skip connections", type=int)
    parser.add_argument("--learning_rate", default=1e-5, help="learning rate for gradient descent", type=float)
    parser.add_argument("--alpha", default=0., help="ell-2 regularization weight", type=float)
    parser.add_argument("--epochs", default=10, help="number of epochs to run training", type=int)
    parser.add_argument("--batch_size", default=128, help="number of samples per batch during training", type=int)
    parser.add_argument("--early_stopping", help="use early stopping during training", action='store_true')
    parser.add_argument("--normalize", help="normalize enrichment scores", action='store_true')
    parser.add_argument("--weighted_loss", help="use David's weighted loss function", action='store_true')
    parser.add_argument("--gradient_clip", help="max value for gradient clipping", type=float)
    parser.add_argument("--description", help="optional description to add to output filenames", type=str)
    parser.add_argument("--n_folds", default=3, help="number of folds to use for CV; pass n_folds=1 to train on full data.", type=int)
    args = parser.parse_args()
    main(args)
