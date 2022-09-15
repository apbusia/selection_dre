import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy.stats import pearsonr, spearmanr
import data_prep
from pre_process import AA_ORDER

tfk = tf.keras
tfkl = tf.keras.layers


def get_dataset(sequences, ids, encoding_fn, enrich_scores=None, counts=None, batch_size=1024, shuffle=True, shuffle_buffer=int(1e6), seed=None, flatten=True, tile=True):
    if enrich_scores is not None:
        return get_enrichment_dataset(sequences, enrich_scores, ids, encoding_fn, batch_size, shuffle, shuffle_buffer, seed, flatten)
    elif counts is not None:
        return get_classification_dataset(sequences, counts, ids, encoding_fn, batch_size, shuffle, shuffle_buffer, seed, flatten, tile)
    

def get_enrichment_dataset(sequences, enrich_scores, ids, encoding_fn, batch_size=1024, shuffle=True, shuffle_buffer=int(1e4), seed=None, flatten=True):
    """Returns a tf.data.Dataset that generates input/log-enrichment score data."""
    seq_len = len(sequences[0])
    X = np.stack(sequences.iloc[ids].apply(encoding_fn))
    enrich_scores = enrich_scores[ids]
    ds = tf.data.Dataset.from_tensor_slices((X, enrich_scores[:,0], enrich_scores[:,1]))
    if shuffle:
        ds = ds.shuffle(shuffle_buffer, seed=seed)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    onehot_depth = len(AA_ORDER)
    flat_shape = [-1, seq_len * onehot_depth]
    split_sizes = None
    if X.shape[1] > seq_len:
        pair_onehot_depth = onehot_depth ** 2
        split_sizes = [seq_len, X.shape[1] - seq_len]
        pair_flat_shape = [-1, split_sizes[1] * pair_onehot_depth]
    def tf_encoding_fn(x, e, v):
        if split_sizes is not None:
            x_is, x_pairs = tf.split(x, split_sizes, 1)
            x_is = tf.one_hot(x_is, onehot_depth)
            x_pairs = tf.one_hot(x_pairs, pair_onehot_depth)
            x = tf.concat([tf.reshape(x_is, flat_shape), tf.reshape(x_pairs, pair_flat_shape)], 1)
        else:
            x = tf.one_hot(x, onehot_depth)
            if flatten:
                x = tf.reshape(x, flat_shape)
        return x, e, v
    ds = ds.map(tf_encoding_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


def get_classification_dataset(sequences, counts, ids, encoding_fn, batch_size=1024, shuffle=True, shuffle_buffer=int(1e4), seed=None, flatten=True, tile=True):
    """Returns a tf.data.Dataset that generates input/label/count data."""
    seq_len = len(sequences[0])
    X = np.stack(sequences.iloc[ids].apply(encoding_fn))
    counts = counts[ids]
    pre_counts, post_counts = counts[:, 0], counts[:, 1]
    ds = tf.data.Dataset.from_tensor_slices((X, pre_counts, post_counts))
    if shuffle:
        ds = ds.shuffle(shuffle_buffer, seed=seed)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    onehot_depth = len(AA_ORDER)
    flat_shape = [-1, seq_len * onehot_depth]
    split_sizes = None
    if X.shape[1] > seq_len:
        pair_onehot_depth = onehot_depth ** 2
        split_sizes = [seq_len, X.shape[1] - seq_len]
        pair_flat_shape = [-1, split_sizes[1] * pair_onehot_depth]
    tile_dims = [2, 1] if flatten else [2, 1, 1]
    def tf_encoding_fn(x, pre, post):
        if split_sizes is not None:
            x_is, x_pairs = tf.split(x, split_sizes, 1)
            x_is = tf.one_hot(x_is, onehot_depth)
            x_pairs = tf.one_hot(x_pairs, pair_onehot_depth)
            x = tf.concat([tf.reshape(x_is, flat_shape), tf.reshape(x_pairs, pair_flat_shape)], 1)
        if split_sizes is None:
            x = tf.one_hot(x, onehot_depth)
            if flatten:
                x = tf.reshape(x, flat_shape)
        if tile:
            x = tf.tile(x, tile_dims)
        counts = tf.concat([post, pre], 0)
        classes = tf.concat([tf.ones_like(post), tf.zeros_like(pre)], 0)
        return x, classes, counts
    ds = ds.map(tf_encoding_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


def get_dataset_from_csv(csv_name, encoding, enrich_cols=None, count_cols=None, batch_size=1024, epochs=5, shuffle=True, shuffle_buffer=int(1e4), seed=None, flatten=True):
    if enrich_cols is not None:
        return get_enrichment_dataset_from_csv(csv_name, enrich_cols, encoding, batch_size, epochs, shuffle, shuffle_buffer, seed, flatten)
    elif count_cols is not None:
        return get_classification_dataset_from_csv(csv_name, count_cols, encoding, batch_size, epochs, shuffle, shuffle_buffer, seed, flatten)


def get_enrichment_dataset_from_csv(csv_name, enrich_cols, encoding, batch_size=1024, epochs=5, shuffle=True, shuffle_buffer=int(1e4), seed=None, flatten=True):
    enrich_col, var_col = enrich_cols[0], enrich_cols[1]
    cols = pd.read_csv(csv_name, nrows=1).columns.tolist()
    select_cols = enrich_cols + [c for c in cols if '{}_'.format(encoding) in c]
    ds = tf.data.experimental.make_csv_dataset(csv_name, batch_size, select_columns=select_cols, num_epochs=epochs, shuffle=shuffle, shuffle_buffer_size=shuffle_buffer, shuffle_seed=seed, prefetch_buffer_size=tf.data.AUTOTUNE, compression_type='GZIP')    
    onehot_depth = len(AA_ORDER)
    if encoding in ['pairwise', 'neighbors']:
        onehot_depth = onehot_depth ** 2
    flat_shape_fn = lambda s: [-1, s.shape[1] * onehot_depth]

    def tf_encoding_fn(t):
        columns = [k for k in t.keys() if '{}_'.format(encoding) in k]
        enc_seq = tf.stack([t[k] for k in t.keys() if '{}_'.format(encoding) in k], axis=1)
        if flatten:
            enc_seq = tf.reshape(tf.one_hot(enc_seq, onehot_depth), flat_shape_fn(enc_seq))
        else:
            enc_seq = tf.one_hot(enc_seq, onehot_depth)
        w = tf.ones_like(t[enrich_col]) if var_col is None else 1. / (2 * t[var_col])
        return (enc_seq, t[enrich_col], w)

    ds = ds.map(tf_encoding_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


def get_classification_dataset_from_csv(csv_name, count_cols, encoding, batch_size=1024, epochs=5, shuffle=True, shuffle_buffer=int(1e4), seed=None, flatten=True):
    pre_col, post_col = count_cols
    cols = pd.read_csv(csv_name, nrows=1).columns.tolist()
    select_cols = count_cols + [c for c in cols if '{}_'.format(encoding) in c]
    ds = tf.data.experimental.make_csv_dataset(csv_name, batch_size, select_columns=select_cols, num_epochs=epochs, shuffle=shuffle, shuffle_buffer_size=shuffle_buffer, shuffle_seed=seed, prefetch_buffer_size=tf.data.AUTOTUNE, compression_type='GZIP')    
    onehot_depth = len(AA_ORDER)
    if encoding in ['pairwise', 'neighbors']:
        onehot_depth = onehot_depth ** 2
    flat_shape_fn = lambda s: [-1, s.shape[1] * onehot_depth]

    def tf_encoding_fn(t):
        columns = [k for k in t.keys() if '{}_'.format(encoding) in k]
        enc_seq = tf.stack([t[k] for k in t.keys() if '{}_'.format(encoding) in k], axis=1)
        if flatten:
            enc_seq = tf.reshape(tf.one_hot(enc_seq, onehot_depth), flat_shape_fn(enc_seq))
            enc_seq = tf.tile(enc_seq, [2, 1])
        else:
            enc_seq = tf.one_hot(enc_seq, onehot_depth)
            enc_seq = tf.tile(enc_seq, [2, 1, 1])
        counts = tf.concat([t[post_col], t[pre_col]], 0)
        classes = tf.concat([tf.ones_like(t[post_col]), tf.zeros_like(t[pre_col])], 0)
        return (enc_seq, classes, counts)

    ds = ds.map(tf_encoding_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds
# 

# def get_classification_dataset_from_csv(csv_name, count_cols, encoding, batch_size=1024, shuffle=True, shuffle_buffer=int(1e4), seed=None, flatten=True):
#     cols = pd.read_csv(csv_name, nrows=1).columns.tolist()
#     select_cols = count_cols + [c for c in cols if '{}_'.format(encoding) in c]
#     ds = tf.data.experimental.CsvDataset(csv_name, [int()]*len(select_cols), header=True, select_cols=[cols.index(c) for c in select_cols])
#     if shuffle:
#         ds = ds.shuffle(shuffle_buffer, seed=seed)
#     ds = ds.batch(batch_size)
#     ds = ds.prefetch(tf.data.AUTOTUNE)
#     onehot_depth = len(AA_ORDER)
#     if encoding in ['pairwise', 'neighbors']:
#         onehot_depth = onehot_depth + onehot_depth ** 2
#     flat_shape_fn = lambda s: [-1, s.shape[1] * onehot_depth]
#     classes = tf.concat([tf.ones([batch_size]), tf.zeros(batch_size)], 0)
#     def tf_encoding_fn(*inputs):
#         enc_seq = tf.stack(inputs[2:], axis=1)
#         if flatten:
#             enc_seq = tf.reshape(tf.one_hot(enc_seq, onehot_depth), flat_shape_fn(enc_seq))
#             enc_seq = tf.tile(enc_seq, [2, 1])
#         else:
#             enc_seq = tf.one_hot(enc_seq, onehot_depth)
#             enc_seq = tf.tile(enc_seq, [2, 1, 1])
#         counts = tf.concat(inputs[:2], 0)
#         return (enc_seq, classes, counts)

#     ds = ds.map(tf_encoding_fn, num_parallel_calls=tf.data.AUTOTUNE)
#     return ds
    
    
def get_regularizer(l1_reg=0., l2_reg=0.):
    """
    Returns a keras regularizer object given 
    the l1 and l2 regularization parameters
    """
    if l1_reg > 0 and l2_reg > 0:
        reg = regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
    elif l1_reg > 0:
        reg = tfk.regularizers.l1(l1_reg)
    elif l2_reg > 0:
        reg = tfk.regularizers.l2(l2_reg)
    else:
        reg = None
    return reg

        
def make_linear_model(input_shape, lr=0.001, l1_reg=0., l2_reg=0., gradient_clip=None, epsilon=None, amsgrad=True):
    """
    Makes a linear keras model.
    """
    reg = get_regularizer(l1_reg, l2_reg)

    inp = tfkl.Input(shape=input_shape)
    output = tfkl.Dense(1, activation='linear', kernel_regularizer=reg, bias_regularizer=reg)(inp)
    model = tfk.models.Model(inputs=inp, outputs=output)
    model.compile(optimizer=tfk.optimizers.Adam(learning_rate=lr, epsilon=epsilon, clipvalue=gradient_clip, amsgrad=amsgrad),
                  loss=tfk.losses.MeanSquaredError(),
                  metrics=[tfk.metrics.MeanSquaredError()],
                  weighted_metrics=[tfk.metrics.MeanSquaredError()])
    return model


def make_linear_classifier(input_shape, lr=0.001, l1_reg=0., l2_reg=0., gradient_clip=None, epsilon=None, amsgrad=True):
    """
    Makes a logistic keras model.
    """
    reg = get_regularizer(l1_reg, l2_reg)

    inp = tfkl.Input(shape=input_shape)
    output = tfkl.Dense(2, activation='linear', kernel_regularizer=reg, bias_regularizer=reg)(inp)
    model = tfk.models.Model(inputs=inp, outputs=output)
    model.compile(optimizer=tfk.optimizers.Adam(learning_rate=lr, epsilon=epsilon, clipvalue=gradient_clip, amsgrad=amsgrad),
                  loss=tfk.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tfk.metrics.SparseCategoricalCrossentropy(from_logits=True)],
                  weighted_metrics=[tfk.metrics.SparseCategoricalCrossentropy(from_logits=True)])
    return model


def make_ann_model(input_shape, num_hid=2, hid_size=100, lr=0.001, l1_reg=0., l2_reg=0., gradient_clip=None, epsilon=None, amsgrad=True):
    """
    Builds an artificial neural network model for regression.
    """
    reg = get_regularizer(l1_reg, l2_reg)
    inp = tfkl.Input(shape=input_shape)
    z = inp
    for i in range(num_hid):
        z = tfkl.Dense(hid_size, activation='relu', kernel_regularizer=reg, bias_regularizer=reg)(z)
    out = tfkl.Dense(1, activation='linear', kernel_regularizer=reg, bias_regularizer=reg)(z)
    model = tfk.models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=tfk.optimizers.Adam(learning_rate=lr, epsilon=epsilon, clipvalue=gradient_clip, amsgrad=amsgrad),
                  loss=tfk.losses.MeanSquaredError(),
                  metrics=[tfk.metrics.MeanSquaredError()],
                  weighted_metrics=[tfk.metrics.MeanSquaredError()])
    return model


def make_ann_classifier(input_shape, num_hid=2, hid_size=100, lr=0.001, l1_reg=0., l2_reg=0., gradient_clip=None, epsilon=None, amsgrad=True):
    """
    Builds an artificial neural network model for classification.
    """
    reg = get_regularizer(l1_reg, l2_reg)
    inp = tfkl.Input(shape=input_shape)
    z = inp
    for i in range(num_hid):
        z = tfkl.Dense(hid_size, activation='relu', kernel_regularizer=reg, bias_regularizer=reg)(z)
    out = tfkl.Dense(2, activation='linear', kernel_regularizer=reg, bias_regularizer=reg)(z)
    model = tfk.models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=tfk.optimizers.Adam(learning_rate=lr, epsilon=epsilon, clipvalue=gradient_clip, amsgrad=amsgrad),
                  loss=tfk.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tfk.metrics.SparseCategoricalCrossentropy(from_logits=True)],
                  weighted_metrics=[tfk.metrics.SparseCategoricalCrossentropy(from_logits=True)])
    return model


def make_cnn_model(input_shape, num_hid=2, hid_size=100, win_size=2, residual_channels=16, skip_channels=16, padding='same', lr=0.001, l1_reg=0., l2_reg=0., gradient_clip=None, epsilon=None, amsgrad=True):
    """
    Builds a convolutional neural network model for regression.
    """
    reg = get_regularizer(l1_reg, l2_reg)
    variable_len_shape = (None, input_shape[-1])
    inp = tfkl.Input(shape=variable_len_shape)
    z = inp
    outputs = []
    
    # Pre-process the input with a 1D convolution.
    z = tfkl.Conv1D(residual_channels, win_size, padding=padding, dilation_rate=1, kernel_regularizer=reg, use_bias=False)(z)
    
    max_dilation_pow = 8
    dilations = [1 for i in range(num_hid)]
    for dilation in dilations:
        z_out = tfkl.Conv1D(
            hid_size, win_size, padding=padding, dilation_rate=dilation, kernel_regularizer=reg, use_bias=False, activation='relu')(z)
        z_skip = tfkl.Conv1D(skip_channels, 1, padding='same', kernel_regularizer=reg, bias_regularizer=reg)(z_out)
        z_res = tfkl.Conv1D(residual_channels, 1, padding='same', kernel_regularizer=reg, bias_regularizer=reg)(z_out)
        z = tfkl.Add()([z, z_res])
        outputs.append(z_skip)
    z = tfkl.Add()(outputs) if len(outputs) > 1 else outputs[0]
    z = tfkl.ReLU()(z)
    # To accomodate variable-length sequences, use global pooling instead of flatten.
#     z = tfkl.Flatten()(z)
#     z = tfkl.Conv1D(hid_size, 100, padding='valid', kernel_regularizer=reg, use_bias=False)(z)
    z = tfkl.GlobalMaxPool1D()(z)
    out = tfkl.Dense(1, activation='linear', kernel_regularizer=reg, bias_regularizer=reg)(z)
    model = tfk.models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=tfk.optimizers.Adam(learning_rate=lr, epsilon=epsilon, clipnorm=gradient_clip, amsgrad=amsgrad),
                  loss=tfk.losses.MeanSquaredError(),
                  metrics=[tfk.metrics.MeanSquaredError()],
                  weighted_metrics=[tfk.metrics.MeanSquaredError()])
    return model


def make_cnn_classifier(input_shape, num_hid=2, hid_size=100, win_size=2, residual_channels=16, skip_channels=16, padding='same', lr=0.001, l1_reg=0., l2_reg=0., gradient_clip=None, epsilon=None, amsgrad=True):
    """
    Builds a convolutional neural network model for classification.
    """
    reg = get_regularizer(l1_reg, l2_reg)
    variable_len_shape = (None, input_shape[-1])
    inp = tfkl.Input(shape=variable_len_shape)
    z = inp
    outputs = []
    
    # Pre-process the input with a 1D convolution.
    z = tfkl.Conv1D(residual_channels, win_size, padding=padding, dilation_rate=1, kernel_regularizer=reg, use_bias=False)(z)
    
    max_dilation_pow = 8
    dilations = [1 for i in range(num_hid)]
    for dilation in dilations:
        z_out = tfkl.Conv1D(
            hid_size, win_size, padding=padding, dilation_rate=dilation, kernel_regularizer=reg, use_bias=False, activation='relu')(z)
        z_skip = tfkl.Conv1D(skip_channels, 1, padding='same', kernel_regularizer=reg, bias_regularizer=reg)(z_out)
        z_res = tfkl.Conv1D(residual_channels, 1, padding='same', kernel_regularizer=reg, bias_regularizer=reg)(z_out)
        z = tfkl.Add()([z, z_res])
        outputs.append(z_skip)
    z = tfkl.Add()(outputs) if len(outputs) > 1 else outputs[0]
    z = tfkl.ReLU()(z)
    # To accomodate variable-length sequences, use global pooling instead of flatten.
#     z = tfkl.Flatten()(z)
#     z = tfkl.Conv1D(hid_size, 100, padding='valid', kernel_regularizer=reg, use_bias=False)(z)
    z = tfkl.GlobalMaxPool1D()(z)
    out = tfkl.Dense(2, activation='linear', kernel_regularizer=reg, bias_regularizer=reg)(z)
    model = tfk.models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=tfk.optimizers.Adam(learning_rate=lr, epsilon=epsilon, clipnorm=gradient_clip, amsgrad=amsgrad),
                  loss=tfk.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tfk.metrics.SparseCategoricalCrossentropy(from_logits=True)],
                  weighted_metrics=[tfk.metrics.SparseCategoricalCrossentropy(from_logits=True)])
    return model


def calculate_receptive_field(win_size, dilations):
    return (win_size - 1) * (np.sum(dilations) + 1) + 1
