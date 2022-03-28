import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy.stats import pearsonr, spearmanr
import data_prep
from pre_process import AA_ORDER

tfk = tf.keras
tfkl = tf.keras.layers


def get_dataset(sequences, ids, encoding_fn, enrich_scores=None, classes=None, counts=None, batch_size=1024, shuffle=True, shuffle_buffer=int(1e6), seed=None):
    if enrich_scores is not None:
        return get_enrichment_dataset(sequences, enrich_scores, ids, encoding_fn, batch_size, shuffle, shuffle_buffer, seed)
    elif classes is not None:
        return get_classification_dataset(sequences, classes, counts, ids, encoding_fn, batch_size, shuffle, shuffle_buffer, seed)
    

def get_enrichment_dataset(sequences, enrich_scores, ids, encoding_fn, batch_size=1024, shuffle=True, shuffle_buffer=int(1e4), seed=None):
    """Returns a tf.data.Dataset that generates input/log-enrichment score data."""
    X = np.stack(sequences.iloc[ids].apply(encoding_fn))
    enrich_scores = enrich_scores[ids]
    ds = tf.data.Dataset.from_tensor_slices((X, enrich_scores[:,0], enrich_scores[:,1]))
    if shuffle:
        ds = ds.shuffle(shuffle_buffer, seed=seed)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    onehot_depth = len(AA_ORDER)
    if X.shape[1] > len(sequences[0]):
        onehot_depth = onehot_depth + onehot_depth ** 2
    flat_shape = [-1, X.shape[1] * onehot_depth]
    onehot_fn = lambda x, y_0, y_1: (tf.reshape(tf.one_hot(x, onehot_depth), flat_shape), y_0, y_1)
    ds = ds.map(onehot_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


def get_classification_dataset(sequences, classes, counts, ids, encoding_fn, batch_size=1024, shuffle=True, shuffle_buffer=int(1e4), seed=None):
    """Returns a tf.data.Dataset that generates input/label/count data."""
    X = np.stack(sequences.iloc[ids].apply(encoding_fn))
    classes = classes[ids]
    counts = counts[ids]
    ds = tf.data.Dataset.from_tensor_slices((X, classes, counts))
    if shuffle:
        ds = ds.shuffle(shuffle_buffer, seed=seed)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    onehot_depth = len(AA_ORDER)
    if X.shape[1] > len(sequences[0]):
        onehot_depth = onehot_depth + onehot_depth ** 2
    flat_shape = [-1, X.shape[1] * onehot_depth]
    onehot_fn = lambda x, y_0, y_1: (tf.reshape(tf.one_hot(x, onehot_depth), flat_shape), y_0, y_1)
    ds = ds.map(onehot_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds
    
    
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

        
def make_linear_model(input_shape, lr=0.001, l1_reg=0., l2_reg=0., gradient_clip=None):
    """
    Makes a linear keras model.
    """
    reg = get_regularizer(l1_reg, l2_reg)

    inp = tfkl.Input(shape=input_shape)
    output = tfkl.Dense(1, activation='linear', kernel_regularizer=reg, bias_regularizer=reg)(inp)
    model = tfk.models.Model(inputs=inp, outputs=output)
    model.compile(optimizer=tfk.optimizers.Adam(learning_rate=lr, clipvalue=gradient_clip),
                  loss=tfk.losses.MeanSquaredError(),
                  metrics=[tfk.metrics.MeanSquaredError()],
                  weighted_metrics=[tfk.metrics.MeanSquaredError()])
    return model


def make_linear_classifier(input_shape, lr=0.001, l1_reg=0., l2_reg=0., gradient_clip=None):
    """
    Makes a logistic keras model.
    """
    reg = get_regularizer(l1_reg, l2_reg)

    inp = tfkl.Input(shape=input_shape)
    output = tfkl.Dense(2, activation='linear', kernel_regularizer=reg, bias_regularizer=reg)(inp)
    model = tfk.models.Model(inputs=inp, outputs=output)
    model.compile(optimizer=tfk.optimizers.Adam(learning_rate=lr, clipvalue=gradient_clip),
                  loss=tfk.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tfk.metrics.SparseCategoricalCrossentropy(from_logits=True)],
                  weighted_metrics=[tfk.metrics.SparseCategoricalCrossentropy(from_logits=True)])
    return model


def make_ann_model(input_shape, num_hid=2, hid_size=100, lr=0.001, l1_reg=0., l2_reg=0., gradient_clip=None):
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
    model.compile(optimizer=tfk.optimizers.Adam(learning_rate=lr, clipvalue=gradient_clip),
                  loss=tfk.losses.MeanSquaredError(),
                  metrics=[tfk.metrics.MeanSquaredError()],
                  weighted_metrics=[tfk.metrics.MeanSquaredError()])
    return model


def make_ann_classifier(input_shape, num_hid=2, hid_size=100, lr=0.001, l1_reg=0., l2_reg=0., gradient_clip=None):
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
    model.compile(optimizer=tfk.optimizers.Adam(learning_rate=lr, clipvalue=gradient_clip),
                  loss=tfk.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tfk.metrics.SparseCategoricalCrossentropy(from_logits=True)],
                  weighted_metrics=[tfk.metrics.SparseCategoricalCrossentropy(from_logits=True)])
    return model
