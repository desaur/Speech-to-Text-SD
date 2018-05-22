#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import sys

log_level_index = sys.argv.index('--log_level') + 1 if '--log_level' in sys.argv else 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = sys.argv[log_level_index] if log_level_index > 0 and log_level_index < len(sys.argv) else '3'

import datetime
import pickle
import shutil
import six
import subprocess
import tensorflow as tf
import time
import traceback
import inspect

from six.moves import zip, range, filter, urllib, BaseHTTPServer
from tensorflow.python.tools import freeze_graph
from threading import Thread, Lock
from util.audio import audiofile_to_input_vector
from util.feeding import DataSet, ModelFeeder
from util.gpu import get_available_gpus
from util.shared_lib import check_cupti
from util.text import sparse_tensor_value_to_texts, wer, levenshtein, Alphabet, ndarray_to_text
from xdg import BaseDirectory as xdg
import numpy as np


# Importer
# ========

tf.app.flags.DEFINE_string  ('train_files',      '',          'comma separated list of files specifying the dataset used for training. multiple files will get merged')
tf.app.flags.DEFINE_string  ('dev_files',        '',          'comma separated list of files specifying the dataset used for validation. multiple files will get merged')
tf.app.flags.DEFINE_string  ('test_files',       '',          'comma separated list of files specifying the dataset used for testing. multiple files will get merged')
tf.app.flags.DEFINE_boolean ('fulltrace',        False,       'if full trace debug info should be generated during training')

# Cluster configuration
# =====================

tf.app.flags.DEFINE_string  ('ps_hosts',         '',          'parameter servers - comma separated list of hostname:port pairs')
tf.app.flags.DEFINE_string  ('worker_hosts',     '',          'workers - comma separated list of hostname:port pairs')
tf.app.flags.DEFINE_string  ('job_name',         'localhost', 'job name - one of localhost (default), worker, ps')
tf.app.flags.DEFINE_integer ('task_index',       0,           'index of task within the job - worker with index 0 will be the chief')
tf.app.flags.DEFINE_integer ('replicas',         -1,          'total number of replicas - if negative, its absolute value is multiplied by the number of workers')
tf.app.flags.DEFINE_integer ('replicas_to_agg',  -1,          'number of replicas to aggregate - if negative, its absolute value is multiplied by the number of workers')
tf.app.flags.DEFINE_integer ('coord_retries',    100,         'number of tries of workers connecting to training coordinator before failing')
tf.app.flags.DEFINE_string  ('coord_host',       'localhost', 'coordination server host')
tf.app.flags.DEFINE_integer ('coord_port',       2500,        'coordination server port')
tf.app.flags.DEFINE_integer ('iters_per_worker', 1,           'number of train or inference iterations per worker before results are sent back to coordinator')

# Global Constants
# ================

tf.app.flags.DEFINE_boolean ('train',            True,        'whether to train the network')
tf.app.flags.DEFINE_boolean ('test',             True,        'whether to test the network')
tf.app.flags.DEFINE_integer ('epoch',            75,          'target epoch to train - if negative, the absolute number of additional epochs will be trained')

tf.app.flags.DEFINE_boolean ('use_warpctc',      False,       'whether to use GPU bound Warp-CTC')

tf.app.flags.DEFINE_float   ('dropout_rate',     0.05,        'dropout rate for feedforward layers')
tf.app.flags.DEFINE_float   ('dropout_rate2',    -1.0,        'dropout rate for layer 2 - defaults to dropout_rate')
tf.app.flags.DEFINE_float   ('dropout_rate3',    -1.0,        'dropout rate for layer 3 - defaults to dropout_rate')
tf.app.flags.DEFINE_float   ('dropout_rate4',    0.0,         'dropout rate for layer 4 - defaults to 0.0')
tf.app.flags.DEFINE_float   ('dropout_rate5',    0.0,         'dropout rate for layer 5 - defaults to 0.0')
tf.app.flags.DEFINE_float   ('dropout_rate6',    -1.0,        'dropout rate for layer 6 - defaults to dropout_rate')

tf.app.flags.DEFINE_float   ('relu_clip',        20.0,        'ReLU clipping value for non-recurrant layers')

# Adam optimizer (http://arxiv.org/abs/1412.6980) parameters

tf.app.flags.DEFINE_float   ('beta1',            0.9,         'beta 1 parameter of Adam optimizer')
tf.app.flags.DEFINE_float   ('beta2',            0.999,       'beta 2 parameter of Adam optimizer')
tf.app.flags.DEFINE_float   ('epsilon',          1e-8,        'epsilon parameter of Adam optimizer')
tf.app.flags.DEFINE_float   ('learning_rate',    0.001,       'learning rate of Adam optimizer')

# Batch sizes

tf.app.flags.DEFINE_integer ('train_batch_size', 1,           'number of elements in a training batch')
tf.app.flags.DEFINE_integer ('dev_batch_size',   1,           'number of elements in a validation batch')
tf.app.flags.DEFINE_integer ('test_batch_size',  1,           'number of elements in a test batch')

# Sample limits

tf.app.flags.DEFINE_integer ('limit_train',      0,           'maximum number of elements to use from train set - 0 means no limit')
tf.app.flags.DEFINE_integer ('limit_dev',        0,           'maximum number of elements to use from validation set- 0 means no limit')
tf.app.flags.DEFINE_integer ('limit_test',       0,           'maximum number of elements to use from test set- 0 means no limit')

# Step widths

tf.app.flags.DEFINE_integer ('display_step',     0,           'number of epochs we cycle through before displaying detailed progress - 0 means no progress display')
tf.app.flags.DEFINE_integer ('validation_step',  0,           'number of epochs we cycle through before validating the model - a detailed progress report is dependent on "--display_step" - 0 means no validation steps')

# Checkpointing

tf.app.flags.DEFINE_string  ('checkpoint_dir',   '',          'directory in which checkpoints are stored - defaults to directory "deepspeech/checkpoints" within user\'s data home specified by the XDG Base Directory Specification')
tf.app.flags.DEFINE_integer ('checkpoint_secs',  600,         'checkpoint saving interval in seconds')
tf.app.flags.DEFINE_integer ('max_to_keep',      5,           'number of checkpoint files to keep - default value is 5')

# Exporting

tf.app.flags.DEFINE_string  ('export_dir',       '',          'directory in which exported models are stored - if omitted, the model won\'t get exported')
tf.app.flags.DEFINE_integer ('export_version',   1,           'version number of the exported model')
tf.app.flags.DEFINE_boolean ('remove_export',    False,       'whether to remove old exported models')
tf.app.flags.DEFINE_boolean ('use_seq_length',   True,        'have sequence_length in the exported graph (will make tfcompile unhappy)')

# Reporting

tf.app.flags.DEFINE_integer ('log_level',        1,           'log level for console logs - 0: INFO, 1: WARN, 2: ERROR, 3: FATAL')
tf.app.flags.DEFINE_boolean ('log_traffic',      False,       'log cluster transaction and traffic information during debug logging')

tf.app.flags.DEFINE_string  ('wer_log_pattern',  '',          'pattern for machine readable global logging of WER progress; has to contain %%s, %%s and %%f for the set name, the date and the float respectively; example: "GLOBAL LOG: logwer(\'12ade231\', %%s, %%s, %%f)" would result in some entry like "GLOBAL LOG: logwer(\'12ade231\', \'train\', \'2017-05-18T03:09:48-0700\', 0.05)"; if omitted (default), there will be no logging')

tf.app.flags.DEFINE_boolean ('log_placement',    False,       'whether to log device placement of the operators to the console')
tf.app.flags.DEFINE_integer ('report_count',     10,          'number of phrases with lowest WER (best matching) to print out during a WER report')

tf.app.flags.DEFINE_string  ('summary_dir',      '',          'target directory for TensorBoard summaries - defaults to directory "deepspeech/summaries" within user\'s data home specified by the XDG Base Directory Specification')
tf.app.flags.DEFINE_integer ('summary_secs',     0,           'interval in seconds for saving TensorBoard summaries - if 0, no summaries will be written')

# Geometry

tf.app.flags.DEFINE_integer ('n_hidden',         2048,        'layer width to use when initialising layers')

# Initialization

tf.app.flags.DEFINE_integer ('random_seed',      4567,        'default random seed that is used to initialize variables')
tf.app.flags.DEFINE_float   ('default_stddev',   0.046875,    'default standard deviation to use when initialising weights and biases')

# Early Stopping

tf.app.flags.DEFINE_boolean ('early_stop',       True,        'enable early stopping mechanism over validation dataset. Make sure that dev FLAG is enabled for this to work')

# This parameter is irrespective of the time taken by single epoch to complete and checkpoint saving intervals.
# It is possible that early stopping is triggered far after the best checkpoint is already replaced by checkpoint saving interval mechanism.
# One has to align the parameters (earlystop_nsteps, checkpoint_secs) accordingly as per the time taken by an epoch on different datasets.

tf.app.flags.DEFINE_integer ('earlystop_nsteps',  4,          'number of steps to consider for early stopping. Loss is not stored in the checkpoint so when checkpoint is revived it starts the loss calculation from start at that point')
tf.app.flags.DEFINE_float   ('estop_mean_thresh', 0.5,        'mean threshold for loss to determine the condition if early stopping is required')
tf.app.flags.DEFINE_float   ('estop_std_thresh',  0.5,        'standard deviation threshold for loss to determine the condition if early stopping is required')

# Decoder

tf.app.flags.DEFINE_string  ('decoder_library_path', 'native_client/libctc_decoder_with_kenlm.so', 'path to the libctc_decoder_with_kenlm.so library containing the decoder implementation.')
tf.app.flags.DEFINE_string  ('alphabet_config_path', 'data/alphabet.txt', 'path to the configuration file specifying the alphabet used by the network. See the comment in data/alphabet.txt for a description of the format.')
tf.app.flags.DEFINE_string  ('lm_binary_path',       'data/lm/lm.binary', 'path to the language model binary file created with KenLM')
tf.app.flags.DEFINE_string  ('lm_trie_path',         'data/lm/trie', 'path to the language model trie file created with native_client/generate_trie')
tf.app.flags.DEFINE_integer ('beam_width',        1024,       'beam width used in the CTC decoder when building candidate transcriptions')
tf.app.flags.DEFINE_float   ('lm_weight',         1.75,       'the alpha hyperparameter of the CTC decoder. Language Model weight.')
tf.app.flags.DEFINE_float   ('word_count_weight', 1.00,      'the beta hyperparameter of the CTC decoder. Word insertion weight (penalty).')
tf.app.flags.DEFINE_float   ('valid_word_count_weight', 1.00, 'valid word insertion weight. This is used to lessen the word insertion penalty when the inserted word is part of the vocabulary.')

# Inference mode

tf.app.flags.DEFINE_string  ('one_shot_infer',       '',       'one-shot inference mode: specify a wav file and the script will load the checkpoint and perform inference on it. Disables training, testing and exporting.')

# Initialize from frozen model

tf.app.flags.DEFINE_string  ('initialize_from_frozen_model', '', 'path to frozen model to initialize from. This behaves like a checkpoint, loading the weights from the frozen model and starting training with those weights. The optimizer parameters aren\'t restored, so remember to adjust the learning rate.')

for var in ['b1', 'h1', 'b2', 'h2', 'b3', 'h3', 'b5', 'h5', 'b6', 'h6']:
    tf.app.flags.DEFINE_float('%s_stddev' % var, None, 'standard deviation to use when initialising %s' % var)

FLAGS = tf.app.flags.FLAGS

def initialize_globals():

    global alphabet
    alphabet = Alphabet(os.path.abspath(FLAGS.alphabet_config_path))

    # Geometric Constants
    # ===================

    # For an explanation of the meaning of the geometric constants, please refer to
    # doc/Geometry.md

    # Number of MFCC features
    global n_input
    n_input = 26 # TODO: Determine this programatically from the sample rate

    # The number of frames in the context
    global n_context
    n_context = 9 # TODO: Determine the optimal value using a validation data set

    if len(FLAGS.one_shot_infer) > 0:
        FLAGS.train = False
        FLAGS.test = False
        FLAGS.export_dir = ''
        if not os.path.exists(FLAGS.one_shot_infer):
            log_error('Path specified in --one_shot_infer is not a valid file.')
            exit(1)

    if not os.path.exists(os.path.abspath(FLAGS.decoder_library_path)):
        print('ERROR: The decoder library file does not exist. Make sure you have ' \
              'downloaded or built the native client binaries and pass the ' \
              'appropriate path to the binaries in the --decoder_library_path parameter.')

    global custom_op_module
    custom_op_module = tf.load_op_library(FLAGS.decoder_library_path)


# Helpers
# =======

def calculate_report(results_tuple):
    r'''
    This routine will calculate a WER report.
    It'll compute the `mean` WER and create ``Sample`` objects of the ``report_count`` top lowest
    loss items from the provided WER results tuple (only items with WER!=0 and ordered by their WER).
    '''
    samples = []
    items = list(zip(*results_tuple))
    total_levenshtein = 0.0
    total_label_length = 0.0
    for label, decoding, distance, loss in items:
        sample_wer = wer(label, decoding)
        sample = Sample(label, decoding, loss, distance, sample_wer)
        samples.append(sample)
        total_levenshtein += levenshtein(label.split(), decoding.split())
        total_label_length += float(len(label.split()))

    # Getting the WER from the accumulated levenshteins and lengths
    samples_wer = total_levenshtein / total_label_length

    # Filter out all items with WER=0
    samples = [s for s in samples if s.wer > 0]

    # Order the remaining items by their loss (lowest loss on top)
    samples.sort(key=lambda s: s.loss)

    # Take only the first report_count items
    samples = samples[:FLAGS.report_count]

    # Order this top FLAGS.report_count items by their WER (lowest WER on top)
    samples.sort(key=lambda s: s.wer)

    return samples_wer, samples

def collect_results(results_tuple, returns):
    r'''
    This routine will help collecting partial results for the WER reports.
    The ``results_tuple`` is composed of an array of the original labels,
    an array of the corresponding decodings, an array of the corrsponding
    distances and an array of the corresponding losses. ``returns`` is built up
    in a similar way, containing just the unprocessed results of one
    ``session.run`` call (effectively of one batch).
    Labels and decodings are converted to text before splicing them into their
    corresponding results_tuple lists. In the case of decodings,
    for now we just pick the first available path.
    '''
    # Each of the arrays within results_tuple will get extended by a batch of each available device
    for i in range(len(available_devices)):
        # Collect the labels
        results_tuple[0].extend(sparse_tensor_value_to_texts(returns[0][i], alphabet))

        # Collect the decodings - at the moment we default to the first one
        results_tuple[1].extend(sparse_tensor_value_to_texts(returns[1][i][0], alphabet))

        # Collect the distances
        results_tuple[2].extend(returns[2][i])

        # Collect the losses
        results_tuple[3].extend(returns[3][i])


# For reporting we also need a standard way to do time measurements.
def stopwatch(start_duration=0):
    r'''
    This function will toggle a stopwatch.
    The first call starts it, second call stops it, third call continues it etc.
    So if you want to measure the accumulated time spent in a certain area of the code,
    you can surround that code by stopwatch-calls like this:

    .. code:: python

        fun_time = 0 # initializes a stopwatch
        [...]
        for i in range(10):
          [...]
          # Starts/continues the stopwatch - fun_time is now a point in time (again)
          fun_time = stopwatch(fun_time)
          fun()
          # Pauses the stopwatch - fun_time is now a duration
          fun_time = stopwatch(fun_time)
        [...]
        # The following line only makes sense after an even call of :code:`fun_time = stopwatch(fun_time)`.
        print 'Time spent in fun():', format_duration(fun_time)

    '''
    if start_duration == 0:
        return datetime.datetime.utcnow()
    else:
        return datetime.datetime.utcnow() - start_duration

def format_duration(duration):
    '''Formats the result of an even stopwatch call as hours:minutes:seconds'''
    duration = duration if isinstance(duration, int) else duration.seconds
    m, s = divmod(duration, 60)
    h, m = divmod(m, 60)
    return '%d:%02d:%02d' % (h, m, s)


"""
def create_inference_graph(batch_size=None, use_new_decoder=False):
    # Input tensor will be of shape [batch_size, n_steps, n_input + 2*n_input*n_context]
    input_tensor = tf.placeholder(tf.float32, [batch_size, None, n_input + 2*n_input*n_context], name='input_node')
    seq_length = tf.placeholder(tf.int32, [batch_size], name='input_lengths')

    # Calculate the logits of the batch using BiRNN
    logits = BiRNN(input_tensor, tf.to_int64(seq_length) if FLAGS.use_seq_length else None, no_dropout)

    # Beam search decode the batch
    decoder = decode_with_lm if use_new_decoder else tf.nn.ctc_beam_search_decoder

    decoded, _ = decoder(logits, seq_length, merge_repeated=False, beam_width=FLAGS.beam_width)
    decoded = tf.convert_to_tensor(
        [tf.sparse_tensor_to_dense(sparse_tensor) for sparse_tensor in decoded], name='output_node')

    return (
        {
            'input': input_tensor,
            'input_lengths': seq_length,
        },
        {
            'outputs': decoded,
        }
    )
"""

"""
def do_single_file_inference(input_file_path):
    with tf.Session(config=session_config) as session:
        inputs, outputs = create_inference_graph(batch_size=1, use_new_decoder=True)

        # Create a saver using variables from the above newly created graph
        saver = tf.train.Saver(tf.global_variables())

        # Restore variables from training checkpoint
        # TODO: This restores the most recent checkpoint, but if we use validation to counterract
        #       over-fitting, we may want to restore an earlier checkpoint.
        checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if not checkpoint:
            log_error('Checkpoint directory ({}) does not contain a valid checkpoint state.'.format(FLAGS.checkpoint_dir))
            exit(1)

        checkpoint_path = checkpoint.model_checkpoint_path
        saver.restore(session, checkpoint_path)

        mfcc = audiofile_to_input_vector(input_file_path, n_input, n_context)

        output = session.run(outputs['outputs'], feed_dict = {
            inputs['input']: [mfcc],
            inputs['input_lengths']: [len(mfcc)],
        })

        text = ndarray_to_text(output[0][0], alphabet)

        print(text)
"""

def main(_) :

    start = stopwatch()
    initialize_globals()

    if len(FLAGS.one_shot_infer):
        #load the frozen graph as in train(...) or as in https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc 
        with tf.gfile.FastGFile("../../models/output_graph.pb", 'rb') as fin:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fin.read()) 
        with tf.Graph().as_default() as pretrained_model:
            tf.import_graph_def(graph_def, name="pretrained_")
        """
        for op in pretrained_model.get_operations():
            print(op.name)
        """

#        print("------------***-------------")

# https://stackoverflow.com/questions/36883949/in-tensorflow-get-the-names-of-all-the-tensors-in-a-graph?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
        lstTensors = [op.values() for op in pretrained_model.get_operations()]
        input_node = lstTensors[0]
        input_lengths = lstTensors[1]
        output_node = lstTensors[-1]
        """
        print("input node name: ")
        print(input_node[0].name)
        print("input node shape: ")
        print(input_node[0].shape)#V IMP: shape of input node is [x,y,z] where x = batch_size. For one shot infer, batch_size = 1
        print("input lengths name: ")
        print(input_lengths[0].name)
        print("input lengths shape: ")
        print(input_lengths[0].shape) #V IMP: shape of input_lengths node is [x,y] where x = batch_size. For one shot infer, batch_size = 1
        print("output node name: ")
        print(output_node[0].name)
        print("output node shape: ")
        print(output_node[0].shape)
        """

#        do_single_file_inference(FLAGS.one_shot_infer)
#        print("n_input = "+repr(n_input))
#        print("n_context = "+repr(n_context))
        mfcc = audiofile_to_input_vector(FLAGS.one_shot_infer, n_input, n_context)
#        print(mfcc.shape)
        
#        output_node = pretrained_model.get_tensor_by_name(pretrained_model.get_operations()[-1].name)

        batch_size = 1
        with tf.Session(graph=pretrained_model) as sess:
            output = sess.run(output_node, feed_dict = {
                input_node: [mfcc.reshape((batch_size, mfcc.shape[0], mfcc.shape[1]))],
                input_lengths: [np.array(len(mfcc)).reshape((batch_size, ))]
            })
#            print(output)
            text = ndarray_to_text(output[0][0][0], alphabet)
            print("\n\nResult:")    
            print(text)
    else:
        print("Correct usage: python3 _this.py --one_shot_infer <<path-of-input-wav-file>>")

    delta = stopwatch(start)
    print("Net execution time including loading of the graph = "+format_duration(delta))

if __name__ == '__main__' :
    tf.app.run()

