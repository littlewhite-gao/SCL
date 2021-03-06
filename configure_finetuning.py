# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Config controlling hyperparameters for fine-tuning ELECTRA."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow.compat.v1 as tf


class FinetuningConfig(object):
    """Fine-tuning hyperparameters."""

    def __init__(self, pretrained_model, data_dir, epochs, output_dir, dropout_prob, **kwargs):
        # general
        self.pretrained_model = pretrained_model
        self.debug = False  # debug mode for quickly running things
        self.log_examples = True  # print out some train examples for debugging
        self.num_trials = 1  # how many train+eval runs to perform
        self.do_train = True  # train a model
        self.do_eval = True  # evaluate the model
        self.keep_all_models = True  # if False, only keep the last trial's ckpt
        self.keep_checkpoint_max = 3
        # model
        self.model_size = "base"  # one of "small", "base", or "large"
        self.task_names = ["match"]  # which tasks to learn
        # override the default transformer hparams for the provided model size; see
        # modeling.BertConfig for the possible hparams and util.training_utils for
        # the defaults
        self.model_hparam_overrides = (
            kwargs["model_hparam_overrides"]
            if "model_hparam_overrides" in kwargs else {})
        self.embedding_size = None  # bert hidden size by default
        self.vocab_size = 21128  # number of tokens in the vocabulary
        self.do_lower_case = True

        # training
        self.learning_rate = 1e-4
        self.weight_decay_rate = 0.01
        self.layerwise_lr_decay = 0.8
        self.num_train_epochs = epochs  # passes over the dataset during training
        self.warmup_proportion = 0.1  # how much of training to warm up the LR for
        self.save_checkpoints_steps = 5000
        self.iterations_per_loop = 1000
        self.use_tfrecords_if_existing = True
        self.hidden_dropout_prob = dropout_prob
        self.attention_probs_dropout_prob = dropout_prob


        # writing model outputs to disc
        self.write_test_outputs = True  # whether to write test set outputs,
                                         # currently supported for GLUE + SQuAD 2.0
        self.n_writes_test = 5  # write test set predictions for the first n trials

        # sizing
        self.max_seq_length = 512
        self.num_hidden_layers = 12
        self.train_batch_size = 4
        self.eval_batch_size = 8
        self.predict_batch_size = 8
        # for tasks like paraphrase where sentence order doesn't matter, train the model
        # on both sentence orderings for each example
        self.double_unordered = True

        # for qa tasks
        self.max_query_length = 64   # max tokens in q as opposed to context
        self.doc_stride = 128  # stride when splitting doc into multiple examples
        self.n_best_size = 20  # number of predictions per example to save
        self.max_answer_length = 30  # filter out answers longer than this length
        self.answerable_classifier = True  # answerable classifier for SQuAD 2.0
        self.answerable_uses_start_logits = True  # more advanced answerable
                                                  # classifier using predicted start
        self.answerable_weight = 0.5  # weight for answerability loss
        self.joint_prediction = True  # jointly predict the start and end positions
                                      # of the answer span
        self.beam_size = 20  # beam size when doing joint predictions
        self.qa_na_threshold = -2.75  # threshold for "no answer" when writing SQuAD
                                      # 2.0 test outputs
        # TPU settings
        self.use_tpu = False
        self.num_tpu_cores = 1
        self.tpu_job_name = None
        self.tpu_name = None  # cloud TPU to use for training
        self.tpu_zone = None  # GCE zone where the Cloud TPU is located in
        self.gcp_project = None  # project name for the Cloud TPU-enabled project

        # default locations of data files
        self.output_dir = output_dir
        self.data_dir = data_dir

        self.vocab_file = os.path.join(self.pretrained_model, "vocab.txt")

        self.update(kwargs)
        if not tf.io.gfile.exists(self.vocab_file):
            self.vocab_file = os.path.join(self.data_dir, "vocab.txt")
        self.task_names_str = ",".join(
            kwargs["task_names"] if "task_names" in kwargs else self.task_names)
        self.init_checkpoint = None if self.debug else self.pretrained_model

        qa_topdir = os.path.join(self.output_dir, self.task_names_str + "_qa")
        self.qa_eval_file = os.path.join(qa_topdir, "{:}_eval.json").format
        self.qa_preds_file = os.path.join(qa_topdir, "{:}_preds.json").format
        self.qa_na_file = os.path.join(qa_topdir, "{:}_null_odds.json").format
        tfrecords = "finetuning_tfrecords"
        self.preprocessed_data_dir = self.data_dir if self.use_tfrecords_if_existing else self.output_dir + '_1'
        self.preprocessed_data_dir = os.path.join(
            self.preprocessed_data_dir, tfrecords,
            self.task_names_str + "_tfrecords" + ("-debug" if self.debug else ""))
        from util import utils
        utils.heading('Tfrecords path:', self.preprocessed_data_dir)

        # update defaults with passed-in hyperparameters
        self.update(kwargs)

        # default hyperparameters for single-task models
        if len(self.task_names) == 1:
            task_name = self.task_names[0]
            if task_name == "rte":
                self.num_train_epochs = 10.0
            elif task_name == "sts":
                self.double_unordered = True
                self.num_train_epochs = 10.0
            elif "squad" in task_name or "qa" in task_name:
                self.max_seq_length = 512
                self.num_train_epochs = 2.0
                self.write_distill_outputs = False
                self.write_test_outputs = False
            elif task_name == "chunk":
                self.max_seq_length = 256
            else:
                self.num_train_epochs = epochs

        # default hyperparameters for different model sizes
        if self.model_size == "large":
            self.learning_rate = 5e-5
            self.layerwise_lr_decay = 0.9
            self.weight_decay_rate = 0.01
            self.num_hidden_layers = 24
        elif self.model_size == "small":
            self.embedding_size = 128

        # debug-mode settings
        if self.debug:
            self.save_checkpoints_steps = 1000000
            self.use_tfrecords_if_existing = False
            self.num_trials = 1
            self.iterations_per_loop = 1
            self.train_batch_size = 32
            self.num_train_epochs = epochs
            self.log_examples = True

        # passed-in-arguments override (for example) debug-mode defaults
        self.update(kwargs)

    def update(self, kwargs):
        for k, v in kwargs.items():
            if k not in self.__dict__:
                raise ValueError("Unknown hparam " + k)
            self.__dict__[k] = v
