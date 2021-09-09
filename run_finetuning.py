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

"""Fine-tunes an ELECTRA model on a downstream task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import collections
import json
import re
import os
import csv
import pickle
import codecs
import random
import numpy as np

import tensorflow.compat.v1 as tf

import configure_finetuning
from finetune import preprocessing
from finetune import task_builder
from model import modeling
from model import optimization
from util import training_utils
from util import utils
# from util import tsne
import arguments


args = arguments.parse_args()


class FinetuningModel(object):
    """
    Finetuning model with support for multi-task training.
    """

    def __init__(self, config: configure_finetuning.FinetuningConfig, tasks,
                 is_training, features, num_train_steps):
        # Create a shared transformer encoder
        bert_config = training_utils.get_bert_config(config)
        self.bert_config = bert_config
        if config.debug:
            bert_config.num_hidden_layers = 3
            bert_config.hidden_size = 144
            bert_config.intermediate_size = 144 * 4
            bert_config.num_attention_heads = 4
        assert config.max_seq_length <= bert_config.max_position_embeddings

        bert_model = modeling.BertModel(
            bert_config=bert_config,
            is_training=is_training,
            input_ids=features["input_ids"],
            input_mask=features["input_mask"],
            token_type_ids=features["segment_ids"],
            use_one_hot_embeddings=config.use_tpu,
            pooling=args.pooling,
            scope='electra',
            noise_scale=args.noise_scale,
            embedding_size=config.embedding_size,
            use_cl=False,
            c_type=args.c_type,
            drop_rate=args.scl_drop,
            cut_rate=args.cut_rate)
        if args.use_cl:
            utils.heading('------------------using contrastive learning-------------------')
            bert_model_cl = modeling.BertModel(
                bert_config=bert_config,
                is_training=is_training,
                input_ids=features["input_ids"],
                input_mask=features["input_mask"],
                token_type_ids=features["segment_ids"],
                use_one_hot_embeddings=config.use_tpu,
                pooling=args.pooling,
                scope='electra',
                noise_scale=args.noise_scale,
                embedding_size=config.embedding_size,
                use_cl=args.use_cl,
                c_type=args.c_type,
                drop_rate=args.scl_drop,
                cut_rate=args.cut_rate)

        percent_done = (tf.cast(tf.train.get_or_create_global_step(), tf.float32) /
                        tf.cast(num_train_steps, tf.float32))

        # Add specific tasks
        self.outputs = {"task_id": features["task_id"]}
        losses = []
        for task in tasks:
            with tf.variable_scope("task_specific/" + task.name):
                if task.name == 'sts':
                    task_losses, task_outputs, reprs = task.get_prediction_module(
                        bert_model, features, is_training, percent_done)
                    if args.use_cl:
                        utils.log("---------------------using cl---------------------")
                        task_losses_with_cl, task_outputs_with_cl, reprs_cl = task.get_prediction_module(
                            bert_model_cl, features, is_training, percent_done)
                        contrast_tensor = tf.concat([reprs, reprs_cl], axis=0)
                        targets = features[task.name + "_targets"]
                        scl_loss = task.sim_scl_loss
                        scl_losses = scl_loss(targets, contrast_tensor, args.tau)
                        losses.append(task_losses + args.alpha * scl_losses)
                    else:
                        losses.append(task_losses)
                    self.outputs[task.name] = task_outputs
                else:
                    task_losses, task_outputs, input_logits, reprs, probs = task.get_prediction_module(
                        bert_model, features, is_training, percent_done)
                    if args.use_cl:
                        utils.log("---------------------using cl---------------------")
                        task_losses_with_cl, task_outputs_with_cl, cl_logits, reprs_cl, probs_cl = task.get_prediction_module(
                            bert_model_cl, features, is_training, percent_done)
                        contrast_tensor = tf.concat([reprs, reprs_cl], axis=0)
                        global_step = tf.train.get_or_create_global_step()
                        global_step = tf.cast(global_step, dtype=tf.int32)
                        label_ids = features[task.name + "_label_ids"]
                        scl_losses = task.scl_loss(label_ids, contrast_tensor, args.tau)
                        losses.append(task_losses + args.alpha * scl_losses)
                    else:
                        losses.append(task_losses)
                    task_outputs.update({'reprs': reprs})
                    self.outputs[task.name] = task_outputs
        self.loss = tf.reduce_sum(
            tf.stack(losses, -1) *
            tf.one_hot(features["task_id"], len(config.task_names)))


def model_fn_builder(config: configure_finetuning.FinetuningConfig, tasks,
                     num_train_steps, pretraining_config=None):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):
        """The `model_fn` for TPUEstimator."""
        utils.log("Building model...")
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        model = FinetuningModel(
            config, tasks, is_training, features, num_train_steps)

        # Load pre-trained weights from checkpoint
        init_checkpoint = config.init_checkpoint
        # print(init_checkpoint)
        tvars = tf.trainable_variables()
        scaffold_fn = None
        assignment_map, initialized_variable_names = \
            modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        if config.use_tpu:
            def tpu_scaffold():
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                return tf.train.Scaffold()
            scaffold_fn = tpu_scaffold
        else:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        utils.log("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            utils.log(var.name, var.shape, init_string)

        # Build model for training or prediction
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                model.loss, config.learning_rate, num_train_steps,
                weight_decay_rate=args.weight_decay,
                use_tpu=config.use_tpu,
                warmup_proportion=args.warmup,
                layerwise_lr_decay_power=args.layerwise,
                n_transformer_layers=model.bert_config.num_hidden_layers)

            output_spec = tf.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=model.loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn,
                training_hooks=[training_utils.ETAHook(
                                {} if config.use_tpu else dict(loss=model.loss),
                                num_train_steps, config.iterations_per_loop, config.use_tpu, args.log_every)])

        else:
            assert mode == tf.estimator.ModeKeys.PREDICT
            output_spec = tf.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=utils.flatten_dict(model.outputs),
                scaffold_fn=scaffold_fn)

        utils.log("Building complete")
        return output_spec

    return model_fn


class ModelRunner(object):
    """Fine-tunes a model on a supervised task."""

    def __init__(self, config: configure_finetuning.FinetuningConfig, tasks,
                 pretraining_config=None):
        self._config = config
        self._tasks = tasks
        self._preprocessor = preprocessing.Preprocessor(config, self._tasks)

        is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
        tpu_cluster_resolver = None
        if config.use_tpu and config.tpu_name:
            tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
                config.tpu_name, zone=config.tpu_zone, project=config.gcp_project)
        tpu_config = tf.estimator.tpu.TPUConfig(
            iterations_per_loop=config.iterations_per_loop,
            num_shards=config.num_tpu_cores,
            per_host_input_for_training=is_per_host,
            tpu_job_name=config.tpu_job_name)
        run_config = tf.estimator.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=config.output_dir,
            save_checkpoints_steps=config.save_checkpoints_steps,
            save_checkpoints_secs=None,
            tpu_config=tpu_config)

        if self._config.do_train:
            (self._train_input_fn,
             self.train_steps) = self._preprocessor.prepare_train('train')
        else:
            self._train_input_fn, self.train_steps = None, 0
        model_fn = model_fn_builder(
            config=config,
            tasks=self._tasks,
            num_train_steps=self.train_steps,
            pretraining_config=pretraining_config)
        self._estimator = tf.estimator.tpu.TPUEstimator(
            use_tpu=config.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=config.train_batch_size,
            eval_batch_size=config.eval_batch_size,
            predict_batch_size=config.predict_batch_size,
            )

    def train(self):
        utils.log("Training for {:} steps".format(self.train_steps))
        self._estimator.train(
            input_fn=self._train_input_fn, max_steps=self.train_steps)

    def evaluate(self):
        return {task.name: self.evaluate_task(task) for task in self._tasks}

    def evaluate_task(self, task, split='dev', return_results=True):
        """Evaluate the current model."""
        utils.log("Evaluating", task.name)
        eval_input_fn, _ = self._preprocessor.prepare_predict([task], split)
        results = self._estimator.predict(input_fn=eval_input_fn,
                                          yield_single_examples=True)
        scorer = task.get_scorer()
        for r in results:
            if r["task_id"] != len(self._tasks):  # ignore padding examples
                r = utils.nest_dict(r, self._config.task_names)
                scorer.update(r[task.name])
        if return_results:
            res = task.name + ": " + scorer.results_str(split)
            utils.log(res)
            return dict(scorer.get_results())
        else:
            return scorer

    def write_classification_outputs(self, generic_output_dir, tasks, trial, split, label_map):
        """Write classification predictions to disk."""
        utils.log("Writing out predictions for", tasks, split)
        predict_input_fn, _ = self._preprocessor.prepare_predict(tasks, split)
        results = self._estimator.predict(input_fn=predict_input_fn,
                                          yield_single_examples=True)
        # task name -> eid -> model-logits
        logits = collections.defaultdict(dict)
        for r in results:
            if r["task_id"] != len(self._tasks):
                r = utils.nest_dict(r, self._config.task_names)
                task_name = self._config.task_names[r["task_id"]]
                logits[task_name][r[task_name]["eid"]] = (
                    r[task_name]["logits"] if "logits" in r[task_name]
                    else r[task_name]["predictions"])
        for task_name in logits:
            utils.log("Pickling predictions for {:} {:} examples ({:})".format(
                len(logits[task_name]), task_name, split))
            if trial <= self._config.n_writes_test:
                output_name = generic_output_dir[9:].replace('/', "_") + "_" + str(trial) + \
                              "_" + split + "_predictions.pkl"
                result_path = os.path.join(generic_output_dir + "_" + str(trial), output_name)
                utils.write_pickle(logits[task_name], result_path)
                from_pkl_to_csv(label_map, result_path, result_path.replace('.pkl', '.tsv'))


def write_cls_outputs(config: configure_finetuning.FinetuningConfig,
                      generic_output_dir, trial, split, result):
    # task name -> eid -> model-logits
    if config.task_names_str != 'sts':
        reprs = result[config.task_names_str]['reprs']
        logits = result[config.task_names_str]['logits']
        utils.log("Writing reprs, logits for %s, %s" % (config.task_names_str, split))
        output_name = generic_output_dir[9:].replace('/', "_") + "_" + str(trial) + "_" + split + "_reprs.csv"
        result_path = os.path.join(generic_output_dir + "_" + str(trial), output_name)
        logits_output_name = generic_output_dir[9:].replace('/', "_") + "_" + str(trial) + "_" + split + "_logits.csv"
        result_logits_path = os.path.join(generic_output_dir + "_" + str(trial), logits_output_name)
        with open(result_path, 'w') as w:
            w.write(str(reprs))
        with open(result_logits_path, 'w') as w:
            w.write(str(logits))


def write_results(results_txt, splits, results):
    """Write evaluation metrics to disk."""
    utils.log("Writing results to", results_txt)
    utils.mkdir(results_txt.rsplit("/", 1)[0])
    with tf.io.gfile.GFile(results_txt, "w") as f:
        results_str = ""
        for i, trial_results in enumerate(results):
            for task_name, task_results in trial_results.items():
                if task_name == "time" or task_name == "global_step":
                    continue
                results_str += task_name + "/" + splits[i % len(splits)] + ": " + " - ".join(
                    ["{:}: {:.2f}".format(k, v) if k not in ['probs', 'logits', 'reprs'] else ''
                     for k, v in task_results.items()]) + "\n"
        f.write(results_str)

def run_finetuning(config: configure_finetuning.FinetuningConfig):
    """Run finetuning."""
    # Setup for training
    results = []
    trial = 1

    heading = lambda msg: utils.heading(msg + ":")
    heading("Config")
    utils.log_config(config)
    generic_output_dir = config.output_dir
    tasks = task_builder.get_tasks(config)

    # Train and evaluate num_trials models with different random seeds
    while config.num_trials < 0 or trial <= config.num_trials:
        heading_info = "model={:}, trial {:}/{:}".format(
            config.pretrained_model, trial, config.num_trials)
        utils.heading(heading_info)
        config.output_dir = generic_output_dir + "_" + str(trial)
        utils.heading(config.output_dir)
        if config.do_train:
            utils.rmkdir(config.output_dir)

        model_runner = ModelRunner(config, tasks)
        if config.do_train:
            heading("Start training")
            model_runner.train()
            utils.log()

        if config.do_eval:
            heading("Run dev set evaluation")
            res = model_runner.evaluate()
            results.append(res)
            write_results(os.path.join(generic_output_dir + "_" + str(trial),
                                       config.task_names_str + "_dev_results.txt"), 'dev', results)
            write_cls_outputs(config, generic_output_dir, trial, 'dev', res)

            if config.write_test_outputs and trial <= config.n_writes_test:
                heading("Running on the test set and writing the predictions")
                for task in tasks:
                    # Currently only writing preds for GLUE and SQuAD 2.0 is supported

                    if task.name in ["cola", 'clean', "wnli", "mrpc", "mnli", "sst", "rte", "qnli", "qqp",
                                     "sts"]:
                        if task.name in ["rte", "qnli"]:
                            label_map_ = {0: "entailment", 1: "not_entailment"}
                        elif task.name == 'mnli':
                            label_map_ = {1: "entailment", 2: "neutral", 0: "contradiction"}
                        elif task.name == 'sts':
                            label_map_ = None
                        else:
                            label_map_ = {0: "0", 1: "1"}
                        for split in task.get_test_splits():
                            model_runner.write_classification_outputs(generic_output_dir,
                               [task], trial, split, label_map_)
                    elif task.name == "squad":
                        scorer = model_runner.evaluate_task(task, "test",  False)
                        scorer.write_predictions()
                        preds = utils.load_json(config.qa_preds_file("squad"))
                        null_odds = utils.load_json(config.qa_na_file("squad"))
                        for q, _ in preds.items():
                            if null_odds[q] > config.qa_na_threshold:
                              preds[q] = ""
                        utils.write_json(preds, config.test_predictions(
                            task.name, "test", trial))
                    else:
                        utils.log("Skipping task", task.name,
                                  "- writing predictions is not supported for this task")

        if trial != config.num_trials and (not config.keep_all_models):
            utils.rmrf(config.output_dir)
        trial += 1


def from_pkl_to_csv(label_map, pkl_file, csv_file):
    with codecs.open(csv_file, 'w', encoding='utf-8') as w:
        w.write('id' + '\t' + 'label' + '\n')
        with open(pkl_file, 'rb') as f:
            reader = pickle.load(f)
            for i, (key, value) in enumerate(reader.items()):
                if label_map is not None:
                    label = np.argmax(value, axis=-1)
                    label = label_map[label]
                    w.write(str(i) + '\t' + str(label) + '\n')
                else:
                    label = value
                    w.write(str(i) + '\t' + str(label) + '\n')


if __name__ == "__main__":
    if args.set_seed:
        assert args.seed
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)
        random.seed(args.seed)
    if args.hparams.endswith(".json"):
        hparams = utils.load_json(args.hparams)
    else:
        hparams = json.loads(args.hparams)
    tf.logging.set_verbosity(tf.logging.ERROR)
    run_finetuning(configure_finetuning.FinetuningConfig(
        args.electra_model, args.data_dir, args.epochs, args.output_dir, args.dropout, **hparams))


