from __future__ import print_function

import os
import pickle
import logging
import time
import shutil
import sys
import select
import math
import cv2
from PIL import ImageFont, ImageDraw, Image  

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python import debug as tf_debug
from debugger.custom_filters import add_filters
from utils.helpers import *
from utils.data_utils import *
from utils.image_utils import *
from utils.accuracy_metrics import *
from models.lm import ctc_prefix_beam_search_decoder

plt.switch_backend('agg')
logging.basicConfig(level=logging.INFO)
tf.logging.set_verbosity(tf.logging.ERROR)

SPLIT_RATIO = 0.80
TIMEOUT = 3

class Model():
    def __init__(self, config, eval_path=None, infer=False):
        tf.reset_default_graph()    # In case some tensorflow graph is already loaded
        self.config = config
        
        self.TRAIN = eval_path is None and not infer
        self.EVALUATE = eval_path is not None and not infer
        self.INFER = infer and eval_path is None

        if self.TRAIN:
            save_dir = os.path.join(config.save_dir, config.model)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            self._load_training_data()

        if self.EVALUATE:
            self._load_evaluation_data(eval_path)

        self._handle_constants()
        self._initialize_graph()

        if self.INFER:
            self._initialize_model(True, verbose=False)

    def __call__(self, path=None): 
        if self.TRAIN:
            logging.info("-"*50 + "\nStarting Training\n" + "-"*50)
            train_loss_history, val_loss_history, grad_norm_history = self._fit()
            _plot_results(train_loss_history, val_loss_history, grad_norm_history, self.config.save_dir)

        elif self.EVALUATE:
            self._evaluate()

        elif self.INFER:
            self._load_inferring_data(path)
            out = self._infer()
            urdu_out = [convert_to_urdu(o, self.config.data_folder)[1] for o in out]

        return (out, urdu_out) if self.INFER else None

    def _load_training_data(self):
        config = self.config

        X, X_seq_len, y, y_seq_len = self._load_data_from_folder(config.data_folder, size=config.num_train)
        self.vocab_size = int(max([max(yij) for yi in y for yij in yi]) + 1)
        self.y_max_len = np.max([np.max(s) for s in y_seq_len])

        save_dir = os.path.join(config.save_dir, config.model)

        dataset = (X, X_seq_len, y, y_seq_len)

        if self.config.restore_previous_model:
            indices = pickle.load(open(os.path.join(config.save_dir, config.model, "split_indices.pkl"), "rb"))["indices"]
        else:
            indices = None

        self.train_dataset, self.val_dataset, indices = prepare_dataset(dataset,
                                                                        r=SPLIT_RATIO,
                                                                        shuffle=True,
                                                                        all_indices=indices)
        if not self.config.restore_previous_model:
            with open(os.path.join(save_dir,  "split_indices.pkl"), 'wb') as f:
                pickle.dump({"indices": indices}, f, pickle.HIGHEST_PROTOCOL)

        logging.info("Number of training examples: %d" % sum([d[0].shape[0] for d in self.train_dataset]))
        logging.info("Number of validation examples: %d" % sum([d[0].shape[0] for d in self.val_dataset]))

    def _load_evaluation_data(self, eval_path):
        X, X_seq_len, y, y_seq_len = self._load_data_from_folder(eval_path, size=None)
        dataset = (X, X_seq_len, y, y_seq_len)

        self.eval_dataset, _, _ = prepare_dataset(dataset,
                                                  r=1.0,
                                                  shuffle=False,
                                                  all_indices=None)
        
        logging.info("Size of evaluation dataset: %d" % sum([d[0].shape[0] for d in self.eval_dataset]))

    def _load_inferring_data(self, images):
        config = self.config

        self.X_infer, self.X_infer_seq_len = handle_inferring(images,
                                                              config.image_size,
                                                              flip_image=config.flip_image,
                                                              buckets=1)

    def _load_data_from_folder(self, data_folder, size=None):
        X, X_seq_len, y, y_seq_len = load_data(self.config.char_or_lig,
                                               data_folder,
                                               self.config.image_size,
                                               size=size,
                                               image_width_range=self.config.image_width_range,
                                               flip_image=self.config.flip_image,
                                               buckets=self.config.buckets)

        return X, X_seq_len, y, y_seq_len

    def _handle_constants(self):
        config = self.config

        save_dir = os.path.join(config.save_dir, config.model,  "constants.pkl")

        if self.TRAIN and not config.restore_previous_model:
            if config.use_teacher_forcing:
                self.extra_codes = {'<SOS>': self.vocab_size, '<EOS>': self.vocab_size+1, '<PAD>': self.vocab_size+2}
                self.vocab_size += 3
                self.y_max_len += 1

            with open(save_dir, 'wb') as f:
                pickle.dump({"vocab_size": self.vocab_size,
                            "y_max_len": self.y_max_len}, f, pickle.HIGHEST_PROTOCOL)

        else:
            constants = pickle.load(open(save_dir, "rb"))
            self.vocab_size, self.y_max_len = constants['vocab_size'], constants['y_max_len']

            if config.use_teacher_forcing:
                self.extra_codes = {'<SOS>': self.vocab_size-3, '<EOS>': self.vocab_size-2, '<PAD>': self.vocab_size-1}

        if self.TRAIN:
            logging.info("Vocab Size: %d " % self.vocab_size)
            logging.info("Maximum Length of Output Sequence: %d" % self.y_max_len)

    def _placeholders(self):
        config = self.config

        self.X_placeholder = tf.placeholder(tf.float32, [None, config.image_size[0], config.image_size[1]], name="X_placeholder")
        self.X_seq_len_placeholder = tf.placeholder(tf.int32, [None], name="X_seq_len_placeholder")
 
        if config.use_teacher_forcing:
            self.y_in_placeholder = tf.placeholder(tf.int32, [None, None], name="y_in_placeholder")
            self.y_out_placeholder = tf.placeholder(tf.int32, [None, None], name="y_out_placeholder")

        else:
            self.y_placeholder = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2], name="y_index_placeholder"),
                                                 tf.placeholder(tf.int32, [None], name="y_values_placeholder"),
                                                 tf.placeholder(tf.int64,[2], name="y_size_placeholder"))
 
        self.y_seq_len_placeholder = tf.placeholder(tf.int32, [None], name="y_seq_len_placeholder")
        self.is_training_placeholder = tf.placeholder(tf.bool, name="is_training_placeholder")

        if config.dropout is not None:
            self.dropout_placeholder = tf.placeholder(tf.float32, [], name="dropout_placeholder")

    def _initialize_graph(self):
        tic = time.time()
        
        self.global_step = tf.Variable(0, trainable=False)
        self._placeholders()

        regularizer = tf.contrib.layers.l2_regularizer(self.config.l2_regularizer_scale)
        with tf.variable_scope("model", regularizer=regularizer):
            self._build_graph()
            self.merged_summary = tf.summary.merge_all()
        
        toc = time.time()
        
        if self.TRAIN: logging.info("Time taken to build graph: %09.5f secs" % (toc-tic)) 

    def _build_graph(self):
        raise NotImplementedError

    def _setup_CTC(self, logits, labels, logits_seq_len, vocab_size):
        config = self.config
        if not config.use_dynamic_lengths:
            logits_seq_len = tf.fill([tf.shape(logits)[0]], tf.shape(logits)[1])

        logits_T = tf.transpose(logits, [1, 0, 2])
        loss = tf.nn.ctc_loss(labels,
                              logits_T,
                              logits_seq_len,
                              ctc_merge_repeated=True,
                              ignore_longer_outputs_than_inputs=False,
                              time_major=True)

        if config.decoder_type == "greedy_search":
            decoded, _ = tf.nn.ctc_greedy_decoder(logits_T, logits_seq_len, merge_repeated=True)
            decoded = decoded[0]

        elif config.decoder_type == "beam_search":
            decoded, _ = tf.nn.ctc_beam_search_decoder(logits_T,
                                                       logits_seq_len,
                                                       beam_width=config.beam_width,
                                                       top_paths=1,
                                                       merge_repeated=True)
            decoded = decoded[0]

        elif config.decoder_type == "prefix_beam_search":
            decoded = ctc_prefix_beam_search_decoder(logits,
                                                     vocab_size+1,
                                                     config.data_folder,
                                                     config.lm_path,
                                                     n_grams=config.ngrams,
                                                     beams=config.beam_width,
                                                     alpha=config.alpha,
                                                     beta=config.beta,
                                                     discard_probability=config.discard_probability)
        else:
            raise ValueError("decoder_type %s not supported" % config.decoder_type)
        return tf.reduce_mean(loss), decoded

    def _optimize(self, loss):
        config = self.config

        loss += sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        if config.anneal_lr:
            lr = tf.train.exponential_decay(config.lr, self.global_step, config.anneal_lr_every, config.anneal_lr_rate)
        else:
            lr = config.lr

        optfn = _get_optfn(config.optimizer)(lr)

        gvp = optfn.compute_gradients(loss)
        if config.max_grad_norm is not None:
            grad = [g for g, _ in gvp]
            var  = [v for _, v in gvp]
            clipped, grad_norm = tf.clip_by_global_norm(grad, config.max_grad_norm)
            gvp = zip(clipped, var)
        else:
            grad_norm = tf.global_norm([g for g, _ in gvp])

        train_op = optfn.apply_gradients(gvp, global_step=self.global_step)

        return train_op, grad_norm

    def _initialize_model(self, restore_previous, verbose=True):
        config = self.config

        self.sess = self._get_session()
        self.saver = tf.train.Saver(save_relative_paths=True)
       
        if not os.path.isdir(config.save_dir):
            os.mkdir(config.save_dir)

        if restore_previous:
            if verbose:
                logging.info('\nInitializing with stored values')
            restored_model = tf.train.latest_checkpoint(os.path.join(config.save_dir, config.model, ""))

            if not restored_model:
                raise Exception('No saved model found!')
            self.saver.restore(self.sess, restored_model)

        else:
            if verbose:
                logging.info('\nInitializing with new values')
            self.sess.run(tf.global_variables_initializer())

        if verbose:
            tic = time.time()
            params = tf.trainable_variables()
            num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval(session=self.sess)), params))
            toc = time.time()
            logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        self.writer = tf.summary.FileWriter(os.path.join(config.save_dir, "Tensorboard"), self.sess.graph)

    def _get_session(self):
        if self.config.use_gpu:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            session = tf.Session(config=config)
        else:
            session = tf.Session()

        if self.config.debug_mode:
            session = tf_debug.LocalCLIDebugWrapperSession(tf.Session())
            add_filters(session)

        return session

    def _create_feed_dict(self, X, X_seq_len, y=None, y_seq_len=None, dropout=None, is_training=True):
        config = self.config

        feed_dict = {self.X_placeholder: X, self.X_seq_len_placeholder: X_seq_len, self.is_training_placeholder:is_training}

        if y is not None:
            if config.use_sparse_labels and not config.use_teacher_forcing:
                y = to_sparse(y)
                feed_dict[self.y_placeholder] = y

            if config.use_teacher_forcing:
                max_len = max([len(yi) for yi in y])

                y_in = append_start_token(y, self.extra_codes["<SOS>"])
                y_in = append_end_token(y_in, self.extra_codes["<EOS>"], except_max=True)
                y_in = pad_labels(y_in, max_len+1, self.extra_codes["<PAD>"])

                y_out = append_end_token(y, self.extra_codes["<EOS>"], except_max=False)
                y_out = pad_labels(y_out, max_len+1, self.extra_codes["<PAD>"])

                feed_dict[self.y_in_placeholder] = y_in
                feed_dict[self.y_out_placeholder] = y_out

                y_seq_len += 1

            feed_dict[self.y_seq_len_placeholder] = y_seq_len

        if dropout is not None:
            feed_dict[self.dropout_placeholder] = dropout

        return feed_dict

    def _get_output_indices(self, preds, labels=None, num_examples=None, test_flag=False):
        config = self.config

        if config.use_sparse_labels:
            preds = from_sparse(preds, num_examples)

        if config.use_teacher_forcing:
            pruned_preds = []
            if test_flag:
                for pred in preds:
                    pred = list(pred)
                    pruned_pred = []
                    for p in pred:
                        if p == self.extra_codes["<EOS>"]: break
                        if p != self.extra_codes["<EOS>"] and p != self.extra_codes["<PAD>"]:
                            pruned_pred.append(p)
                    pruned_preds.append(pruned_pred)
            else:
                preds = np.argmax(preds, 2)
                for label, pred in zip(labels, preds):
                    pruned_preds.append(pred[:len(label)])

            preds = pruned_preds

        return preds

    def _get_accuracy(self, preds, labels, test_flag=False, average=True):
        preds = self._get_output_indices(preds, labels, labels.shape[0], test_flag)

        lev_dist = []
        for label, pred in zip(list(labels), list(preds)):
            lev_dist.append(Levenshtein(labels=label, logits=pred))

        if average:
            return sum(lev_dist)/len(lev_dist)
        else:
            return lev_dist

    def _train_on_batch(self, dataset, dropout, sess, find_accuracy=False):
        feed_dict = self._create_feed_dict(*dataset, dropout=dropout, is_training=True)

        if find_accuracy:
            _, grad_norm, loss, decoded, summary = sess.run([self.train_op, self.grad_norm, self.loss, 
                                                            self.decoded_train, self.merged_summary], feed_dict)
            accuracy = self._get_accuracy(decoded, dataset[2], test_flag=False) 

            return grad_norm, loss, summary, accuracy

        else:
            _, grad_norm, loss, summary = self.sess.run([self.train_op, self.grad_norm, self.loss, self.merged_summary], feed_dict)

            return grad_norm, loss, summary

    def _train(self, dataset, sess, histories):
        config = self.config

        dropout = config.dropout
        for i, mini in enumerate(minibatches(dataset, config.batch_size, shuffle=True)):
            if i % config.print_every == 0:
                tic = time.time()
                grad_norm, loss, summary, accuracy = self._train_on_batch(mini, config.dropout, sess, find_accuracy=True)
                self.writer.add_summary(summary, i)
                toc = time.time()
                _log_metrics(iteration=i, loss=loss, accuracy=accuracy, time=toc-tic)
            else:
                grad_norm, loss, summary = self._train_on_batch(mini, config.dropout, sess, find_accuracy=False)

            histories['grad_norm_history'].append(grad_norm)
            histories['train_loss_history'].append(loss)

        return histories

    def _log_validation_results(self, val_loss, val_train_accuracy, val_infer_accuracy):
        if self.config.use_teacher_forcing:
            _log_metrics(loss=val_loss, accuracy=val_train_accuracy, custom_message="With teacher forcing:")
            _log_metrics(accuracy=val_infer_accuracy, custom_message="Without teacher forcing:")
        else:
            _log_metrics(loss=val_loss, accuracy=val_train_accuracy)

    def _validate(self, dataset, sess):
        config = self.config

        dropout = 0.0 if config.dropout is not None else None
        val_loss = []
        train_accuracy = []
        infer_accuracy = []

        for i, mini in enumerate(minibatches(dataset, config.batch_size, shuffle=False)):
            feed_dict_val = self._create_feed_dict(*mini, dropout=dropout, is_training=False)
            loss, decoded_train, decoded_infer = sess.run([self.loss, self.decoded_train, self.decoded_infer], feed_dict_val)

            val_loss.append(loss)
            _train_accuracy = self._get_accuracy(decoded_train, mini[2], test_flag=False, average=False)
            train_accuracy.extend(_train_accuracy)

            if config.use_teacher_forcing:
                infer_accuracy.extend(self._get_accuracy(decoded_infer, mini[2], test_flag=True, average=False))
            else:
                infer_accuracy.extend(_train_accuracy)
            
            if self.EVALUATE:
                print("Done: ", i)

        val_loss = sum(val_loss)/len(val_loss)
        val_train_accuracy = sum(train_accuracy)/len(train_accuracy)
        val_infer_accuracy = sum(infer_accuracy)/len(infer_accuracy)

        self._log_validation_results(val_loss, val_train_accuracy, val_infer_accuracy)

        return val_loss, val_train_accuracy, val_infer_accuracy

    def _get_history(self, restore_previous):
        histories = {}
        if restore_previous:
            histories_file = os.path.join(self.config.save_dir, self.config.model, "histories.pkl")
            histories = pickle.load(open(histories_file, "rb"))

        else:
            histories = {"train_loss_history": [],
                         "grad_norm_history": [],
                         "val_loss_history": [],
                         "val_train_accuracy_history": [],
                         "val_infer_accuracy_history": [],
                         "best_val_loss": None,
                         "best_val_train_accuracy": None,
                         "best_val_infer_accuracy": None,
                         "true_epochs": 0,
                         "eff_epochs": 0}

        return histories

    def _fit(self):
        config = self.config

        self._initialize_model(config.restore_previous_model)
        histories = self._get_history(self.config.restore_previous_model)
        histories['true_epochs'] = histories['eff_epochs'] + 1 # reset true epochs to eff_epochs

        for epoch in range(config.num_epochs):
            tic = time.time()
            _log_metrics(epoch=(histories['true_epochs'], histories['eff_epochs']))

            histories = self._train(self.train_dataset, self.sess, histories)
            
            logging.info("\nEvaluating on Validation Set")
            val_loss, val_train_accuracy, val_infer_accuracy = self._validate(self.val_dataset, self.sess)
            histories['val_loss_history'].append(val_loss)
            histories['val_train_accuracy_history'].append(val_train_accuracy)
            histories['val_infer_accuracy_history'].append(val_infer_accuracy)

            save_dir = os.path.join(config.save_dir, config.model)

            if (config.save_best and (histories['best_val_loss'] == None or val_loss <= histories['best_val_loss'])) or not config.save_best:
                histories['eff_epochs'] = histories['true_epochs']
                histories['best_val_loss'] = val_loss
                histories['best_val_train_accuracy'] = val_train_accuracy
                histories['best_val_infer_accuracy'] = val_infer_accuracy

                self.saver.save(self.sess, os.path.join(save_dir, config.model))

            elif config.restore_best_model == True and \
                 val_loss <= histories['best_val_loss'] - config.restore_best_model_range and \
                 epoch != config.num_epochs-1:
                    logging.info("Restoring best model")
                    restored_model = tf.train.latest_checkpoint(save_dir)

                    if restored_model:
                        self.saver.restore(self.sess, restored_model)
                        histories['true_epochs'] = histories['eff_epochs'] + 1
                    else:
                        logging.info("No saved model found! Continuing with current model.")
           
            with open(os.path.join(save_dir, "histories.pkl"), 'wb') as f:
                pickle.dump(histories, f, pickle.HIGHEST_PROTOCOL) 
 
            toc = time.time()
            logging.info("Time taken for this epoch: %010.5f" % (toc-tic))

            if config.early_stopping:
                if histories['true_epochs'] - histories['eff_epochs'] >= config.stop_after_num_epochs:
                    logging.info("No improvement for last %d epochs. Hence, stopping." % config.stop_after_num_epochs)
                    break

            print("Terminate? (Enter Y)")
            t_i, t_o, t_e = select.select([sys.stdin], [], [], TIMEOUT)
            if (t_i):
                terminate = sys.stdin.readline().strip()
                if terminate == 'Y' or terminate == 'y':
                    logging.info("Terminating at user's request")
                    break
            else:
                histories['true_epochs'] += 1
                continue

        logging.info("\nReporting evaluations on {} model".format("best" if config.save_best else "final"))
        self._log_validation_results(histories['best_val_loss'], histories['best_val_train_accuracy'], histories['best_val_infer_accuracy'])

        return histories['train_loss_history'], histories['val_loss_history'], histories['grad_norm_history']

    def _evaluate(self):
        self._initialize_model(True, verbose=False)
        self._validate(self.eval_dataset, self.sess)

    def _infer(self):
        dropout = 0.0 if self.config.dropout is not None else None
        feed_dict = self._create_feed_dict(self.X_infer, self.X_infer_seq_len, y=None, y_seq_len=None, dropout=dropout, is_training=False)

        if self.config.save_alignments:
            decoded, alignment = self.sess.run([self.decoded_infer, self.infer_alignment], feed_dict)
            _create_attention_video(self.X_infer, alignment, self.config.save_dir, decoded, self.config.data_folder, self.extra_codes)

        else:
            decoded, = self.sess.run([self.decoded_infer], feed_dict)

        output = self._get_output_indices(decoded, labels=None, num_examples=self.X_infer.shape[0], test_flag=True)

        return list(output)

#######################################################################################################

def _log_metrics(epoch=None, iteration=None, loss=None, accuracy=None, time=None, custom_message=None):
    log_str = "" if custom_message is None else custom_message
    if epoch is not None:
        log_str = log_str + "\nEpoch %d / %d" % (epoch[0], epoch[1])
    if iteration is not None:
        log_str = log_str + " " if log_str != "" else log_str
        log_str += ("Iteration: %03d" % iteration)
    if loss is not None:
        log_str = log_str + " " if log_str != "" else log_str
        log_str += "Loss: %010.5f" % loss
    if accuracy is not None:
        log_str = log_str + " " if log_str != "" else log_str
        log_str += "Accuracy (Levenshtein): %06.2f%%" % accuracy
    if time is not None:
        log_str = log_str + " " if log_str != "" else log_str
        log_str += "Time taken: %09.5f secs" % time

    logging.info(log_str)

def _get_optfn(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        raise ValueError("Unknown optimizer: %s", opt)

    return optfn

def _plot_results(train_loss, val_loss, grad_norms, save_dir):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    plt.subplot(2, 1, 1)
    plt.title("Losses and Gradient Norm")
    plt.plot(np.arange(0, len(val_loss), len(val_loss)/len(train_loss)), train_loss, label="Train Loss")
    plt.plot(np.arange(1, len(val_loss)+1, 1), val_loss, label="Val Loss")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(0, len(val_loss), len(val_loss)/len(train_loss)), grad_norms, label="Gradients")
    plt.xlabel("Minibatch")
    plt.ylabel("Gradients")

    plt.savefig(save_dir + "/plots.png")
    plt.close()

# TODO: Only supports one image right now - Extend to multiple images.
def _create_attention_video(images, alignments, save_dir, outputs, data_folder, codes):
    alignment = alignments[:,0,:]
    image = images[0,:,:]
    output = outputs[0]

    alignment = np.concatenate([np.zeros([1,alignment.shape[1]]), alignment], axis=0)
    alignment = 255.0 - 255.0/np.max(alignment)*alignment
    image = 255.0 - 255.0/np.max(image)*image
    scale = math.ceil(image.shape[0]/alignment.shape[1])

    alignment_expanded = np.repeat(np.expand_dims(alignment, axis=2), 4, axis=0)
    alignment_expanded = np.repeat(alignment_expanded, 3, axis=1)
    alignment_concat = np.concatenate((alignment_expanded, alignment_expanded, alignment_expanded), axis=2)
    background = np.zeros_like(alignment_concat)
    background[:,:,0] = 255
    background[:,:,1] = 218
    background[:,:,2] = 179
    alignment_overlayed = cv2.addWeighted(background, 0.1, alignment_concat.astype(background.dtype), 0.5, 1.0)
    cv2.imwrite(os.path.join(save_dir, "infer_alignment.png"), np.uint8(alignment_overlayed))

    alignment = np.repeat(alignment, scale, axis=1)
    diff = alignment.shape[1] - image.shape[0]

    if diff > 0:
        image = np.pad(image, (diff, 0), 'constant')
    elif diff < 0:
        alignment = np.pad(alignment, (0, diff), 'constant')

    image_list = []
    image = np.expand_dims(image, axis=2)
    image = np.concatenate((image, image, image), axis=2)
    for a in alignment:
        a = np.reshape(a, [-1,1])
        a_repeat = np.expand_dims(np.repeat(a, image.shape[1], axis=1), axis=2)
        a_rgb = np.concatenate((a_repeat, a_repeat, a_repeat), axis=2)
        a_rgb[:,:,1] = 0
        a_rgb[:,:,2] = 0
        overlayed = cv2.addWeighted(image, 0.5, a_rgb.astype(image.dtype), 0.5, 1.0)
        overlayed = np.transpose(overlayed, [1, 0, 2])
        image_list.append(np.uint8(overlayed))

    text_height = 12
    preds = _video_helper_for_preds(output, codes, data_folder)

    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    writer = cv2.VideoWriter(os.path.join(save_dir, "infer_alignment.mp4"),
                             fourcc,
                             0.5,
                             (image.shape[0], image.shape[1]+text_height))

    font = ImageFont.truetype("fonts/jameel_noori_nastaleeq.ttf", 10)
    
    save_images_dir = os.path.join(save_dir, "attention_images")
    if os.path.isdir(save_images_dir):
        shutil.rmtree(save_images_dir)
    os.mkdir(save_images_dir)

    image_list = image_list[:-1]
    assert len(image_list) == len(preds)
    for i, (img, pred) in enumerate(zip(image_list, preds)):
        txt = np.zeros([text_height, img.shape[1], 3], dtype=np.uint8)
        txt = Image.fromarray(txt)
        
        draw = ImageDraw.Draw(txt)
        w, h = draw.textsize(pred, font=font)
        W, H = img.shape[1], text_height
        draw.text((((W-w)/2,(H-h)/2)), pred, font=font)
        
        concat = np.concatenate((np.array(txt), img), axis=0)

        cv2.imwrite(os.path.join(save_images_dir, "infer_alignment_{}.png".format(i)), cv2.bitwise_not(concat))
        writer.write(cv2.bitwise_not(concat))

    writer.release()

def _video_helper_for_preds(pred, codes, data_folder):
    pred = list(pred)
    lt = get_lookup_table(data_folder)

    pruned_pred = []

    for p in pred:
        if p == codes["<SOS>"]:
            pruned_pred.append("<START>")
        if p == codes["<EOS>"]:
            pruned_pred.append("<END>")
            break
        if p == codes["<PAD>"]:
            pruned_pred.append("PAD")
        else:
            pruned_pred.append(lt[p])

    return pruned_pred  #[1:]
