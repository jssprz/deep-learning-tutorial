#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 16:05:34 2018

@author: jsaavedr
"""

import enum
import math
import threading
import numpy as np
from sklearn.metrics import average_precision_score
from . import norm


class JNorm(enum.Enum):
    SQUARE_ROOT_UNIT = 1
    UNIT = 2


class Metrics(enum.Enum):
    L1 = 1
    L2 = 2


# class DeepSearcher:
#
#     def __init__(self, configuration, params, load_feats_vectors=False):
#         self.configuration = configuration
#         # loading predictor
#         self.fast_predictor = fp.FastPredictor(configuration, params)
#         self.metric = params['metric']
#         self.dim = -1
#         self.norm = JNorm.SQUARE_ROOT_UNIT
#
#         if load_feats_vectors:
#             self.__load_feats_vectors()
#             self.size = len(self.true_labels)
#         else:
#             with open(os.path.join(self.configuration.data_dir, 'filelist.txt')) as f:
#                 paths, self.true_labels = zip(*[l.split('\t') for l in f.readlines()])
#                 self.size = len(paths)
#                 self.__compute_features(paths)
#
#     def load(self, path):
#         t_start = time.time()
#         name_file = os.path.join(path, "names.txt")
#         feature_file = os.path.join(path, "features.des")
#         f_feature = open(feature_file, "r+b")  # open binary file
#         header = f_feature.read(3 * 4)  # reading 3 int32, each one with 4 bytes
#         header = np.array(struct.unpack("i" * 3, header))
#         self.size = header[0]
#         self.dim = header[1]
#         # reading data
#         data_size = self.size * self.dim  # dim times number of objects
#         data = f_feature.read(data_size * 4)
#         data = np.array(struct.unpack("f" * data_size, data))
#         # reshape in header[0] x header[1]
#         self.data = data.reshape([header[0], header[1]])
#         f_feature.close()
#
#         # reading labels
#         with open(name_file) as f_name:
#             self.true_labels = [line.rstrip() for line in f_name]
#
#         if self.norm == JNorm.SQUARE_ROOT_UNIT:
#             self.data = norm.square_root_norm(self.data)
#         print("Data was loaded OK")
#         elapsed = (time.time() - t_start) * 1000
#         print(">>> shape of data {}".format(self.data.shape))
#         print(">>> loaded took {} ms".format(elapsed))
#
#     def __compute_features(self, paths):
#         features = []
#         for p in paths:
#             features.append(self.normalize(self.fast_predictor.get_deep_features(p)))
#         self.feats_vectors = np.array(features)
#
#     def normalize(self, features_vec):
#         if self.norm == JNorm.SQUARE_ROOT_UNIT:
#             return norm.square_root_norm(features_vec)
#         return features_vec
#
#     def get_feature(self, img_filename):
#         """
#         input_image is a filename or a numpy array
#         """
#         norm_features = self.normalize(self.fast_predictor.get_deep_features(img_filename))
#         return norm_features
#
#     # def compute_metric(self):
#     #     if self.metric == 'euclidean':
#
#     def search(self, feats_vec, k, leave_one_out=True):
#         """
#
#         :param feats_vec: features vector to be search
#         :param k: the number of elements to be taken into account of the rank
#         :param leave_one_out:
#         :return:
#         """
#         if k >= len(self.feats_vectors):
#             k = -1
#
#         t_start = time.time()
#         dist = np.sqrt(np.sum((self.data - feats_vec) ** 2, axis=1))
#         sorted_idx = sorted(range(self.size), key=lambda x: dist[x])
#         t_elapsed = (time.time() - t_start) * 1000
#         print('>>> search took {} ms'.format(t_elapsed))
#
#         s = 1 if leave_one_out else 0
#         return sorted_idx[s:], self.true_labels[sorted_idx[s:]], dist[sorted_idx[s:]] if k == -1 else \
#             sorted_idx[s:k], self.true_labels[sorted_idx[s:k]], dist[sorted_idx[s:k]]
#
#     def get_true_label(self, idx):
#         """return tne name of the object with id = idx"""
#         return self.true_labels[idx]
#
#     def average_precision(self, feats_vec, label, k):
#         y_ids, y_true, y_score = self.search(feats_vec, k)
#         return average_precision_score(y_true, y_score, pos_label=label)
#
#     def mean_average_precision(self, feats_vec_list, label_list, k):
#         ap_sum = 0
#         for i in range(len(feats_vec_list)):
#             ap_sum += self.average_precision(feats_vec_list[i], label_list[i], k)
#         return ap_sum / len(feats_vec_list)
#
#     def save_feats_vectors(self):
#         path = os.path.join(self.configuration.data_dir, '{}_features.h5'.format(self.configuration.arch))
#         with h5py.File(path, 'w') as h5:
#             ds_labels = h5.create_dataset('true_labels')
#             ds_feats = h5.create_dataset('{}_feats'.format(self.configuration.arch))
#             for i, feats in enumerate(self.feats_vectors):
#                 ds_labels[i] = self.true_labels[i]
#                 ds_feats[i] = feats
#
#     def __load_feats_vectors(self):
#         path = os.path.join(self.configuration.data_dir, '{}_features.h5'.format(self.configuration.arch))
#         with h5py.File(path, 'rb') as h5:
#             self.true_labels = h5['true_labels'][...]
#             self.feats_vectors = h5['{}_feats'.format(self.configuration.arch)][...]


class DeepSearcher:
    def __init__(self, features, true_labels, params):
        assert len(features) == len(true_labels), 'there are not the same count of labels and features vectors'
        self.metric = Metrics[params['metric']]
        self.norm = JNorm[params['norm']]
        self.feats_vectors = self.normalize(np.array(features))
        self.true_labels = np.array(true_labels)
        self.size = len(features)

    def search(self, feats_vec, k, leave_one_out=True):
        """

        :param feats_vec: features vector to be search
        :param k: the number of elements to be taken into account of the rank
        :param leave_one_out:
        :return:
        """
        if k >= len(self.feats_vectors):
            k = -1

        if self.metric == Metrics.L2:
            dist = np.sqrt(np.sum((self.feats_vectors - feats_vec) ** 2, axis=1))
        else:  # L1
            dist = np.sqrt(np.sum((self.feats_vectors - feats_vec), axis=1))

        sorted_idx = sorted(range(self.size), key=lambda x: dist[x])

        s = 1 if leave_one_out else 0
        return (sorted_idx[s:], self.true_labels[sorted_idx[s:]], dist[sorted_idx[s:]]) if k == -1 else \
            (sorted_idx[s:k], self.true_labels[sorted_idx[s:k]], dist[sorted_idx[s:k]])

    def average_precision(self, feats_vec, label, k):
        y_ids, y_true, y_score = self.search(feats_vec, k)
        return average_precision_score([1 if v == label else 0 for v in y_true], y_score)

    def mean_average_precision(self, feats_vec_list, label_list, k, normalize=True):
        assert len(feats_vec_list) == len(label_list)
        if normalize:
            norm_feats_vec_list = self.normalize(np.array(feats_vec_list))
        else:
            norm_feats_vec_list = feats_vec_list
        ap_sum = 0
        for i in range(len(feats_vec_list)):
            ap_sum += self.average_precision(norm_feats_vec_list[i], label_list[i], k)
        return ap_sum / len(feats_vec_list)

    def inner_mean_average_precision(self, k):
        threads_count = 4
        threads = []
        partials_sums = [0 for _ in range(threads_count)]

        def worker(id):
            partial_size = int(self.size/threads_count)
            for i in range(partial_size * (id-1), partial_size * id):
                ap = self.average_precision(self.feats_vectors[i], self.true_labels[i], k)
                partials_sums[id] += 0 if math.isnan(ap) else ap
                if i % 1000 == 0:
                    print(ap, partials_sums[id])

        for i in range(threads_count):
            t = threading.Thread(target=worker, args=(i+1,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        return sum(partials_sums) / self.size

    def normalize(self, features_vec):
        if self.norm == JNorm.SQUARE_ROOT_UNIT:
            return norm.square_root_norm(features_vec)
        return features_vec
