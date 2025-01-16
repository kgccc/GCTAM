# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  14/9/2022
# version： Python 3.7.8
# @File : raw_affinity.py
# @Software: PyCharm
# @Institution: SMU


import torch.nn as nn
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_auc_score
import random
import os
import dgl
import pandas as pd
from utils import *
import argparse
from tqdm import tqdm
import time
import matplotlib

matplotlib.use('agg')

parser = argparse.ArgumentParser(description='Truncated Affinity Maximization for Graph Anomaly Detection')
parser.add_argument('--dataset', type=str,
                    default='YelpChi-all')  # 'BlogCatalog'  'ACM'  'Amazon' 'Facebook'  'Reddit'  'YelpChi' 'Amazon-all' 'YelpChi-all'
args = parser.parse_args()
# Load and preprocess data
adj, features, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)

if args.dataset in ['Amazon', 'YelpChi']:
    features, _ = preprocess_features(features)
    raw_features = features

else:
    raw_features = features.todense()
    features = raw_features

dgl_graph = adj_to_dgl_graph(adj)
nb_nodes = features.shape[0]
ft_size = features.shape[1]
raw_adj = adj
raw_adj = (raw_adj + sp.eye(adj.shape[0])).todense()
adj = (adj + sp.eye(adj.shape[0])).todense()
raw_features = torch.FloatTensor(raw_features[np.newaxis])
features = torch.FloatTensor(features[np.newaxis])
adj = torch.FloatTensor(adj[np.newaxis])
raw_adj = torch.FloatTensor(raw_adj[np.newaxis])


def raw_affinity(feature, adj_matrix):
    feature = feature / torch.norm(feature, dim=-1, keepdim=True)
    sim_matrix = torch.mm(feature, feature.T)
    sim_matrix = torch.squeeze(sim_matrix) * adj_matrix

    sim_matrix[torch.isinf(sim_matrix)] = 0
    sim_matrix[torch.isnan(sim_matrix)] = 0
    row_sum = torch.sum(adj_matrix, 0)
    r_inv = torch.pow(row_sum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    message = torch.sum(sim_matrix, 1)

    message = message * r_inv
    # message = (message - torch.min(message)) / (torch.max(message) - torch.min(message))
    # message[torch.isinf(message)] = 0.
    # message[torch.isnan(message)] = 0.
    return message


def GCTAM_draw_boxplot(message, ano_label, dataset, message_name):
    normal_message_all = message[ano_label == 0]
    abnormal_message_all = message[ano_label == 1]

    # Create a boxplot to compare normal and abnormal message distributions
    plt.boxplot([normal_message_all, abnormal_message_all], labels=['Normal', 'Abnormal'], patch_artist=True,
                boxprops=dict(facecolor='skyblue', color='blue'),
                flierprops=dict(marker='o', color='red', markersize=6))

    # Labels and title
    plt.title(f'{dataset} - {message_name} Boxplot', fontsize=15)
    plt.ylabel('Value', fontsize=12)

    # Save the plot
    plt.savefig(f'pdf/{dataset}_{message_name}_Boxplot.pdf', format="pdf")
    plt.close()


message_sum = raw_affinity(features[0, :, :], raw_adj[0, :, :])
message = np.array(message_sum)
message = 1 - normalize_score(message)
# # message = normalize_score(message)
GCTAM_draw_pdf_separated_1(1 - message, ano_label, args.dataset, "RA_message")

# message_name = "mes_1"
# message_name = "mes_2"
# message_name = "mes_3"

# final_message = np.load(f'{args.dataset}_final_message.npy')
# GCTAM
message_name = "final_message"
# final_message = np.load(f'{args.dataset}_0_GCTAM_{message_name}.npy')
final_message = np.load(f'{args.dataset}_{message_name}.npy')
# GCTAM_draw_pdf_separated(final_message, ano_label, args.dataset, message_name)
GCTAM_draw_pdf_separated_1(1 - final_message, ano_label, args.dataset, message_name)
# GCTAM_draw_pdf_separated_1(final_message, ano_label, args.dataset, message_name)

# TAM
final_message = np.load(f'{args.dataset}_TAM_message.npy')
message_name = "TAM_message"
# GCTAM_draw_pdf_separated(1 - final_message, ano_label, args.dataset, message_name)
GCTAM_draw_pdf_separated_1(1 - final_message, ano_label, args.dataset, message_name)

# GCTAM_draw_pdf(1 - final_message, ano_label, args.dataset, message_name)
# GCTAM_draw_boxplot(1 - final_message, ano_label, args.dataset, message_name)

# message_name = "mes_1"
# # final_message = np.load(f'{args.dataset}_final_message.npy')
# final_message = np.load(f'{args.dataset}_{message_name}.npy')
# GCTAM_draw_pdf(final_message, ano_label, args.dataset, message_name)

# # message_name = "mes_1"
# message_name = "mes_2"
# # message_name = "mes_3"
# # message_name = "final_message"
#
# # final_message = np.load(f'{args.dataset}_final_message.npy')
# final_message = np.load(f'{args.dataset}_{message_name}.npy')
# GCTAM_draw_pdf(final_message, ano_label, args.dataset, message_name)
#
# # message_name = "mes_1"
# # message_name = "mes_2"
# message_name = "mes_3"
# # message_name = "final_message"
#
# # final_message = np.load(f'{args.dataset}_final_message.npy')
# final_message = np.load(f'{args.dataset}_{message_name}.npy')
# GCTAM_draw_pdf(final_message, ano_label, args.dataset, message_name)
