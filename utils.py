import os
import shutil

import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
import dgl
from scipy.stats import norm
import torch.nn.functional as F

import matplotlib

matplotlib.use('agg')


def inference(feature, adj_matrix):
    feature = feature / torch.norm(feature, dim=-1, keepdim=True)
    sim_matrix = torch.mm(feature, feature.T)

    sim_matrix = torch.squeeze(sim_matrix) * adj_matrix

    # todo修改 隐藏
    # sim_matrix = torch.squeeze(sim_matrix) * adj_matrix

    row_sum = torch.sum(adj_matrix, 0)
    r_inv = torch.pow(row_sum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    message = torch.sum(sim_matrix, 1)
    message = message * r_inv

    return message


def max_message(feature, adj_matrix):
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

    return - torch.sum(message), message


def max_message_1(feature, adj_matrix):
    feature = feature / torch.norm(feature, dim=-1, keepdim=True)
    sim_matrix = torch.mm(feature, feature.T)

    sim_matrix = torch.squeeze(sim_matrix) * adj_matrix
    sim_matrix[torch.isinf(sim_matrix)] = 0
    sim_matrix[torch.isnan(sim_matrix)] = 0

    row_sum = torch.sum(adj_matrix, 0)
    r_inv = torch.pow(row_sum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.

    message = torch.sum(sim_matrix, 1)

    # message1
    loss = 1 - (message * r_inv)

    # # message2
    # message = message * r_inv
    # loss = 1 - message

    # message = (message - torch.min(message)) / (torch.max(message) - torch.min(message))
    # message[torch.isinf(message)] = 0.
    # message[torch.isnan(message)] = 0.

    return torch.sum(loss), message


def max_message_2(feature, adj_matrix):
    feature = feature / torch.norm(feature, dim=-1, keepdim=True)
    sim_matrix = torch.mm(feature, feature.T)

    sim_matrix = torch.squeeze(sim_matrix) * adj_matrix
    sim_matrix[torch.isinf(sim_matrix)] = 0
    sim_matrix[torch.isnan(sim_matrix)] = 0

    row_sum = torch.sum(adj_matrix, 0)
    r_inv = torch.pow(row_sum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.

    message = torch.sum(sim_matrix, 1)

    # message1
    # message = message * r_inv
    # loss = 1 - (message * r_inv)

    # message2
    message = message * r_inv
    loss = 1 - message

    # message = (message - torch.min(message)) / (torch.max(message) - torch.min(message))
    # message[torch.isinf(message)] = 0.
    # message[torch.isnan(message)] = 0.

    return torch.sum(loss), message


def reg_edge(emb, adj):
    emb = emb / torch.norm(emb, dim=-1, keepdim=True)
    sim_u_u = torch.mm(emb, emb.T)
    adj_inverse = (1 - adj)
    sim_u_u = sim_u_u * adj_inverse
    sim_u_u_no_diag = torch.sum(sim_u_u, 1)
    row_sum = torch.sum(adj_inverse, 1)
    r_inv = torch.pow(row_sum, -1)
    r_inv[torch.isinf(r_inv)] = 0.
    sim_u_u_no_diag = sim_u_u_no_diag * r_inv
    loss_reg = torch.sum(sim_u_u_no_diag)

    return loss_reg


def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj_tensor(raw_adj):
    adj = raw_adj[0]

    row_sum = torch.sum(adj, 0)
    r_inv = torch.pow(row_sum, -0.5).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    adj = torch.mm(adj, torch.diag_embed(r_inv))
    adj = torch.mm(torch.diag_embed(r_inv), adj)
    adj = adj.unsqueeze(0)
    return adj


def normalize_adj_tensor_1(adj):
    # adj = raw_adj[0]

    row_sum = torch.sum(adj, 0)
    r_inv = torch.pow(row_sum, -0.5).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    adj = torch.mm(adj, torch.diag_embed(r_inv))
    adj = torch.mm(torch.diag_embed(r_inv), adj)
    adj = adj.unsqueeze(0)
    return adj


def normalize_score(ano_score):
    ano_score = ((ano_score - np.min(ano_score)) / (
            np.max(ano_score) - np.min(ano_score)))
    return ano_score


def process_dis(init_value, cutting_dis_array):
    r_inv = np.power(init_value, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    cutting_dis_array = cutting_dis_array.dot(sp.diags(r_inv))
    cutting_dis_array = sp.diags(r_inv).dot(cutting_dis_array)
    return cutting_dis_array


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def load_mat(dataset):
    """Load .mat dataset."""

    data = sio.loadmat("./data/{}.mat".format(dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']

    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)

    ano_labels = np.squeeze(np.array(label))
    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None

    return adj, feat, ano_labels, str_ano_labels, attr_ano_labels


def adj_to_dgl_graph(adj):
    """Convert adjacency matrix to dgl format."""
    nx_graph = nx.from_scipy_sparse_matrix(adj)
    dgl_graph = dgl.DGLGraph(nx_graph)
    return dgl_graph


# compute the distance between each node
def calc_distance(adj, seq):
    dis_array = torch.zeros((adj.shape[0], adj.shape[1]))
    row = adj.shape[0]

    for i in range(row):
        # print(i)
        node_index = torch.argwhere(adj[i, :] > 0)
        for j in node_index:
            dis = torch.sqrt(torch.sum((seq[i] - seq[j]) * (seq[i] - seq[j])))
            dis_array[i][j] = dis

    return dis_array


# 使用 GPU 加速计算欧式距离
def calc_distance_optimized(adj, seq):
    """
    计算节点之间的欧式距离，使用 GPU 加速。
    adj: 邻接矩阵 (dense tensor)
    seq: 特征矩阵 (tensor)
    """
    device = seq.device  # 确保计算在正确的设备上
    row_indices, col_indices = adj.nonzero(as_tuple=True)  # 获取所有边的索引
    src_features = seq[row_indices]  # 源节点的特征
    dst_features = seq[col_indices]  # 目标节点的特征

    # 计算欧式距离
    distances = torch.sqrt(torch.sum((src_features - dst_features) ** 2, dim=1))

    # 将距离填入距离矩阵
    dis_array = torch.zeros_like(adj).to(device)  # 初始化一个距离矩阵
    dis_array[row_indices, col_indices] = distances  # 填入距离

    return dis_array


def calculate_and_save_dis_array(adj, seq, base_path, dataset_name):
    """
    计算并保存 dis_array，如果文件存在则直接加载。

    Args:
        adj (torch.Tensor): 邻接矩阵
        seq (torch.Tensor): 特征矩阵
        dataset_name (str): 数据集名称，用作文件名
        base_path (str): 保存文件的基础路径
    Returns:
        torch.Tensor: 距离矩阵 dis_array
    """
    # 创建存储路径

    os.makedirs(base_path, exist_ok=True)
    file_path = os.path.join(base_path, f"{dataset_name}.pt")

    # 检查文件是否存在
    if os.path.exists(file_path):
        print(f"文件已存在，加载距离矩阵: {file_path}")
        dis_array = torch.load(file_path, map_location="cpu")  # 加载文件
    else:
        print(f"文件不存在，开始计算距离矩阵: {file_path}")
        dis_array = calc_distance_optimized(adj, seq)  # 计算距离矩阵
        torch.save(dis_array, file_path)  # 保存到本地
        print(f"距离矩阵保存至: {file_path}")
    return dis_array


# compute the distance between each node
def my_calc_distance(adj, seq, ana_label):
    dis_array = torch.zeros((adj.shape[0], adj.shape[1]))
    row = adj.shape[0]
    for i in range(row):
        # print(i)
        node_index = torch.argwhere(adj[i, :] > 0)
        for j in node_index:
            dis = torch.sqrt(torch.sum((seq[i] - seq[j]) * (seq[i] - seq[j])))
            dis_array[i][j] = dis
    return dis_array


# compute the distance between each node
def calc_mahalanobis_distance(adj, seq):
    dis_array = torch.zeros((adj.shape[0], adj.shape[1]))
    row = adj.shape[0]
    # 计算特征矩阵的协方差矩阵及其逆矩阵
    cov_matrix = torch.cov(seq.T)  # torch.cov 需要特征为列向量
    inv_cov_matrix = torch.linalg.inv(cov_matrix)  # 计算逆矩阵
    for i in range(row):
        # print(f"Processing node {i}")
        node_indices = torch.argwhere(adj[i, :] > 0).squeeze(1)  # 获取与节点 i 相邻的节点索引

        for j in node_indices:
            diff = seq[i] - seq[j]  # 计算节点特征差
            # 使用马氏距离公式：sqrt((x - y)^T * Sigma^(-1) * (x - y))
            dis = torch.sqrt(torch.dot(diff @ inv_cov_matrix, diff))
            dis_array[i, j] = dis

    return dis_array


def get_cos_similar(v1: list, v2: list):
    num = float(torch.dot(v1, v2))  # 向量点乘
    denom = torch.linalg.norm(v1) * torch.linalg.norm(v2)  # 求模长的乘积
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0


def calc_sim_optimized(adj_matrix, attr_matrix):
    """
    计算节点之间的余弦相似度，使用 GPU 加速。

    Args:
        adj_matrix (torch.Tensor): 邻接矩阵 (dense tensor)
        attr_matrix (torch.Tensor): 特征矩阵 (dense tensor)

    Returns:
        torch.Tensor: 相似度矩阵 (dense tensor)
    """
    device = attr_matrix.device  # 确保计算在正确的设备上
    row_indices, col_indices = adj_matrix.nonzero(as_tuple=True)  # 获取邻接矩阵中的非零索引
    src_features = attr_matrix[row_indices]  # 源节点的特征
    dst_features = attr_matrix[col_indices]  # 目标节点的特征

    # 批量计算余弦相似度
    similarities = F.cosine_similarity(src_features, dst_features, dim=1)

    # 初始化相似度矩阵，并填充计算结果
    sim_array = torch.zeros_like(adj_matrix).to(device)
    sim_array[row_indices, col_indices] = similarities

    return sim_array


def calc_sim(adj_matrix, attr_matrix):
    row = adj_matrix.shape[0]
    col = adj_matrix.shape[1]
    dis_array = torch.zeros((row, col))
    for i in range(row):
        # print(i)
        node_index = torch.argwhere(adj_matrix[i, :] > 0)[:, 0]
        for j in node_index:
            # dis = get_cos_similar(attr_matrix[i].tolist(), attr_matrix[j].tolist())
            dis = get_cos_similar(attr_matrix[i], attr_matrix[j])
            dis_array[i][j] = dis

    return dis_array


def original_graph_nsgt(dis_array, adj):
    dis_array = dis_array.cuda()
    row = dis_array.shape[0]
    dis_array_u = dis_array * adj
    mean_dis = dis_array_u[dis_array_u != 0].mean()
    for i in range(row):
        node_index = torch.argwhere(adj[i, :] > 0)
        if node_index.shape[0] != 0:
            max_dis = dis_array[i, node_index].max()
            min_dis = mean_dis
            if max_dis > min_dis:
                random_value = (max_dis - min_dis) * np.random.random_sample() + min_dis
                cutting_edge = torch.argwhere(dis_array[i, node_index[:, 0]] > random_value)
                if cutting_edge.shape[0] != 0:
                    adj[i, node_index[cutting_edge[:, 0]]] = 0

    adj = adj + adj.T
    adj[adj > 1] = 1
    return adj


def graph_nsgt(dis_array, adj):
    dis_array = dis_array.cuda()
    row = dis_array.shape[0]
    dis_array_u = dis_array * adj
    mean_dis = dis_array_u[dis_array_u != 0].mean()
    cut_edge_num = 0
    for i in range(row):
        node_index = torch.argwhere(adj[i, :] > 0)
        if node_index.shape[0] != 0:
            max_dis = dis_array[i, node_index].max()
            min_dis = mean_dis
            if max_dis > min_dis:
                random_value = (max_dis - min_dis) * np.random.random_sample() + min_dis
                cutting_edge = torch.argwhere(dis_array[i, node_index[:, 0]] > random_value)
                if cutting_edge.shape[0] != 0:
                    cut_edge_num += cutting_edge.shape[0]
                    adj[i, node_index[cutting_edge[:, 0]]] = 0

    adj = adj + adj.T
    adj[adj > 1] = 1
    return adj, cut_edge_num


def graph_nsgt_3(dis_array, adj, sim_adj, args):
    dis_array = dis_array.cuda()
    sim_adj = sim_adj.cuda()
    row = dis_array.shape[0]
    dis_array_u = dis_array * adj
    sim_adj = sim_adj * adj
    mean_dis = dis_array_u[dis_array_u != 0].mean()
    cut_edge_num = 0
    # sim_threshold = sim_adj[sim_adj != 0 ].mean()
    sim_threshold = sim_adj[(sim_adj != 1) & (sim_adj != 0)].mean()

    for i in range(row):
        node_index = torch.argwhere(adj[i, :] > 0).flatten()
        if node_index.shape[0] != 0:
            max_dis = dis_array[i, node_index].max()
            min_dis = mean_dis
            top_k = node_index.shape[0] // 2

            _, min_sim_indices = torch.topk(sim_adj[i, node_index], k=top_k, largest=False)

            if max_dis > min_dis:
                random_value = (max_dis - min_dis) * np.random.random_sample() + min_dis
                cutting_edge = torch.argwhere((dis_array[i, node_index[min_sim_indices]] > random_value) &
                                              (sim_adj[i, node_index[min_sim_indices]] < sim_threshold))

                keep_mask = torch.rand(cutting_edge.shape[0]) < 0.5
                # 步骤3：保留的边缘
                cutting_edge = cutting_edge[keep_mask]

                if cutting_edge.shape[0] != 0:
                    cut_edge_num += cutting_edge.shape[0]
                    adj[i, node_index[min_sim_indices[cutting_edge[:, 0]]]] = 0

    adj = adj + adj.T
    adj[adj > 1] = 1
    return adj, cut_edge_num


def graph_nsgt_1(dis_array, adj, sim_adj):
    dis_array = dis_array.cuda()
    sim_adj = sim_adj.cuda()
    row = dis_array.shape[0]
    dis_array_u = dis_array * adj
    sim_adj = sim_adj * adj
    mean_dis = dis_array_u[dis_array_u != 0].mean()
    cut_edge_num = 0
    # sim_threshold = sim_adj[sim_adj != 0 ].mean()
    sim_threshold = sim_adj[(sim_adj != 1) & (sim_adj != 0)].mean()

    for i in range(row):
        node_index = torch.argwhere(adj[i, :] > 0).flatten()
        if node_index.shape[0] != 0:
            max_dis = dis_array[i, node_index].max()
            min_dis = mean_dis
            top_k = node_index.shape[0] // 2

            _, min_sim_indices = torch.topk(sim_adj[i, node_index], k=top_k, largest=False)

            if max_dis > min_dis:
                random_value = (max_dis - min_dis) * np.random.random_sample() + min_dis
                cutting_edge = torch.argwhere((dis_array[i, node_index[min_sim_indices]] > random_value) &
                                              (sim_adj[i, node_index[min_sim_indices]] < sim_threshold))
                if cutting_edge.shape[0] != 0:
                    cut_edge_num += cutting_edge.shape[0]
                    adj[i, node_index[min_sim_indices[cutting_edge[:, 0]]]] = 0

    adj = adj + adj.T
    adj[adj > 1] = 1
    return adj, cut_edge_num


def graph_nsgt_2(dis_array, adj, sim_adj, args):
    dis_array = dis_array.cuda()
    sim_adj = sim_adj.cuda()
    row = dis_array.shape[0]
    dis_array_u = dis_array * adj
    sim_adj = sim_adj * adj
    mean_dis = dis_array_u[dis_array_u != 0].mean()
    cut_edge_num = 0
    # sim_threshold = sim_adj[sim_adj != 0 ].mean()
    sim_threshold = sim_adj[(sim_adj != 1) & (sim_adj != 0)].mean()

    for i in range(row):
        node_index = torch.argwhere(adj[i, :] > 0).flatten()
        if node_index.shape[0] != 0:
            max_dis = dis_array[i, node_index].max()
            min_dis = mean_dis

            top_k = node_index.shape[0] // 2
            _, min_sim_indices = torch.topk(sim_adj[i, node_index], k=top_k, largest=False)

            if max_dis > min_dis:
                random_value = (max_dis - min_dis) * np.random.random_sample() + min_dis
                cutting_edge = torch.argwhere((dis_array[i, node_index[min_sim_indices]] > random_value) & (
                        sim_adj[i, node_index[min_sim_indices]] < sim_threshold))

                if cutting_edge.shape[0] != 0:
                    cut_edge_num += cutting_edge.shape[0]
                    adj[i, node_index[min_sim_indices[cutting_edge[:, 0]]]] = 0

    adj = adj + adj.T
    adj[adj > 1] = 1
    return adj, cut_edge_num


def my_graph_nsgt(dis_array, adj, ana_label):
    dis_array = dis_array.cuda()
    row = dis_array.shape[0]
    dis_array_u = dis_array * adj
    mean_dis = dis_array_u[dis_array_u != 0].mean()
    print(f"mean_dis:{mean_dis}")
    for i in range(row):
        node_index = torch.argwhere(adj[i, :] > 0)
        if node_index.shape[0] != 0:
            max_dis = dis_array[i, node_index].max()
            min_dis = mean_dis
            if max_dis > min_dis:
                random_value = (max_dis - min_dis) * np.random.random_sample() + min_dis
                cutting_edge = torch.argwhere(dis_array[i, node_index[:, 0]] > random_value)
                if cutting_edge.shape[0] != 0:
                    adj[i, node_index[cutting_edge[:, 0]]] = 0

    adj = adj + adj.T
    adj[adj > 1] = 1
    return adj


import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.backends.backend_pdf import PdfPages


def RA_draw_pdf(message, ano_label, dataset):
    with PdfPages('{}-RA.pdf'.format(dataset)) as pdf:
        normal_message_all = message[ano_label == 0]
        abnormal_message_all = message[ano_label == 1]
        message_all = [normal_message_all, abnormal_message_all]
        mu_0 = np.mean(message_all[0])
        sigma_0 = np.std(message_all[0])
        print('The mean of normal {}'.format(mu_0))
        print('The std of normal {}'.format(sigma_0))
        mu_1 = np.mean(message_all[1])
        sigma_1 = np.std(message_all[1])
        print('The mean of abnormal {}'.format(mu_1))
        print('The std of abnormal {}'.format(sigma_1))
        # n, bins, patches = plt.hist(message_all, bins=30, normed=1, label=['Normal', 'Abnormal'])
        # n, bins, patches = plt.hist(message_all, bins=30, normed=1, label=['Normal', 'Abnormal'])
        n, bins, patches = plt.hist(message_all, bins=30, density=True, label=['Normal', 'Abnormal'])

        # y_0 = mlab.normpdf(bins, mu_0, sigma_0)
        # y_1 = mlab.normpdf(bins, mu_1, sigma_1)

        # 使用 scipy.stats.norm.pdf 替代 mlab.normpdf
        y_0 = norm.pdf(bins, loc=mu_0, scale=sigma_0)  # 正态分布拟合
        y_1 = norm.pdf(bins, loc=mu_1, scale=sigma_1)  # 正态分布拟合

        plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=7.5)
        plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=7.5)
        plt.yticks(fontsize=30)
        plt.xticks(fontsize=30)
        plt.legend(loc='upper left', fontsize=30)
        plt.title(''.format(dataset), fontsize=25)
        plt.savefig('pdf/{}-RA.pdf'.format(dataset), format="pdf")
        # plt.show()


# def GCTAM_draw_pdf(message, ano_label, dataset, message_name):
#     normal_message_all = message[ano_label == 0]
#     abnormal_message_all = message[ano_label == 1]
#     message_all = [normal_message_all, abnormal_message_all]
#
#     mu_0 = np.mean(message_all[0])
#     sigma_0 = np.std(message_all[0])
#     mu_1 = np.mean(message_all[1])
#     sigma_1 = np.std(message_all[1])
#
#     print(f'The mean of normal: {mu_0}')
#     print(f'The std of normal: {sigma_0}')
#     print(f'The mean of abnormal: {mu_1}')
#     print(f'The std of abnormal: {sigma_1}')
#
#     # 绘制直方图
#     n, bins, patches = plt.hist(message_all, bins=30, density=True, label=['Normal', 'Abnormal'], alpha=0.6,
#                                 edgecolor='black')
#
#     # 计算拟合正态分布
#     y_0 = norm.pdf(bins, loc=mu_0, scale=sigma_0)
#     y_1 = norm.pdf(bins, loc=mu_1, scale=sigma_1)
#
#     # 绘制正态分布拟合曲线
#     plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=2)
#     plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=2)
#
#     # 计算总面积并添加注释
#     total_area = np.sum(np.diff(bins) * n)
#     plt.annotate(f'Total Area = {total_area:.2f}', xy=(0.6, np.max(n) - 0.5), fontsize=15, color='red')
#
#     # 调整显示
#     plt.yticks(fontsize=12)
#     plt.xticks(fontsize=12)
#     plt.legend(loc='upper left', fontsize=12)
#     plt.title(f'{dataset} - {message_name} Distribution', fontsize=15)
#
#     # 保存图像
#     plt.savefig(f'pdf/{dataset}_{message_name}_GCTAM.pdf', format="pdf")
#     plt.close()

def GCTAM_draw_pdf_new(message, ano_label, dataset, message_name):
    normal_message_all = message[ano_label == 0]
    abnormal_message_all = message[ano_label == 1]
    message_all = [normal_message_all, abnormal_message_all]

    mu_0 = np.mean(message_all[0])
    sigma_0 = np.std(message_all[0])
    mu_1 = np.mean(message_all[1])
    sigma_1 = np.std(message_all[1])

    print(f'The mean of normal: {mu_0}')
    print(f'The std of normal: {sigma_0}')
    print(f'The mean of abnormal: {mu_1}')
    print(f'The std of abnormal: {sigma_1}')

    # 绘制直方图（返回 n 为 2x30 矩阵）
    n, bins, patches = plt.hist(message_all, bins=30, density=True, label=['Normal', 'Abnormal'], alpha=0.6,
                                edgecolor='black')

    # 计算区间宽度
    bin_width = np.diff(bins)[0]

    # 计算区间概率 (2x30 ndarray)
    probabilities = n * bin_width

    # 绘制正态分布曲线
    y_0 = norm.pdf(bins, loc=mu_0, scale=sigma_0)
    y_1 = norm.pdf(bins, loc=mu_1, scale=sigma_1)
    plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=2, label='Normal Fit')
    plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=2, label='Abnormal Fit')

    # 在柱子顶部显示每个区间的概率
    for idx, label in enumerate(['Normal', 'Abnormal']):
        for i in range(len(n[idx])):  # 遍历每个区间
            probability = probabilities[idx, i]  # 取出该类别对应区间的概率值
            if n[idx, i] > 0:  # 只对有数据的柱子进行标注
                plt.text(
                    bins[i] + bin_width / 2,  # 横坐标为柱子中心
                    n[idx, i],  # 纵坐标为柱子高度
                    f'{probability:.2%}',  # 格式化为百分比
                    ha='center',
                    fontsize=10,
                    color='black'
                )

    # 添加横纵坐标和标题
    plt.xlabel('Affinity', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='upper left', fontsize=12)
    plt.title(f'{dataset} - {message_name} Affinity Distribution', fontsize=15)

    # 保存图像
    plt.savefig(f'pdf/{dataset}_{message_name}_GCTAM.pdf', format="pdf")
    plt.close()


def GCTAM_draw_pdf_new_1(message, ano_label, dataset, message_name):
    # 分离 Normal 和 Abnormal 数据
    normal_message_all = message[ano_label == 0]
    abnormal_message_all = message[ano_label == 1]
    message_all = [normal_message_all, abnormal_message_all]

    # 计算每类数据的均值和标准差
    mu_0 = np.mean(message_all[0])
    sigma_0 = np.std(message_all[0])
    mu_1 = np.mean(message_all[1])
    sigma_1 = np.std(message_all[1])

    print(f'The mean of normal: {mu_0}')
    print(f'The std of normal: {sigma_0}')
    print(f'The mean of abnormal: {mu_1}')
    print(f'The std of abnormal: {sigma_1}')

    # 绘制直方图并返回频数 n
    n, bins, patches = plt.hist(
        message_all, bins=30, density=False, label=['Normal', 'Abnormal'], alpha=0.6, edgecolor='black'
    )

    # 计算区间宽度
    bin_width = bins[1] - bins[0]

    # 计算每类数据的样本总数
    total_samples = np.sum(n, axis=1)  # 每类的总样本数

    # 将频数转换为比例（百分比）
    percentages = (n / total_samples[:, None]) * 100  # 按列归一化并转化为百分比

    # 清空当前图表，重新绘制柱状图
    plt.clf()
    for idx in range(len(message_all)):
        plt.bar(
            bins[:-1] + (bin_width / 2),  # 柱子的中心点
            percentages[idx],  # 使用百分比作为高度
            width=bin_width * 0.9,  # 调整柱子宽度
            align='center',
            alpha=0.6,
            edgecolor='black',
            label=['Normal', 'Abnormal'][idx]
        )

    # 绘制正态分布曲线（调整为百分比单位）
    y_0 = (norm.pdf(bins, loc=mu_0, scale=sigma_0) * bin_width) * 100  # 调整为百分比
    y_1 = (norm.pdf(bins, loc=mu_1, scale=sigma_1) * bin_width) * 100  # 调整为百分比
    plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=2, label='Normal Fit')
    plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=2, label='Abnormal Fit')

    # 添加横纵坐标和标题
    plt.xlabel('Affinity', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)  # 纵坐标为百分比
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 将纵坐标值转换为百分比格式
    y_ticks = plt.gca().get_yticks()
    plt.gca().set_yticklabels([f'{tick:.0f}%' for tick in y_ticks])  # 转换为百分比

    # 添加图例和标题
    plt.legend(loc='upper left', fontsize=12)
    plt.title(f'{dataset} - {message_name} Affinity Distribution', fontsize=15)

    # 保存图像
    plt.savefig(f'pdf/{dataset}_{message_name}_GCTAM_percentage_fixed.pdf', format="pdf")
    plt.close()


def GCTAM_draw_pdf_separated(message, ano_label, dataset, message_name):
    # 分离 Normal 和 Abnormal 数据
    normal_message_all = message[ano_label == 0]
    abnormal_message_all = message[ano_label == 1]
    message_all = [normal_message_all, abnormal_message_all]

    # 计算每类数据的均值和标准差
    mu_0 = np.mean(message_all[0])
    sigma_0 = np.std(message_all[0])
    mu_1 = np.mean(message_all[1])
    sigma_1 = np.std(message_all[1])

    print(f'The mean of normal: {mu_0}')
    print(f'The std of normal: {sigma_0}')
    print(f'The mean of abnormal: {mu_1}')
    print(f'The std of abnormal: {sigma_1}')

    # 绘制直方图并返回频数 n
    n, bins, patches = plt.hist(
        message_all, bins=30, density=False, label=['Normal', 'Abnormal'], alpha=0.0
    )  # 使用 alpha=0.0 隐藏原始直方图

    # 计算区间宽度
    bin_width = bins[1] - bins[0]

    # 计算每类数据的样本总数
    total_samples = np.sum(n, axis=1)  # 每类的总样本数

    # 将频数转换为比例（百分比）
    percentages = (n / total_samples[:, None]) * 100  # 按列归一化并转化为百分比

    # 找到正常节点超过异常节点的柱子，累加正常节点的百分比
    normal_above_abnormal_percentage = 0
    for i in range(len(percentages[0])):
        if percentages[0][i] > percentages[1][i]:  # 如果正常节点百分比 > 异常节点百分比
            normal_above_abnormal_percentage += percentages[0][i]  # 累加正常节点百分比

    # 计算正常节点总百分比
    total_normal_percentage = np.sum(percentages[0])
    proportion_above = (normal_above_abnormal_percentage / total_normal_percentage) * 100

    print(f'Total percentage of normal nodes above abnormal: {normal_above_abnormal_percentage:.2f}%')
    print(f'Proportion of normal nodes above abnormal relative to total normal: {proportion_above:.2f}%')

    # 清空当前图表，重新绘制分离的柱状图
    plt.clf()
    for idx in range(len(message_all)):
        x_offset = -bin_width / 4 if idx == 0 else bin_width / 4  # 分开显示的偏移量
        plt.bar(
            bins[:-1] + (bin_width / 2) + x_offset,  # 添加偏移量
            percentages[idx],  # 使用百分比作为高度
            width=bin_width * 0.5,  # 调整柱子宽度（较窄以便分开）
            align='center',
            alpha=0.6,
            edgecolor='black',
            color=['steelblue', 'darkorange'][idx],  # 分别设置不同的颜色
            label=['Normal', 'Abnormal'][idx]
        )

    # 绘制正态分布曲线（调整为百分比单位）
    y_0 = (norm.pdf(bins, loc=mu_0, scale=sigma_0) * bin_width) * 100  # 调整为百分比
    y_1 = (norm.pdf(bins, loc=mu_1, scale=sigma_1) * bin_width) * 100  # 调整为百分比

    # plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=2, label='Normal Fit')
    plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=2)
    # plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=2, label='Abnormal Fit')
    plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=2)

    # 添加横纵坐标和标题
    plt.xlabel('Affinity', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)  # 纵坐标为百分比
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 将纵坐标值转换为百分比格式
    y_ticks = plt.gca().get_yticks()
    plt.gca().set_yticklabels([f'{tick:.0f}%' for tick in y_ticks])  # 转换为百分比

    # 在图中标注总百分比信息
    plt.text(
        0.5,  # 横坐标
        max(np.max(percentages[0]), np.max(percentages[1])) * 0.8,  # 纵坐标
        f'Normal above Abnormal: {normal_above_abnormal_percentage:.2f}%\n',
        # f'Relative Proportion: {proportion_above:.2f}%',
        color='black',
        fontsize=10,
        ha='center',
        bbox=dict(facecolor='white', edgecolor='black', alpha=0.7)
    )

    # 添加图例和标题
    plt.legend(loc='upper left', fontsize=12)
    # plt.title(f'{dataset} - {message_name} Affinity Distribution', fontsize=15)

    # 保存图像
    plt.savefig(f'pdf/{dataset}_{message_name}_GCTAM_separated.pdf', format="pdf")
    plt.close()


def GCTAM_draw_pdf_separated_1(message, ano_label, dataset, message_name):
    # 分离 Normal 和 Abnormal 数据
    normal_message_all = message[ano_label == 0]
    abnormal_message_all = message[ano_label == 1]
    message_all = [normal_message_all, abnormal_message_all]

    # 计算每类数据的均值和标准差
    mu_0 = np.mean(message_all[0])
    sigma_0 = np.std(message_all[0])
    mu_1 = np.mean(message_all[1])
    sigma_1 = np.std(message_all[1])

    print(f'The mean of normal: {mu_0}')
    print(f'The std of normal: {sigma_0}')
    print(f'The mean of abnormal: {mu_1}')
    print(f'The std of abnormal: {sigma_1}')

    # 绘制直方图并返回频数 n
    n, bins, patches = plt.hist(
        message_all, bins=30, density=False, label=['Normal', 'Abnormal'], alpha=0.0
    )  # 使用 alpha=0.0 隐藏原始直方图

    # 计算区间宽度
    bin_width = bins[1] - bins[0]

    # 计算每类数据的样本总数
    total_samples = np.sum(n, axis=1)  # 每类的总样本数

    # 将频数转换为比例（百分比）
    percentages = (n / total_samples[:, None]) * 100  # 按列归一化并转化为百分比

    # 条件筛选并计算插值：Normal 百分比 > Abnormal 百分比
    conditioned_interpolation_values = []
    for i in range(len(percentages[0])):
        if percentages[0][i] > percentages[1][i]:  # 如果 Normal 百分比 > Abnormal 百分比
            interpolation_value = percentages[0][i] - percentages[1][i]
            conditioned_interpolation_values.append(interpolation_value)

    # 累加插值
    total_conditioned_interpolation = np.sum(conditioned_interpolation_values)

    print(f'Total conditioned interpolation value (Normal > Abnormal): {total_conditioned_interpolation:.2f}%')

    # 清空当前图表，重新绘制分离的柱状图
    plt.clf()
    for idx in range(len(message_all)):
        x_offset = -bin_width / 4 if idx == 0 else bin_width / 4  # 分开显示的偏移量
        plt.bar(
            bins[:-1] + (bin_width / 2) + x_offset,  # 添加偏移量
            percentages[idx],  # 使用百分比作为高度
            width=bin_width * 0.5,  # 调整柱子宽度（较窄以便分开）
            align='center',
            alpha=0.6,
            edgecolor='black',
            color=['steelblue', 'darkorange'][idx],  # 分别设置不同的颜色
            label=['Normal', 'Abnormal'][idx]
        )

    # 绘制正态分布曲线（调整为百分比单位）
    y_0 = (norm.pdf(bins, loc=mu_0, scale=sigma_0) * bin_width) * 100  # 调整为百分比
    y_1 = (norm.pdf(bins, loc=mu_1, scale=sigma_1) * bin_width) * 100  # 调整为百分比

    # plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=2, label='Normal Fit')
    # plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=2, label='Abnormal Fit')

    plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=2)
    plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=2)

    # 添加横纵坐标和标题
    # plt.xlabel('Affinity', fontsize=12)
    # plt.ylabel('Percentage (%)', fontsize=12)  # 纵坐标为百分比

    plt.ylabel('Percentage of Nodes (%)', fontsize=16)  # 纵坐标为百分比
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # 将纵坐标值转换为百分比格式
    y_ticks = plt.gca().get_yticks()
    plt.gca().set_yticklabels([f'{tick:.0f}%' for tick in y_ticks])  # 转换为百分比

    # 在图中标注累计插值结果
    plt.text(
        0.5,  # 横坐标
        max(np.max(percentages[0]), np.max(percentages[1])) * 0.7,  # 纵坐标
        f'Affinity Superiority \n Percentage: {total_conditioned_interpolation:.2f}%',
        color='black',
        fontsize=20,
        ha='center',
        bbox=dict(facecolor='white', edgecolor='black', alpha=0.7)
    )

    # 添加图例和标题
    plt.legend(loc='upper left', fontsize=18)
    # plt.title(f'{dataset} - {message_name} Affinity Distribution', fontsize=15)

    # 保存图像
    plt.savefig(f'pdf/{dataset}_{message_name}_GCTAM_separated.pdf', format="pdf")
    plt.close()


def GCTAM_draw_pdf_separated_2(message, ano_label, dataset, message_name):
    # 分离 Normal 和 Abnormal 数据
    normal_message_all = message[ano_label == 0]
    abnormal_message_all = message[ano_label == 1]
    message_all = [normal_message_all, abnormal_message_all]

    # 计算每类数据的均值和标准差
    mu_0 = np.mean(message_all[0])
    sigma_0 = np.std(message_all[0])
    mu_1 = np.mean(message_all[1])
    sigma_1 = np.std(message_all[1])

    print(f'The mean of normal: {mu_0}')
    print(f'The std of normal: {sigma_0}')
    print(f'The mean of abnormal: {mu_1}')
    print(f'The std of abnormal: {sigma_1}')

    # 绘制直方图并返回频数 n
    n, bins, patches = plt.hist(
        message_all, bins=30, density=False, label=['Normal', 'Abnormal'], alpha=0.0
    )  # 使用 alpha=0.0 隐藏原始直方图

    # 计算区间宽度
    bin_width = bins[1] - bins[0]

    # 计算每类数据的样本总数
    total_samples = np.sum(n, axis=1)  # 每类的总样本数

    # 将频数转换为比例（百分比）
    percentages = (n / total_samples[:, None]) * 100  # 按列归一化并转化为百分比

    # 条件筛选并计算插值：Normal 百分比 > Abnormal 百分比
    conditioned_interpolation_values = []
    for i in range(len(percentages[0])):
        if percentages[0][i] > percentages[1][i]:  # 如果 Normal 百分比 > Abnormal 百分比
            interpolation_value = percentages[0][i] - percentages[1][i]
            conditioned_interpolation_values.append(interpolation_value)

    # 累加插值
    total_conditioned_interpolation = np.sum(conditioned_interpolation_values)

    print(f'Total conditioned interpolation value (Normal > Abnormal): {total_conditioned_interpolation:.2f}%')

    # 清空当前图表，重新绘制分离的柱状图
    plt.clf()
    for idx in range(len(message_all)):
        x_offset = -bin_width / 4 if idx == 0 else bin_width / 4  # 分开显示的偏移量
        plt.bar(
            bins[:-1] + (bin_width / 2) + x_offset,  # 添加偏移量
            percentages[idx],  # 使用百分比作为高度
            width=bin_width * 0.5,  # 调整柱子宽度（较窄以便分开）
            align='center',
            alpha=0.6,
            edgecolor='black',
            color=['steelblue', 'darkorange'][idx],  # 分别设置不同的颜色
            label=['Normal', 'Abnormal'][idx]
        )

    # 绘制正态分布曲线（调整为百分比单位）
    y_0 = (norm.pdf(bins, loc=mu_0, scale=sigma_0) * bin_width) * 100  # 调整为百分比
    y_1 = (norm.pdf(bins, loc=mu_1, scale=sigma_1) * bin_width) * 100  # 调整为百分比

    # plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=2, label='Normal Fit')
    # plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=2, label='Abnormal Fit')

    plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=2)
    plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=2)

    # 添加横纵坐标和标题
    # plt.xlabel('Affinity', fontsize=12)
    # plt.ylabel('Percentage (%)', fontsize=12)  # 纵坐标为百分比

    plt.ylabel('Percentage of Nodes (%)', fontsize=16)  # 纵坐标为百分比
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # 将纵坐标值转换为百分比格式
    y_ticks = plt.gca().get_yticks()
    plt.gca().set_yticklabels([f'{tick:.0f}%' for tick in y_ticks])  # 转换为百分比

    # 在图中标注累计插值结果
    plt.text(
        0.5,  # 横坐标
        max(np.max(percentages[0]), np.max(percentages[1])) * 0.7,  # 纵坐标
        f'Affinity Superiority \n Percentage: {total_conditioned_interpolation:.2f}%',
        color='black',
        fontsize=20,
        ha='center',
        bbox=dict(facecolor='white', edgecolor='black', alpha=0.7)
    )

    # 添加图例和标题
    plt.legend(loc='upper left', fontsize=18)
    # plt.title(f'{dataset} - {message_name} Affinity Distribution', fontsize=15)

    # 保存图像
    plt.savefig(f'pdf/{dataset}_{message_name}_GCTAM_separated.pdf', format="pdf")
    plt.close()


def GCTAM_draw_pdf_separated_1(message, ano_label, dataset, message_name):
    # 分离 Normal 和 Abnormal 数据
    normal_message_all = message[ano_label == 0]
    abnormal_message_all = message[ano_label == 1]
    message_all = [normal_message_all, abnormal_message_all]

    # 计算每类数据的均值和标准差
    mu_0 = np.mean(message_all[0])
    sigma_0 = np.std(message_all[0])
    mu_1 = np.mean(message_all[1])
    sigma_1 = np.std(message_all[1])

    print(f'The mean of normal: {mu_0}')
    print(f'The std of normal: {sigma_0}')
    print(f'The mean of abnormal: {mu_1}')
    print(f'The std of abnormal: {sigma_1}')

    # 绘制直方图并返回频数 n
    n, bins, patches = plt.hist(
        message_all, bins=30, density=False, label=['Normal', 'Abnormal'], alpha=0.0
    )  # 使用 alpha=0.0 隐藏原始直方图

    # 计算区间宽度
    bin_width = bins[1] - bins[0]

    # 计算每类数据的样本总数
    total_samples = np.sum(n, axis=1)  # 每类的总样本数

    # 将频数转换为比例（百分比）
    percentages = (n / total_samples[:, None]) * 100  # 按列归一化并转化为百分比

    # 条件筛选并计算插值：Normal 百分比 > Abnormal 百分比
    conditioned_interpolation_values = []
    for i in range(len(percentages[0])):
        if percentages[0][i] > percentages[1][i]:  # 如果 Normal 百分比 > Abnormal 百分比
            interpolation_value = percentages[0][i] - percentages[1][i]
            conditioned_interpolation_values.append(interpolation_value)

    # 累加插值
    total_conditioned_interpolation = np.sum(conditioned_interpolation_values)

    print(f'Total conditioned interpolation value (Normal > Abnormal): {total_conditioned_interpolation:.2f}%')

    # 清空当前图表，重新绘制分离的柱状图
    plt.clf()
    for idx in range(len(message_all)):
        x_offset = -bin_width / 4 if idx == 0 else bin_width / 4  # 分开显示的偏移量
        plt.bar(
            bins[:-1] + (bin_width / 2) + x_offset,  # 添加偏移量
            percentages[idx],  # 使用百分比作为高度
            width=bin_width * 0.5,  # 调整柱子宽度（较窄以便分开）
            align='center',
            alpha=0.6,
            edgecolor='black',
            color=['steelblue', 'darkorange'][idx],  # 分别设置不同的颜色
            label=['Normal', 'Abnormal'][idx]
        )

    # 绘制正态分布曲线（调整为百分比单位）
    y_0 = (norm.pdf(bins, loc=mu_0, scale=sigma_0) * bin_width) * 100  # 调整为百分比
    y_1 = (norm.pdf(bins, loc=mu_1, scale=sigma_1) * bin_width) * 100  # 调整为百分比

    # plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=2, label='Normal Fit')
    # plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=2, label='Abnormal Fit')

    plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=2)
    plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=2)

    # 添加横纵坐标和标题
    # plt.xlabel('Affinity', fontsize=12)
    # plt.ylabel('Percentage (%)', fontsize=12)  # 纵坐标为百分比

    plt.ylabel('Percentage of Nodes (%)', fontsize=16)  # 纵坐标为百分比
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # 将纵坐标值转换为百分比格式
    y_ticks = plt.gca().get_yticks()
    plt.gca().set_yticklabels([f'{tick:.0f}%' for tick in y_ticks])  # 转换为百分比

    # 在图中标注累计插值结果
    plt.text(
        0.5,  # 横坐标
        max(np.max(percentages[0]), np.max(percentages[1])) * 0.7,  # 纵坐标
        f'Affinity Superiority \n Percentage: {total_conditioned_interpolation:.2f}%',
        color='black',
        fontsize=20,
        ha='center',
        bbox=dict(facecolor='white', edgecolor='black', alpha=0.7)
    )

    # 添加图例和标题
    plt.legend(loc='upper left', fontsize=18)
    # plt.title(f'{dataset} - {message_name} Affinity Distribution', fontsize=15)

    # 保存图像
    plt.savefig(f'pdf/{dataset}_{message_name}_GCTAM_separated.pdf', format="pdf")
    plt.close()


def GCTAM_draw_pdf_separated_1(message, ano_label, dataset, message_name):
    plt.rcParams['figure.dpi'] = 600
    # plt.rcParams['figure.figsize'] = (10, 7.5)
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.rcParams['font.size'] = 22  # 设置字体大小
    # plt.rcParams['figure.figsize'] = (9, 6)

    # 分离 Normal 和 Abnormal 数据
    normal_message_all = message[ano_label == 0]
    abnormal_message_all = message[ano_label == 1]
    message_all = [normal_message_all, abnormal_message_all]

    # 计算每类数据的均值和标准差
    mu_0 = np.mean(message_all[0])
    sigma_0 = np.std(message_all[0])
    mu_1 = np.mean(message_all[1])
    sigma_1 = np.std(message_all[1])

    print(f'The mean of normal: {mu_0}')
    print(f'The std of normal: {sigma_0}')
    print(f'The mean of abnormal: {mu_1}')
    print(f'The std of abnormal: {sigma_1}')

    # 绘制直方图并返回频数 n
    n, bins, patches = plt.hist(
        message_all, bins=30, density=False, label=['Normal', 'Abnormal'], alpha=0.0
    )  # 使用 alpha=0.0 隐藏原始直方图

    # 计算区间宽度
    bin_width = bins[1] - bins[0]

    # 计算每类数据的样本总数
    total_samples = np.sum(n, axis=1)  # 每类的总样本数

    # 将频数转换为比例（百分比）
    percentages = (n / total_samples[:, None]) * 100  # 按列归一化并转化为百分比

    # 条件筛选并计算插值：Normal 百分比 > Abnormal 百分比
    conditioned_interpolation_values = []
    for i in range(len(percentages[0])):
        if percentages[0][i] > percentages[1][i]:  # 如果 Normal 百分比 > Abnormal 百分比
            interpolation_value = percentages[0][i] - percentages[1][i]
            conditioned_interpolation_values.append(interpolation_value)

    # 累加插值
    total_conditioned_interpolation = np.sum(conditioned_interpolation_values)

    print(f'Total conditioned interpolation value (Normal > Abnormal): {total_conditioned_interpolation:.2f}%')

    # 清空当前图表，重新绘制分离的柱状图
    plt.clf()
    for idx in range(len(message_all)):
        x_offset = -bin_width / 4 if idx == 0 else bin_width / 4  # 分开显示的偏移量
        plt.bar(
            bins[:-1] + (bin_width / 2) + x_offset,  # 添加偏移量
            percentages[idx],  # 使用百分比作为高度
            width=bin_width * 0.5,  # 调整柱子宽度（较窄以便分开）
            align='center',
            alpha=0.6,
            edgecolor='black',
            color=['steelblue', 'darkorange'][idx],  # 分别设置不同的颜色
            label=['Normal', 'Abnormal'][idx]
        )

    # # # 在 Abnormal 柱状图上方叠加红色差值部分
    # for i in range(len(percentages[0])):
    #     if percentages[0][i] - percentages[1][i] > 5:  # 如果 Normal 百分比 > Abnormal 百分比
    #         # if percentages[0][i] - percentages[1][i] > 10:  # 如果 Normal 百分比 > Abnormal 百分比
    #         # 计算红色柱状图的高度（差值）
    #         overlay_height = percentages[0][i] - percentages[1][i]
    #         # 修正位置，与 Abnormal 对齐
    #         x_offset = -bin_width / 4 if idx == 0 else bin_width / 4  # 分开显示的偏移量
    #         x_position_arr = bins[:-1] + (bin_width / 2) + x_offset
    #         x_position = x_position_arr[i]
    #
    #         plt.bar(
    #             x_position,  # 与 Abnormal 的位置保持一致
    #             overlay_height,  # 差值作为高度
    #             bottom=percentages[1][i],  # 从 Abnormal 的顶部开始叠加
    #             width=bin_width * 0.5,  # 宽度与其他柱宽一致
    #             align='center',
    #             alpha=0.8,
    #             color='red',
    #             edgecolor='black',
    #             label='Truncation Value'
    #         )
    #
    # # 添加文本标签
    # plt.text(
    #     (bins[i] + 0.02 + bin_width / 2),
    #     # (bins[i] + bin_width / 2),
    #     # percentages[0][i] + 0.2,  # 在柱子顶部稍微上方
    #     percentages[0][i] + 1,  # 在柱子顶部稍微上方
    #     'Superior',
    #     color='red',
    #     fontsize=22,
    #     ha='center'
    # )
    # break

    # 绘制正态分布曲线（调整为百分比单位）
    y_0 = (norm.pdf(bins, loc=mu_0, scale=sigma_0) * bin_width) * 100  # 调整为百分比
    y_1 = (norm.pdf(bins, loc=mu_1, scale=sigma_1) * bin_width) * 100  # 调整为百分比

    # plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=2, label='Normal Fit')
    # plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=2, label='Abnormal Fit')

    plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=2)
    plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=2)

    # plt.xlim(0.3, 1.0)
    # plt.xlim(0.4, 1.0)
    plt.xlim(0.0, 1.0)
    # plt.xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=30)
    # plt.xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=30)

    # 添加横纵坐标和标题
    if message_name == "final_message":
        plt.xlabel('Ours Affinity', fontsize=22)
    # plt.ylabel('Percentage (%)', fontsize=12)  # 纵坐标为百分比
    elif message_name == "RA_message":
        plt.xlabel('Original Affinity', fontsize=22)
    else:
        plt.xlabel('TAM-Based Affinity', fontsize=22)

    plt.ylabel('Percentage of Nodes (%)', fontsize=22)  # 纵坐标为百分比

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    # 将纵坐标值转换为百分比格式
    y_ticks = plt.gca().get_yticks()
    plt.gca().set_yticklabels([f'{tick:.0f}%' for tick in y_ticks])  # 转换为百分比

    # # 在图中标注累计插值结果
    # plt.text(
    #     0.6,  # 横坐标
    #     max(np.max(percentages[0]), np.max(percentages[1])) * 0.8,  # 纵坐标
    #     f'Affinity Truncation Value: {total_conditioned_interpolation:.2f}%',
    #     color='black',
    #     fontsize=20,
    #     ha='center',
    #     bbox=dict(facecolor='white', edgecolor='black', alpha=0.7)
    # )

    # 添加图例和标题
    plt.legend(loc='upper left', fontsize=22)
    plt.title(f'{dataset}', fontsize=22)

    # 保存图像
    plt.savefig(f'pdf/{dataset}_{message_name}_GCTAM_separated.pdf', format="pdf")
    plt.close()


def GCTAM_draw_pdf_separated_with_highlight(message, ano_label, dataset, message_name):
    # 分离 Normal 和 Abnormal 数据
    normal_message_all = message[ano_label == 0]
    abnormal_message_all = message[ano_label == 1]
    message_all = [normal_message_all, abnormal_message_all]

    # 计算每类数据的均值和标准差
    mu_0 = np.mean(message_all[0])
    sigma_0 = np.std(message_all[0])
    mu_1 = np.mean(message_all[1])
    sigma_1 = np.std(message_all[1])

    print(f'The mean of normal: {mu_0}')
    print(f'The std of normal: {sigma_0}')
    print(f'The mean of abnormal: {mu_1}')
    print(f'The std of abnormal: {sigma_1}')

    # 绘制直方图并返回频数 n
    n, bins, patches = plt.hist(
        message_all, bins=30, density=False, label=['Normal', 'Abnormal'], alpha=0.0
    )  # 使用 alpha=0.0 隐藏原始直方图

    # 计算区间宽度
    bin_width = bins[1] - bins[0]

    # 计算每类数据的样本总数
    total_samples = np.sum(n, axis=1)  # 每类的总样本数

    # 将频数转换为比例（百分比）
    percentages = (n / total_samples[:, None]) * 100  # 按列归一化并转化为百分比

    # 条件筛选并计算插值：Normal 百分比 > Abnormal 百分比
    total_conditioned_interpolation = 0
    for i in range(len(percentages[0])):
        if percentages[0][i] > percentages[1][i]:  # 如果 Normal 百分比 > Abnormal 百分比
            total_conditioned_interpolation += (percentages[0][i] - percentages[1][i])

    print(f'Total conditioned interpolation value (Normal > Abnormal): {total_conditioned_interpolation:.2f}%')

    # 清空当前图表，重新绘制分离的柱状图
    plt.clf()
    for idx in range(len(message_all)):
        x_offset = -bin_width / 3 if idx == 0 else bin_width / 3  # 分开显示的偏移量，留出缝隙
        plt.bar(
            bins[:-1] + (bin_width / 2) + x_offset,  # 添加偏移量
            percentages[idx],  # 使用百分比作为高度
            width=bin_width * 0.4,  # 调整柱子宽度（较窄以便分开）
            align='center',
            alpha=0.6,
            edgecolor='black',
            color=['steelblue', 'darkorange'][idx],  # 分别设置不同的颜色
            label=['Normal', 'Abnormal'][idx]
        )

    # 绘制满足条件的红色柱状图
    for i in range(len(percentages[0])):
        if percentages[0][i] > percentages[1][i]:  # 如果 Normal 百分比 > Abnormal 百分比
            overlay_height = percentages[0][i] - percentages[1][i]
            plt.bar(
                bins[i] + bin_width / 2 + x_offset,  # 不偏移，绘制在中心
                # percentages[0][i],  # 使用 Normal 百分比作为高度
                overlay_height,
                bottom=percentages[1][i],  # 从 Abnormal 的顶部开始叠加
                width=bin_width * 0.4,  # 与其他柱宽一致
                align='center',
                alpha=0.8,
                color='red',
                edgecolor='black',
                label='Superior' if i == 0 else ""  # 添加标签，仅第一次绘制时添加图例
            )

    # 绘制正态分布曲线（调整为百分比单位）
    y_0 = (norm.pdf(bins, loc=mu_0, scale=sigma_0) * bin_width) * 100  # 调整为百分比
    y_1 = (norm.pdf(bins, loc=mu_1, scale=sigma_1) * bin_width) * 100  # 调整为百分比
    plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=2, label='Normal Fit')
    plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=2, label='Abnormal Fit')

    # 添加横纵坐标和标题
    plt.xlabel('Affinity', fontsize=14)
    plt.ylabel('Percentage of Nodes (%)', fontsize=16)  # 纵坐标为百分比
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 将纵坐标值转换为百分比格式
    y_ticks = plt.gca().get_yticks()
    plt.gca().set_yticklabels([f'{tick:.0f}%' for tick in y_ticks])  # 转换为百分比

    # 在图中标注累计插值结果
    plt.text(
        0.5,  # 横坐标
        max(np.max(percentages[0]), np.max(percentages[1])) * 0.7,  # 纵坐标
        f'Affinity Superiority \n Percentage: {total_conditioned_interpolation:.2f}%',
        color='black',
        fontsize=12,
        ha='center',
        bbox=dict(facecolor='white', edgecolor='black', alpha=0.7)
    )

    # 添加图例和标题
    plt.legend(loc='upper left', fontsize=12)
    plt.title(f'{dataset} - {message_name} Affinity Distribution', fontsize=15)

    # 保存图像
    plt.savefig(f'pdf/{dataset}_{message_name}_GCTAM_separated_highlight.pdf', format="pdf")
    plt.close()


def GCTAM_draw_pdf_separated_with_highlight_1(message, ano_label, dataset, message_name):
    # 分离 Normal 和 Abnormal 数据
    normal_message_all = message[ano_label == 0]
    abnormal_message_all = message[ano_label == 1]
    message_all = [normal_message_all, abnormal_message_all]

    # 计算每类数据的均值和标准差
    mu_0 = np.mean(message_all[0])
    sigma_0 = np.std(message_all[0])
    mu_1 = np.mean(message_all[1])
    sigma_1 = np.std(message_all[1])

    print(f'The mean of normal: {mu_0}')
    print(f'The std of normal: {sigma_0}')
    print(f'The mean of abnormal: {mu_1}')
    print(f'The std of abnormal: {sigma_1}')

    # 绘制直方图并返回频数 n
    n, bins, patches = plt.hist(
        message_all, bins=30, density=False, label=['Normal', 'Abnormal'], alpha=0.0
    )  # 使用 alpha=0.0 隐藏原始直方图

    # 计算区间宽度
    bin_width = bins[1] - bins[0]

    # 计算每类数据的样本总数
    total_samples = np.sum(n, axis=1)  # 每类的总样本数

    # 将频数转换为比例（百分比）
    percentages = (n / total_samples[:, None]) * 100  # 按列归一化并转化为百分比

    # 条件筛选并计算插值：Normal 百分比 > Abnormal 百分比
    total_conditioned_interpolation = 0
    for i in range(len(percentages[0])):
        if percentages[0][i] > percentages[1][i]:  # 如果 Normal 百分比 > Abnormal 百分比
            total_conditioned_interpolation += (percentages[0][i] - percentages[1][i])

    print(f'Total conditioned interpolation value (Normal > Abnormal): {total_conditioned_interpolation:.2f}%')

    # 清空当前图表，重新绘制分离的柱状图
    plt.clf()
    for idx in range(len(message_all)):
        # x_offset = -bin_width / 3 if idx == 0 else bin_width / 3  # 分开显示的偏移量，留出缝隙
        x_offset = -bin_width / 3 if idx == 0 else bin_width / 3  # 分开显示的偏移量，留出缝隙
        plt.bar(
            bins[:-1] + (bin_width / 2) + x_offset,  # 添加偏移量
            percentages[idx],  # 使用百分比作为高度
            width=bin_width * 0.4,  # 调整柱子宽度（较窄以便分开）
            align='center',
            alpha=0.6,
            edgecolor='black',
            color=['steelblue', 'darkorange'][idx],  # 分别设置不同的颜色
            label=['Normal', 'Abnormal'][idx]
        )

    # 在 Abnormal 柱状图上方叠加红色差值部分
    for i in range(len(percentages[0])):
        if percentages[0][i] - percentages[1][i] > 10:  # 如果 Normal 百分比 > Abnormal 百分比
            # 计算红色柱状图的高度（差值）
            overlay_height = percentages[0][i] - percentages[1][i]
            # 修正位置，与 Abnormal 对齐
            x_position = bins[i] + (bin_width / 2) + bin_width / 3
            plt.bar(
                x_position,  # 与 Abnormal 的位置保持一致
                overlay_height,  # 差值作为高度
                # overlay_height - 5,  # 差值作为高度
                bottom=percentages[1][i],  # 从 Abnormal 的顶部开始叠加
                width=bin_width * 0.4,  # 宽度与其他柱宽一致
                align='center',
                alpha=0.8,
                color='red',
                edgecolor='black'
            )

            # 添加文本标签
            plt.text(
                bins[i] + bin_width / 2,
                percentages[0][i] + 0.5,  # 在柱子顶部稍微上方
                'Superior',
                color='red',
                fontsize=10,
                ha='center'
            )

            break

    # 绘制正态分布曲线（调整为百分比单位）
    y_0 = (norm.pdf(bins, loc=mu_0, scale=sigma_0) * bin_width) * 100  # 调整为百分比
    y_1 = (norm.pdf(bins, loc=mu_1, scale=sigma_1) * bin_width) * 100  # 调整为百分比
    plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=2, label='Normal Fit')
    plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=2, label='Abnormal Fit')

    # 添加横纵坐标和标题
    plt.xlabel('Affinity', fontsize=14)
    plt.ylabel('Percentage of Nodes (%)', fontsize=16)  # 纵坐标为百分比
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 将纵坐标值转换为百分比格式
    y_ticks = plt.gca().get_yticks()
    plt.gca().set_yticklabels([f'{tick:.0f}%' for tick in y_ticks])  # 转换为百分比

    # 在图中标注累计插值结果
    plt.text(
        0.5,  # 横坐标
        max(np.max(percentages[0]), np.max(percentages[1])) * 0.7,  # 纵坐标
        f'Affinity Superiority \n Percentage: {total_conditioned_interpolation:.2f}%',
        color='black',
        fontsize=12,
        ha='center',
        bbox=dict(facecolor='white', edgecolor='black', alpha=0.7)
    )

    # 添加图例和标题
    plt.legend(loc='upper left', fontsize=12)
    plt.title(f'{dataset} - {message_name} Affinity Distribution', fontsize=15)

    # 保存图像
    plt.savefig(f'pdf/{dataset}_{message_name}_GCTAM_separated_highlight.pdf', format="pdf")
    plt.close()


def GCTAM_draw_pdf(message, ano_label, dataset, message_name):
    # with PdfPages('{}{}-GCTAM.pdf'.format(dataset, message_name)) as pdf:
    # with PdfPages('pdf/{}_{}_GCTAM.pdf'.format(dataset, message_name)) as pdf:
    normal_message_all = message[ano_label == 0]
    abnormal_message_all = message[ano_label == 1]
    message_all = [normal_message_all, abnormal_message_all]
    mu_0 = np.mean(message_all[0])
    sigma_0 = np.std(message_all[0])
    print('The mean of normal {}'.format(mu_0))
    print('The std of normal {}'.format(sigma_0))
    mu_1 = np.mean(message_all[1])
    sigma_1 = np.std(message_all[1])
    print('The mean of abnormal {}'.format(mu_1))
    print('The std of abnormal {}'.format(sigma_1))
    # n, bins, patches = plt.hist(message_all, bins=30, normed=1, label=['Normal', 'Abnormal'])
    # n, bins, patches = plt.hist(message_all, bins=30, normed=1, label=['Normal', 'Abnormal'])

    n, bins, patches = plt.hist(message_all, bins=30, density=True, label=['Normal', 'Abnormal'])

    # n, bins, patches = plt.hist(message_all, bins=10, density=True, label=['Normal', 'Abnormal'])
    # n, bins, patches = plt.hist(message_all, bins=10, density=True, label=['Normal', 'Abnormal'], alpha=0.6,
    #                             edgecolor='black')

    # y_0 = mlab.normpdf(bins, mu_0, sigma_0)
    # y_1 = mlab.normpdf(bins, mu_1, sigma_1)

    # 使用 scipy.stats.norm.pdf 替代 mlab.normpdf
    y_0 = norm.pdf(bins, loc=mu_0, scale=sigma_0)  # 正态分布拟合
    y_1 = norm.pdf(bins, loc=mu_1, scale=sigma_1)  # 正态分布拟合

    plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=7.5)
    plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=7.5)
    plt.yticks(fontsize=30)
    plt.xticks(fontsize=30)

    # plt.xlim(0.4, 1.0)
    # plt.xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=30)

    # plt.ylim(1, )  # 也可以自定义范围，例如 plt.xlim(0, 10)

    plt.legend(loc='upper left', fontsize=30)
    plt.title(''.format(dataset), fontsize=25)
    plt.savefig('pdf/{}_{}_GCTAM.pdf'.format(dataset, message_name), format="pdf")

    # plt.show()
    # plt.close(
    plt.close()

    # plt.clf()  # 清除当前图像内容


def draw_pdf(message, ano_label, dataset):
    with PdfPages('{}-TAM.pdf'.format(dataset)) as pdf:
        normal_message_all = message[ano_label == 0]
        abnormal_message_all = message[ano_label == 1]
        message_all = [normal_message_all, abnormal_message_all]
        mu_0 = np.mean(message_all[0])
        sigma_0 = np.std(message_all[0])
        print('The mean of normal {}'.format(mu_0))
        print('The std of normal {}'.format(sigma_0))
        mu_1 = np.mean(message_all[1])
        sigma_1 = np.std(message_all[1])
        print('The mean of abnormal {}'.format(mu_1))
        print('The std of abnormal {}'.format(sigma_1))
        # n, bins, patches = plt.hist(message_all, bins=30, normed=1, label=['Normal', 'Abnormal'])
        # n, bins, patches = plt.hist(message_all, bins=30, normed=1, label=['Normal', 'Abnormal'])
        n, bins, patches = plt.hist(message_all, bins=30, density=True, label=['Normal', 'Abnormal'])

        # y_0 = mlab.normpdf(bins, mu_0, sigma_0)
        # y_1 = mlab.normpdf(bins, mu_1, sigma_1)

        # 使用 scipy.stats.norm.pdf 替代 mlab.normpdf
        y_0 = norm.pdf(bins, loc=mu_0, scale=sigma_0)  # 正态分布拟合
        y_1 = norm.pdf(bins, loc=mu_1, scale=sigma_1)  # 正态分布拟合

        plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=7.5)
        plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=7.5)
        plt.yticks(fontsize=30)
        plt.xticks(fontsize=30)
        plt.legend(loc='upper left', fontsize=30)
        plt.title(''.format(dataset), fontsize=25)
        plt.savefig('pdf/{}-TAM.pdf'.format(dataset), format="pdf")
        # plt.show()


def draw_pdf_str_attr(message, ano_label, str_ano_label, attr_ano_label, dataset):
    with PdfPages('{}-TAM.pdf'.format(dataset)) as pdf:
        normal_message_all = message[ano_label == 0]
        str_abnormal_message_all = message[str_ano_label == 1]
        attr_abnormal_message_all = message[attr_ano_label == 1]
        message_all = [normal_message_all, str_abnormal_message_all, attr_abnormal_message_all]

        mu_0 = np.mean(message_all[0])
        sigma_0 = np.std(message_all[0])
        print('The mean of normal {}'.format(mu_0))
        print('The std of normal {}'.format(sigma_0))
        mu_1 = np.mean(message_all[1])
        sigma_1 = np.std(message_all[1])
        print('The mean of str_abnormal {}'.format(mu_1))
        print('The std of str_abnormal {}'.format(sigma_1))
        mu_2 = np.mean(message_all[2])
        sigma_2 = np.std(message_all[2])
        print('The mean of attt_abnormal {}'.format(mu_2))
        print('The std of attt_abnormal {}'.format(sigma_2))
        n, bins, patches = plt.hist(message_all, bins=30, normed=1,
                                    label=['Normal', 'Structural Abnormal', 'Contextual Abnormal'])
        y_0 = mlab.normpdf(bins, mu_0, sigma_0)
        y_1 = mlab.normpdf(bins, mu_1, sigma_1)
        y_2 = mlab.normpdf(bins, mu_2, sigma_2)  #

        plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=3.5)
        plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=3.5)
        plt.plot(bins, y_2, color='green', linestyle='--', linewidth=3.5)

        plt.xlabel('TAM-based Affinity', fontsize=25)
        plt.ylabel('Number of Samples', size=25)
        plt.yticks(fontsize=25)
        plt.xticks(fontsize=25)
        plt.legend(loc='upper left', fontsize=18)
        # plt.title('{}'.format(dataset), fontsize=25)
        plt.show()


def check_writable(path, overwrite=True):
    if not os.path.exists(path):
        os.makedirs(path)
    elif overwrite:
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        pass


def get_split(train_rate, valid_rate, test_rate, ano_label):
    assert abs(train_rate + valid_rate + test_rate - 1.0) < 1e-6, "Ratios must sum up to 1."
    train_idx = None
    valid_idx = None
    test_idx = None

    # 获取正常节点和异常节点的索引
    normal_idx = np.where(ano_label == 0)[0]
    anomaly_idx = np.where(ano_label == 1)[0]

    # 随机打乱索引
    np.random.shuffle(normal_idx)
    np.random.shuffle(anomaly_idx)

    # 计算每个集合的大小
    normal_train_size = int(len(normal_idx) * train_rate)
    normal_valid_size = int(len(normal_idx) * valid_rate)
    normal_test_size = len(normal_idx) - normal_train_size - normal_valid_size

    anomaly_train_size = int(len(anomaly_idx) * train_rate)
    anomaly_valid_size = int(len(anomaly_idx) * valid_rate)
    anomaly_test_size = len(anomaly_idx) - anomaly_train_size - anomaly_valid_size

    # 划分正常节点
    normal_train_idx = normal_idx[:normal_train_size]
    normal_valid_idx = normal_idx[normal_train_size:normal_train_size + normal_valid_size]
    normal_test_idx = normal_idx[normal_train_size + normal_valid_size:]

    # 划分异常节点
    anomaly_train_idx = anomaly_idx[:anomaly_train_size]
    anomaly_valid_idx = anomaly_idx[anomaly_train_size:anomaly_train_size + anomaly_valid_size]
    anomaly_test_idx = anomaly_idx[anomaly_train_size + anomaly_valid_size:]

    # 合并正常和异常节点
    train_idx = np.concatenate([normal_train_idx, anomaly_train_idx])
    valid_idx = np.concatenate([normal_valid_idx, anomaly_valid_idx])
    test_idx = np.concatenate([normal_test_idx, anomaly_test_idx])

    # 转换为Tensor
    train_idx = torch.tensor(train_idx)
    valid_idx = torch.tensor(valid_idx)
    test_idx = torch.tensor(test_idx)

    return train_idx, valid_idx, test_idx


def calculate_isomorphism_score_dgl(graph, y):
    """
    计算子图的同构性分数（使用DGL图结构）。

    参数:
    graph (dgl.DGLGraph): DGL图对象。
    y (torch.Tensor): 每个节点的标签，形状为 (n,)。

    返回:
    beta (float): 图的同构性分数。
    """
    V = graph.num_nodes()
    beta_sum = 0

    for v in range(V):
        neighbors = graph.successors(v).tolist()  # 获取节点 v 的所有邻居节点
        if len(neighbors) > 0:
            same_label_count = torch.sum(y[neighbors] == y[v]).item()
            beta_sum += same_label_count / len(neighbors)

    beta = beta_sum / V
    return beta
