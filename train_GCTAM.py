import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric import seed_everything

from model import Model, my_GCN, MultiHeadAttentionFusion, my_MLP
from utils import *
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_auc_score

from graph_generator import MLP_learner, GNN_learner

import os
import argparse
from tqdm import tqdm
import scipy.sparse as sp
import time


def start_train(args):
    local_time = time.localtime()
    args.lct = time.strftime("%Y_%m_%d_%H_%M_%S", local_time)
    if torch.cuda.is_available():
        print('Using CUDA')
        device = 'cuda:0'
    else:
        device = 'cpu'

    adj, features, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)
    raw_adj = adj.copy()
    raw_adj = (raw_adj + sp.eye(adj.shape[0])).todense()
    if args.dataset in ['Amazon', 'YelpChi', 'Amazon-all', 'YelpChi-all']:
        features, _ = preprocess_features(features)
        raw_features = features
    else:
        raw_features = features.todense()
        features = raw_features

    dgl_graph = adj_to_dgl_graph(adj).to(device)
    dgl_grap_with_self_loop = dgl.from_scipy(adj + sp.eye(adj.shape[0])).to(device)

    row, col = dgl_grap_with_self_loop.adj().to_dense().nonzero(as_tuple=True)  # 获取非零元素的行和列索引
    edges = (row, col)
    # 获取边的权重
    edge_weight = dgl_grap_with_self_loop.adj().to_dense()[row, col]  # 可以选择使用边权重
    # 创建 DGL 图
    dgl_grap_with_self_loop = dgl.graph(edges)
    dgl_grap_with_self_loop.edata["w"] = edge_weight
    ft_size = features.shape[1]
    features = torch.FloatTensor(features).to(device)

    model = my_GCN(ft_size, args.embedding_dim, args.embedding_dim).cuda()

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    global_affinity_truncation_model = MLP_learner(2, args.embedding_dim, args.k, "cosine", 6, True, "relu").cuda()

    optimiser_graph = torch.optim.AdamW(global_affinity_truncation_model.parameters(), lr=args.lr,
                                        weight_decay=args.weight_decay)

    dgl_grap_with_self_loop = dgl_grap_with_self_loop.to(device)
    original_raw_adj_with_loop = dgl_grap_with_self_loop.adj().to_dense().to(device)
    dgl_cut_adj = copy.deepcopy(dgl_grap_with_self_loop)
    dis_arr_dir = Path.cwd().joinpath("dis_array", args.dataset)
    os.makedirs(dis_arr_dir, exist_ok=True)
    dis_arr_file_path = os.path.join(dis_arr_dir, f"{args.dataset}.pt")

    if os.path.exists(dis_arr_file_path):
        dis_array_1 = torch.load(dis_arr_file_path, map_location="cpu")  # 加载文件
    else:
        dis_array_1 = calc_distance(original_raw_adj_with_loop, features).to(device)  # 计算距离矩阵
        torch.save(dis_array_1, dis_arr_file_path)  # 保存到本地

    num_edge = original_raw_adj_with_loop.sum()
    num_node = dgl_grap_with_self_loop.num_nodes()
    sim_arr_dir = Path.cwd().joinpath("sim_array", args.dataset)
    os.makedirs(sim_arr_dir, exist_ok=True)
    sim_arr_file_path = os.path.join(sim_arr_dir, f"{args.dataset}.pt")
    sim_adj = None

    if os.path.exists(sim_arr_file_path):
        sim_adj = torch.load(sim_arr_file_path, map_location="cpu")  # 加载文件
    else:
        sim_adj = calc_sim(dgl_cut_adj.adj().to_dense().to(device), features)  # 计算距离矩阵
        torch.save(sim_adj, sim_arr_file_path)  # 保存到本地

    cut_num_all = 0
    delete_rate = 0
    while delete_rate < args.delete_edge_rate:
        cut_adj = dgl_cut_adj.adj().to_dense().to(device)
        cut_adj, cut_num = graph_nsgt_2(dis_array_1, cut_adj, sim_adj, args)
        # cut_adj, cut_num = graph_nsgt_3(dis_array_1, cut_adj, sim_adj, args)
        cut_num_all += cut_num
        cut_adj = cut_adj.unsqueeze(0)
        delete_rate = cut_num_all / num_edge
        cut_adj = normalize_adj_tensor(cut_adj)
        row, col = cut_adj[0].nonzero(as_tuple=True)  # 获取非零元素的行和列索引
        edges = (row, col)
        edge_weight = cut_adj[0][row, col]  # 可以选择使用边权重
        dgl_cut_adj = dgl.graph(edges)
        dgl_cut_adj.edata["w"] = edge_weight
        dgl_cut_adj = dgl_cut_adj.to(device)

    for epoch in range(1, args.num_epoch + 1):
        model.train()
        global_affinity_truncation_model.train()
        node_emb_1, _, _ = model.forward(dgl_grap_with_self_loop, features)

        node_emb_2, feat1, feat2 = model.forward(dgl_cut_adj, features)
        # node_emb_3, _, _ = model.forward(dgl_grap_with_self_loop, features)
        learned_graph = global_affinity_truncation_model.forward(node_emb_1).to(device)
        # learned_graph = graph_learner.forward(node_emb_2).to(device)
        # learned_graph = graph_learner.forward(node_emb_2).to(device)
        node_emb_3, _, _ = model.forward(learned_graph, features)

        loss_1, message_sum_1 = max_message(node_emb_2, original_raw_adj_with_loop)
        loss_2, message_sum_2 = max_message(node_emb_2, dgl_cut_adj.adj().to_dense().to(device))
        loss_3, message_sum_3 = max_message(node_emb_2, learned_graph.adj().to_dense().to(device))

        # loss_1, message_sum_1 = max_message_2(node_emb_2, original_raw_adj_with_loop)
        # loss_2, message_sum_2 = max_message_2(node_emb_2, dgl_cut_adj.adj().to_dense().to(device))
        # loss_3, message_sum_3 = max_message_2(node_emb_2, learned_graph.adj().to_dense().to(device))

        loss = loss_1 + loss_2 + loss_3  # with A+B

        optimiser.zero_grad()
        optimiser_graph.zero_grad()

        loss.backward()
        optimiser.step()
        optimiser_graph.step()

        loss = loss.detach().cpu().numpy()

        if epoch % 50 == 0:
            print("mean_loss is {}".format(loss))

    message_list = []
    message_list.append(torch.unsqueeze(message_sum_1, 0))
    message_list.append(torch.unsqueeze(message_sum_2, 0))
    message_list.append(torch.unsqueeze(message_sum_3, 0))

    message_list = torch.mean(torch.cat(message_list), 0)
    message = np.array(message_list.cpu().detach())
    final_message = 1 - normalize_score(message)

    mes_1 = np.array(torch.squeeze(message_sum_1).cpu().detach())
    mes_1 = 1 - normalize_score(mes_1)
    auc_1 = roc_auc_score(ano_label, mes_1)
    AP_1 = average_precision_score(ano_label, mes_1, average='macro', pos_label=1, sample_weight=None)

    mes_2 = np.array(torch.squeeze(message_sum_2).cpu().detach())
    mes_2 = 1 - normalize_score(mes_2)
    auc_2 = roc_auc_score(ano_label, mes_2)
    AP_2 = average_precision_score(ano_label, mes_2, average='macro', pos_label=1, sample_weight=None)

    # message_name = f"mes_2_{args.seed}"
    # GCTAM_draw_pdf(1 - mes_2, ano_label, args.dataset, message_name)

    mes_3 = np.array(torch.squeeze(message_sum_3).cpu().detach())
    mes_3 = 1 - normalize_score(mes_3)
    auc_3 = roc_auc_score(ano_label, mes_3)
    AP_3 = average_precision_score(ano_label, mes_3, average='macro', pos_label=1, sample_weight=None)

    # message_name = f"mes_3_{args.seed}"
    # GCTAM_draw_pdf(1 - mes_3, ano_label, args.dataset, message_name)


    final_auc = roc_auc_score(ano_label, final_message)
    print('{} final_message AUC:{:.4f}'.format(args.dataset, final_auc))

    score = final_message
    auc = roc_auc_score(ano_label, score)
    AP = average_precision_score(ano_label, score, average='macro', pos_label=1, sample_weight=None)

    with open(args.output_dir.parent.joinpath(f"{args.dataset}_exp_results"), "a+", encoding='utf-8') as f:
        # f.write('{} message_sum_1  AUC:{:.4f}\n'.format(args.dataset, auc_1))
        # f.write('{} message_sum_2 AUC:{:.4f}\n'.format(args.dataset, auc_2))
        # f.write('{} message_sum_3 AUC:{:.4f}\n'.format(args.dataset, auc_3))
        # f.write('{} AP_1:{:.4f}\n'.format(args.dataset, AP_1))
        # f.write('{} AP_2:{:.4f}\n'.format(args.dataset, AP_2))
        # f.write('{} AP_3:{:.4f}\n'.format(args.dataset, AP_3))

        f.write('{} final_message AUC:{:.4f}\n'.format(args.dataset, final_auc))
        f.write(f'{args.dataset} AUC:{auc:.4f}\n')
        f.write(f'{args.dataset} AP:{AP:.4f}\n')
        f.write(f"args:{args}\n\n")

    res_all = (auc_1, AP_1, auc_2, AP_2, auc_3, AP_3)
    return final_auc, AP, res_all


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [3]))
    os.environ["KMP_DUPLICATE_LnIB_OK"] = "TRUE"
    device_ids = [0]
    # Set argument
    parser = argparse.ArgumentParser(description='Truncated Affinity Maximization for Graph Anomaly Detection')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--num_epoch', type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--drop_prob', type=float, default=0.0)
    parser.add_argument('--subgraph_size', type=int, default=15)
    parser.add_argument('--readout', type=str, default='avg')  # max min avg  weighted_sum
    parser.add_argument('--margin', type=int, default=2)
    parser.add_argument('--negsamp_ratio', type=int, default=2)
    parser.add_argument('--cutting', type=int, default=1)  # 3 5 8 10
    parser.add_argument('--N_tree', type=int, default=1)  # 3 5 8 10

    parser.add_argument('--lamda', type=int, default=0)  # 0  0.5  1

    parser.add_argument('--graph_learner', type=str, default="MLP")  # MLP GCN
    parser.add_argument('--k', type=int, default=10)  # number neighbor

    parser.add_argument('--delete_edge_rate', type=float, default=0.8)  #

    parser.add_argument('--output_path', type=str, default="output")  #
    parser.add_argument('--repeat', type=int, default=5)
    # parser.add_argument('--repeat', type=int, default=1)

    parser.add_argument('--dataset', type=str,
                        default='Amazon')  # 'Amazon' 'YelpChi'  'ACM' 'Facebook'  'Reddit'   Amazon-all 'YelpChi-all'

    args = parser.parse_args()
    local_time = time.localtime()
    args.lct = time.strftime("%Y_%m_%d_%H_%M_%S", local_time)
    print('Dataset: ', args.dataset)
    output_dir = Path.cwd().joinpath(
        args.output_path,
        args.dataset,
        f"seed_{args.seed}")
    args.output_dir = output_dir
    check_writable(output_dir, overwrite=False)
    print(args)

    # delete_edge_rates = [0.4]  # blog
    # delete_edge_rates = [0.4]  # ACM
    delete_edge_rates = [0.9]  # Amazon
    # delete_edge_rates = [0.9]  # Amazon-all
    # delete_edge_rates = [0.3]  # Facebook
    # delete_edge_rates = [0.4]  # Reddit
    # delete_edge_rates = [0.0]  # YelpChi
    auc, ap = start_train(args)

    # delete_edge_rates = [0.2]  # YelpChi
    # delete_edge_rates = [0.0]  # YelpChi
    #
    # delete_edge_rates = [0.1]  # YelpChi-all
    #
    # delete_edge_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    #
    # neighbor = [1, 3, 5, 10, 15, 20]
    #
    # delete_edge_rates = [0.9, 0.95, 0.98]
    # delete_edge_rates = [0.7, 0.8, 0.9, 0.95]
    # for delete in delete_edge_rates:
    #     args.delete_edge_rate = delete
    #     auc, ap = start_train(args)
    #
    # for delete in delete_edge_rates:
    #
    #     args.delete_edge_rate = delete
    #     auc_list = []
    #     ap_list = []
    #     for r in range(args.repeat):
    #         seed_everything(r)
    #         args.delete_edge_rate = delete_edge_rates[0]
    #         auc, ap = start_train(args)
    #         auc_list.append(auc)
    #         ap_list.append(ap)
    #
    #     with open(args.output_dir.parent.joinpath(f"exp_results"), "a+", encoding='utf-8') as f:
    #         auc = np.array(auc_list)
    #         ap = np.array(ap_list)
    #
    #         print(f"final_auc :{auc.mean(axis=0)} + {auc.std(axis=0)}\n")
    #         print(f"final_ap:{ap.mean(axis=0)} + {ap.std(axis=0)}\n")
    #
    #         f.write(f"final_auc:{auc.mean(axis=0)} + {auc.std(axis=0)}\n")
    #         f.write(f"final_ap:{ap.mean(axis=0)} + {ap.std(axis=0)}\n\n")
