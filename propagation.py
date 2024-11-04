import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.linalg import expm

from utils import MixedDropout, sparse_matrix_to_torch
from torch_sparse import SparseTensor

from KANLayer import KANLayer

def full_attention_conv(qs, ks):
    '''
    qs: query tensor [N, H, M]
    ks: key tensor [L, H, M]
    vs: value tensor [L, H, D]

    return output [N, H, D]
    '''
    # normalize input
    qs = qs / torch.norm(qs, p=2) # [N, H, M]
    ks = ks / torch.norm(ks, p=2) # [L, H, M]
    N = qs.shape[0]

    # denominator
    all_ones = torch.ones([ks.shape[0]]).to(ks.device)
    ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
    attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

    # attentive aggregated results
    attention_normalizer = torch.unsqueeze(attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
    attention_normalizer += torch.ones_like(attention_normalizer) * N

    # compute attention for visualization if needed
    attention = torch.einsum("nhm,lhm->nhl", qs, ks) / attention_normalizer # [N, L, H]
    return attention

def On_attention_conv(qs, ks, vs, K_hop):
    qs = qs / torch.norm(qs, p=2)  # [N, H, M]
    ks = ks / torch.norm(ks, p=2)  # [L, H, M]
    N = qs.shape[0]

    # numerator
    new_vs = vs
    for _ in range(K_hop):
        new_vs = torch.einsum("lhm,lhd->hmd", ks, new_vs)
        new_vs = torch.einsum("nhm,hmd->nhd", qs, new_vs)
    all_ones = torch.ones([vs.shape[0]]).to(vs.device)
    vs_sum = torch.einsum("l,lhd->hd", all_ones, vs)  # [H, D]
    new_vs += vs_sum.unsqueeze(0).repeat(vs.shape[0], 1, 1)  # [N, H, D]

    # denominator
    all_ones = torch.ones([ks.shape[0]]).to(ks.device)
    ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
    attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

    # attentive aggregated results
    attention_normalizer = torch.unsqueeze(attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
    attention_normalizer += torch.ones_like(attention_normalizer) * N
    for _ in range(K_hop):
        attn_output = new_vs / attention_normalizer  # [N, H, D]
    return attn_output

def slice_sparse_matrix(sparse_matrix, start_index, end_index):
    # 获取稀疏矩阵的索引和值
    indices = sparse_matrix._indices()
    values = sparse_matrix._values()

    # 仅保留所需行范围内的索引和对应的值
    mask = (indices[0] >= start_index) & (indices[0] < end_index)
    sliced_indices = indices[:, mask]
    sliced_values = values[mask]

    # 调整行索引以反映新矩阵的行号
    sliced_indices[0] -= start_index

    # 获取原始矩阵的尺寸，并调整新矩阵的尺寸
    original_size = sparse_matrix.size()
    sliced_size = (end_index - start_index, original_size[1])

    # 创建新的稀疏矩阵
    sliced_sparse_matrix = torch.sparse_coo_tensor(sliced_indices, sliced_values, sliced_size)

    return sliced_sparse_matrix

def calc_A_hat(adj_matrix: sp.spmatrix) -> sp.spmatrix:
    nnodes = adj_matrix.shape[0]
    A = adj_matrix + sp.eye(nnodes)
    D_vec = np.sum(A, axis=1).A1
    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
    # D_vec_invsqrt_corr = 1 / D_vec
    D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)
    return D_invsqrt_corr @ A @ D_invsqrt_corr
    # return D_invsqrt_corr @ A

def calc_ppr_exact(adj_matrix: sp.spmatrix, alpha: float) -> np.ndarray:
    nnodes = adj_matrix.shape[0]
    M = calc_A_hat(adj_matrix)
    A_inner = sp.eye(nnodes) - (1 - alpha) * M
    return alpha * np.linalg.inv(A_inner.toarray())

class PPRExact(nn.Module):
    def __init__(self, adj_matrix: sp.spmatrix, alpha: float, drop_prob: float = None):
        super().__init__()

        ppr_mat = calc_ppr_exact(adj_matrix, alpha)
        self.register_buffer('mat', torch.FloatTensor(ppr_mat))

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

    def forward(self, predictions: torch.FloatTensor, idx: torch.LongTensor):
        return self.dropout(self.mat[idx]) @ predictions

class PPRPowerIteration(nn.Module):
    def __init__(self, adj_matrix: sp.spmatrix, num_heads: int, nclasses: int, niter: int, npow: int, npow_attn: int,
                 nalpha:float, xi:float, num_feature: int = None, num_hidden: int = None, drop_prob: float = None, device=None, niter_attn: int = None):
        '''
        Parameters
        ----------
        adj_matrix : 原始的图
        niter : 阶数
        npow : 跳
        nalpha : APGNN前面的系数
        drop_prob : dropout
        '''
        super().__init__()

        self.niter = niter
        if niter_attn is None:
            self.niter_attn = niter
        else:
            self.niter_attn = niter_attn
        self.npow = npow
        self.npow_attn = npow_attn
        self.nalpha=nalpha
        self.device = device
        self.num_view = 1
        self.num_heads = num_heads
        self.nclasses = nclasses

        self.ori_adj = calc_A_hat(adj_matrix)
        indices_np = np.array([self.ori_adj.nonzero()[0], self.ori_adj.nonzero()[1]])
        indices = torch.tensor(indices_np)
        values = torch.tensor(self.ori_adj.data, dtype=torch.float)
        size = torch.Size(self.ori_adj.shape)
        self.ori_adj = torch.sparse_coo_tensor(indices, values, size).to(device)

        self.xi = xi
        if num_hidden is not None:
            self.num_hidden = num_hidden
        if num_feature is not None:
            self.num_feature = num_feature

        M = calc_A_hat(adj_matrix)
        tempM = M
        for ti in range(self.npow):
            M = M @ tempM
        indices_np = np.array([M.nonzero()[0], M.nonzero()[1]])
        indices = torch.tensor(indices_np)
        values = torch.tensor(M.data, dtype=torch.float)
        size = torch.Size(M.shape)
        self.A = torch.sparse_coo_tensor(indices, values, size).to(device)
        self.A = self.A.unsqueeze(0)
        self.fc1 = nn.Sequential(
            nn.Linear(niter, 1),
        )
        if num_hidden is None:
            self.Wk = nn.Linear(nclasses, nclasses * num_heads)
            self.Wq = nn.Linear(nclasses, nclasses * num_heads)
            # self.Wk = KANLayer(nclasses, nclasses * num_heads, device=device)
            # self.Wq = KANLayer(nclasses, nclasses * num_heads, device=device)
        else:
            self.Wk = nn.Linear(num_feature, num_hidden * num_heads)
            self.Wq = nn.Linear(num_feature, num_hidden * num_heads)
            # self.Wk = KANLayer(num_feature, num_hidden * num_heads, device=device)
            # self.Wq = KANLayer(num_feature, num_hidden * num_heads, device=device)

        self.linear1 = nn.Linear(niter*self.num_view, 1)
        self.linear2 = nn.Linear(niter_attn*self.num_view, 1)
        self.softmax = torch.nn.Softmax(dim=0)

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

        def reset_parameters(self):
            self.Wk.reset_parameters()
            self.Wq.reset_parameters()
            # pass

    def forward(self, local_preds: torch.FloatTensor, idx: torch.LongTensor, origin_fea: torch.sparse.FloatTensor=None):
        preds = local_preds.float()
        if origin_fea is None:
            source_preds = preds.clone()
            query = self.Wq(preds).reshape(-1, self.num_heads, self.nclasses)
            key = self.Wk(source_preds).reshape(-1, self.num_heads, self.nclasses)
            # total_query = []
            # total_key = []
            # batch_size = 16
            # for i in range(0, preds.size(0), batch_size):
            #     if i + batch_size < preds.size(0):
            #         query = self.Wq(preds[i:i+batch_size]).reshape(-1, self.num_heads, self.nclasses)
            #         key = self.Wk(source_preds[i:i+batch_size]).reshape(-1, self.num_heads, self.nclasses)
            #         self.Wq.update_grid_from_samples(preds[i:i + batch_size])
            #         self.Wk.update_grid_from_samples(source_preds[i:i + batch_size])
            #         total_query.append(query)
            #         total_key.append(key)
            #     else:
            #         query = self.Wq(preds[i:]).reshape(-1, self.num_heads, self.nclasses)
            #         key = self.Wk(source_preds[i:]).reshape(-1, self.num_heads, self.nclasses)
            #         self.Wq.update_grid_from_samples(preds[i:])
            #         self.Wk.update_grid_from_samples(source_preds[i:])
            #         total_query.append(query)
            #         total_key.append(key)
            # query = torch.cat(total_query, dim=0)
            # key = torch.cat(total_key, dim=0)

        else:
            source_preds = origin_fea.clone()
            # source_preds_dense = source_preds.to_dense()
            # origin_fea_dense = origin_fea.to_dense()
            query = self.Wq(origin_fea).reshape(-1, self.num_heads, self.num_hidden)
            key = self.Wk(source_preds).reshape(-1, self.num_heads, self.num_hidden)
            # total_query = []
            # total_key = []
            # batch_size = 4
            # for i in range(0, origin_fea.size(0), batch_size):
            #     if i + batch_size < origin_fea.size(0):
            #         query = self.Wq(origin_fea_dense[i:i + batch_size]).reshape(-1, self.num_heads, self.num_hidden)
            #         key = self.Wk(source_preds_dense[i:i + batch_size]).reshape(-1, self.num_heads, self.num_hidden)
            #         self.Wq.update_grid_from_samples(origin_fea_dense[i:i + batch_size])
            #         self.Wk.update_grid_from_samples(source_preds_dense[i:i + batch_size])
            #         total_query.append(query)
            #         total_key.append(key)
            #     else:
            #         query = self.Wq(origin_fea_dense[i:]).reshape(-1, self.num_heads, self.num_hidden)
            #         key = self.Wk(source_preds_dense[i:]).reshape(-1, self.num_heads, self.num_hidden)
            #         self.Wq.update_grid_from_samples(origin_fea_dense[i:])
            #         self.Wk.update_grid_from_samples(source_preds_dense[i:])
            #         total_query.append(query)
            #         total_key.append(key)
            # query = torch.cat(total_query, dim=0)
            # key = torch.cat(total_key, dim=0)

        # =========================-------------revision_4,28,24'----------============================
        all_one_heads = torch.ones(self.num_heads).to(self.device)
        M__ = None
        tmp = None
        for i in range(0, self.niter_attn):
            if i == 0:
                M__ = preds.unsqueeze(0)
                tmp = preds
                tmp = tmp.unsqueeze(0)
            else:
                tmp = torch.einsum('v, lnd->vnd', all_one_heads, tmp)
                tmp = tmp.permute(1, 0, 2)
                tmp = On_attention_conv(query, key, tmp, self.npow_attn+1)
                tmp = torch.mean(tmp, dim=1).unsqueeze(0)
                M__ = torch.cat([M__, tmp], dim=0)
        # =========================-------------revision_4,28,24'----------============================
        beta = self.linear2.weight.t().unsqueeze(1)
        preds = torch.sum(beta * M__, dim=0)

        all_one = torch.ones(self.num_view).to(self.device)
        M__ = None
        tmp = None
        for i in range(0, self.niter):
            if i == 0:
                M__ = preds.unsqueeze(0)
                M__ = torch.einsum('v, lnd->vnd', all_one, M__)

                tmp = preds
                tmp = tmp.unsqueeze(0)

            else:
                tmp = torch.einsum('v, lnd->vnd', all_one, tmp)
                tmp = torch.bmm(self.A, tmp)
                M__ = torch.cat([M__, tmp], dim=0)

        alph = self.linear1.weight.t().unsqueeze(1)
        preds = torch.sum(alph * M__, dim=0)

        I_mat = torch.eye(self.A.shape[1]).to(self.device)
        graph_a = alph[-1] * I_mat
        for i in range(self.niter - 1, 0, -1):
            graph_a = alph[i - 1] * I_mat + torch.bmm(self.A, graph_a.unsqueeze(0)).squeeze(0)
        # =======================---------revision 4.29.24'----------=======================
        r = 4096
        loss_Resi = 0
        if graph_a.size(0) > r:
            M__ = None
            tmp = None
            for i in range(1, self.niter_attn, 1):
                if i == 1:
                    M__ = query.clone()
                    M__ = torch.mean(M__, dim=1).unsqueeze(0)
                    tmp = query.clone()
                    tmp = tmp.permute(1, 0, 2)
                else:
                    tmp = torch.einsum('v, lnd->vnd', all_one_heads, tmp)
                    tmp = tmp.permute(1, 0, 2)
                    tmp = On_attention_conv(query, key, tmp, self.npow_attn + 1)
                    tmp = torch.mean(tmp, dim=1).unsqueeze(0)
                    tmp = torch.bmm(graph_a.unsqueeze(0), tmp)
                    M__ = torch.cat([M__, tmp], dim=0)

            GraphA_q = torch.sum(beta[1:] * M__, dim=0)
            GraphA_q = torch.einsum('v, nd->vnd', all_one_heads, GraphA_q)

            # 计算需要迭代的次数，考虑到query的行数和r不能整除的情况
            num_iterations = (GraphA_q.size(0) + r - 1) // r
            for i in range(num_iterations):
                # 计算开始和结束的索引
                start_index = i * r
                end_index = min((i + 1) * r, GraphA_q.size(0))
                # 计算Attn
                attn_row = full_attention_conv(GraphA_q[start_index:end_index], key)
                loss_Resi += torch.norm(attn_row, p='fro') ** 2
            loss_Resi = torch.sqrt(loss_Resi)
        else:
            # =======================---------revision 4.29.24'----------=======================
            graph_b = beta[-1] * I_mat
            if self.npow_attn == 0:
                for i in range(self.niter_attn - 1, 0, -1):
                    graph_b_vs = torch.einsum('v, nd->vnd', all_one_heads, graph_b)
                    graph_b_vs = graph_b_vs.permute(1, 0, 2)
                    ks_graphb_vs = torch.einsum("lhm,lhd->hmd", key, graph_b_vs)
                    graph_tmp = torch.einsum("nhm,hmd->nhd", query, ks_graphb_vs)
                    graph_b = beta[i - 1] * I_mat + torch.mean(graph_tmp, dim=1)
            else:
                q_vs = query.clone()
                Attn_qkq = On_attention_conv(query, key, q_vs, self.npow_attn)
                for i in range(self.niter_attn - 1, 0, -1):
                    graph_b_vs = torch.einsum('v, nd->vnd', all_one_heads, graph_b)
                    graph_b_vs = graph_b_vs.permute(1, 0, 2)
                    ks_graphb_vs = torch.einsum("lhm,lhd->hmd", key, graph_b_vs)
                    graph_tmp = torch.einsum("nhm,hmd->nhd", Attn_qkq, ks_graphb_vs)
                    graph_b = beta[i - 1] * I_mat + torch.mean(graph_tmp, dim=1)

            preds_graph = graph_a @ graph_b
            Resi = preds_graph.squeeze(0) - self.ori_adj
            Resi = torch.exp(- self.xi * (Resi) ** 2) * Resi
            loss_Resi = torch.norm(Resi, p='fro')

        return preds[idx], loss_Resi

class HornerSparseIteration_sparse(nn.Module):
    def __init__(self, adj_matrix: sp.spmatrix, num_heads: int, nclasses: int, niter: int, npow: int, npow_attn: int,
                 nalpha: float, xi:float, num_feature: int = None, num_hidden: int = None, drop_prob: float = None, device=None,
                 niter_attn: int = None):
        '''
        Parameters
        ----------
        adj_matrix : 原始的图
        niter : 阶数
        npow : 跳
        nalpha : APGNN前面的系数
        drop_prob : dropout
        '''
        super().__init__()

        self.niter = niter
        if niter_attn is None:
            self.niter_attn = niter
        else:
            self.niter_attn = niter_attn
        self.npow = 0
        self.npow_attn = 1
        self.nalpha = nalpha
        self.device = device
        self.num_view = 1
        self.num_heads = num_heads
        self.nclasses = nclasses
        self.xi = xi

        self.ori_adj = calc_A_hat(adj_matrix)
        indices_np = np.array([self.ori_adj.nonzero()[0], self.ori_adj.nonzero()[1]])
        indices = torch.tensor(indices_np)
        values = torch.tensor(self.ori_adj.data, dtype=torch.float)
        size = torch.Size(self.ori_adj.shape)
        self.ori_adj = torch.sparse_coo_tensor(indices, values, size).to(device)

        if num_hidden is not None:
            self.num_hidden = num_hidden
        if num_feature is not None:
            self.num_feature = num_feature

        M = calc_A_hat(adj_matrix)
        tempM = M
        for ti in range(self.npow):
            M = M @ tempM
        indices_np = np.array([M.nonzero()[0], M.nonzero()[1]])
        indices = torch.tensor(indices_np)
        values = torch.tensor(M.data, dtype=torch.float)
        size = torch.Size(M.shape)
        self.A = torch.sparse_coo_tensor(indices, values, size).to(device)
        self.A = self.A.unsqueeze(0)

        self.fc1 = nn.Sequential(
            nn.Linear(niter, 1),
        )
        if num_hidden is None:
            self.Wk = nn.Linear(nclasses, nclasses * num_heads)
            self.Wq = nn.Linear(nclasses, nclasses * num_heads)
        else:
            self.Wk = nn.Linear(num_feature, num_hidden * num_heads)
            self.Wq = nn.Linear(num_feature, num_hidden * num_heads)

        self.linear1 = nn.Linear(self.niter * self.num_view, 1)
        self.linear2 = nn.Linear(self.niter_attn * self.num_view, 1)
        self.softmax = torch.nn.Softmax(dim=0)

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

        def reset_parameters(self):
            self.Wk.reset_parameters()
            self.Wq.reset_parameters()

    def forward(self, local_preds: torch.FloatTensor, idx: torch.LongTensor,
                origin_fea: torch.sparse.FloatTensor = None):
        preds = local_preds.float()
        if origin_fea is None:
            source_preds = preds.clone()
            query = self.Wq(preds).reshape(-1, self.num_heads, self.nclasses)
            key = self.Wk(source_preds).reshape(-1, self.num_heads, self.nclasses)
            # 初始化索引和值的列表
            indices = []
            values = []
            # 逐行计算Attn
            for i in range(query.size(0)):
                attn_row = full_attention_conv(query[i].unsqueeze(0), key)
                # 使用Gumbel Softmax进行采样
                gumbel_noise = torch.nn.functional.gumbel_softmax(attn_row, tau=1, hard=False)
                # 选择前10个最大的值
                topk_values, topk_indices = torch.topk(gumbel_noise, 10)
                # 更新索引和值的列表
                indices.append((i, topk_indices))
                values.append(topk_values)
            # 创建稀疏矩阵
            indices = torch.tensor(indices, dtype=torch.long)
            values = torch.tensor(values, dtype=torch.float)
            Attn = torch.sparse_coo_tensor(indices.t(), values, query.size())
            # 确保梯度能够回传
            Attn.requires_grad = True
        else:
            query = self.Wq(origin_fea).reshape(-1, self.num_heads, self.num_hidden)
            key = self.Wk(origin_fea).reshape(-1, self.num_heads, self.num_hidden)
            # 初始化索引和值的列表
            indices = []
            values = []
            # 设置每次处理的行数
            r = 4096
            # 计算需要迭代的次数，考虑到query的行数和r不能整除的情况
            num_iterations = (query.size(0) + r - 1) // r
            for i in range(num_iterations):
                # 计算开始和结束的索引
                start_index = i * r
                end_index = min((i + 1) * r, query.size(0))
                # 计算Attn
                attn_row = full_attention_conv(query[start_index:end_index], key)
                # 对attn_row的第二个维度取平均，使用了multihead技术得到的矩阵attn_row的第二个维度是head的维度
                attn_row = attn_row.mean(dim=1)
                if i == 0:
                    # 使用Gumbel Softmax进行采样
                    gumbel_noise = torch.nn.functional.gumbel_softmax(attn_row, tau=1, hard=False)
                    # 选择前10个最大的值
                    topk_values, topk_indices = torch.topk(gumbel_noise, 10, dim=1)
                    j = torch.arange(topk_indices.size(0)).view(-1, 1).expand_as(topk_indices)
                    j = j + start_index
                    j = j.to(self.device)
                    indices = torch.stack((j.flatten(), topk_indices.flatten()))
                    values = topk_values.flatten()

                    # 创建稀疏矩阵
                    indices = torch.tensor(indices, dtype=torch.int)
                    values = torch.tensor(values, dtype=torch.float)
                else:
                    r_indices = torch.tensor([[start_index], [0]])+indices[0:10*(end_index-start_index)]
                    r_values = attn_row[0:(end_index-start_index), :]
                    values_indices = indices[1, 0:10*(end_index-start_index)].reshape(-1, 10)
                    values_indices = values_indices.to(torch.int64)
                    r_values_ = torch.stack([r_values[k, values_indices[k, :]] for k in range(end_index-start_index)]).detach()
                    r_values_ = torch.reshape(r_values_, (-1, ))
                    indices = torch.cat((indices, r_indices), dim=1)
                    values = torch.cat((values, r_values_), dim=0)
            Attn = torch.sparse_coo_tensor(indices, values, (query.size(0), key.size(0)))
            Attn = Attn.to(self.device)

        temp_Attn = Attn
        for ti in range(self.npow_attn):
            Attn = Attn @ temp_Attn
        Attn = Attn.unsqueeze(0)

        all_one = torch.ones(self.num_view).to(self.device)

        M__ = None
        tmp = None
        for i in range(0, self.niter_attn):
            if i == 0:
                M__ = preds.unsqueeze(0)
                M__ = torch.einsum('v, lnd->vnd', all_one, M__)

                tmp = preds
                tmp = tmp.unsqueeze(0)

            else:
                tmp = torch.einsum('v, lnd->vnd', all_one, tmp)
                tmp = torch.bmm(Attn, tmp)
                M__ = torch.cat([M__, tmp], dim=0)

        beta = self.linear2.weight.t().unsqueeze(1)
        preds = torch.sum(beta * M__, dim=0)

        M__ = None
        tmp = None
        for i in range(0, self.niter):
            if i == 0:
                M__ = preds.unsqueeze(0)
                M__ = torch.einsum('v, lnd->vnd', all_one, M__)

                tmp = preds
                tmp = tmp.unsqueeze(0)

            else:
                tmp = torch.einsum('v, lnd->vnd', all_one, tmp)
                tmp = torch.bmm(self.A, tmp)
                M__ = torch.cat([M__, tmp], dim=0)

        alph = self.linear1.weight.t().unsqueeze(1)
        preds = torch.sum(alph * M__, dim=0)

        # I_mat = torch.eye(self.A.shape[1]).to(self.device)
        # # graph_a = alph[-1] * I_mat
        # # for i in range(self.niter - 1, 0, -1):
        # #     graph_a = alph[i - 1] * I_mat + torch.bmm(self.A, graph_a.unsqueeze(0)).squeeze(0)
        # graph_a = torch.bmm(self.A, I_mat.unsqueeze(0)).squeeze(0)
        # graph_b = beta[-1] * I_mat
        # for i in range(self.niter_attn - 1, 0, -1):
        #     graph_b = beta[i - 1] * I_mat + torch.bmm(Attn, graph_b.unsqueeze(0)).squeeze(0)
        #
        # preds_graph = graph_a @ graph_b
        # Resi = preds_graph.squeeze(0) - self.ori_adj
        # Resi = torch.exp(- self.xi * (Resi) ** 2) * Resi
        # loss_Resi = torch.norm(Resi, p='fro')
        loss_Resi = 0

        return preds[idx], loss_Resi

class HSIteration_ApriPolyEncDec(nn.Module):
    def __init__(self, adj_matrix: sp.spmatrix, num_heads: int, nclasses: int, niter: int, npow: int, npow_attn: int,
                 nalpha: float, xi:float, num_feature: int = None, num_hidden: int = None, drop_prob: float = None, device=None,
                 niter_attn: int = None):
        '''
        Parameters
        ----------
        adj_matrix : 原始的图
        niter : 阶数
        npow : 跳
        nalpha : APGNN前面的系数
        drop_prob : dropout
        '''
        super().__init__()

        self.niter = niter
        if niter_attn is None:
            self.niter_attn = niter
        else:
            self.niter_attn = niter_attn
        self.npow = 0
        self.npow_attn = 1
        self.nalpha = nalpha
        self.device = device
        self.num_view = 1
        self.num_heads = num_heads
        self.nclasses = nclasses
        self.xi = xi

        self.ori_adj = calc_A_hat(adj_matrix)
        indices_np = np.array([self.ori_adj.nonzero()[0], self.ori_adj.nonzero()[1]])
        indices = torch.tensor(indices_np)
        values = torch.tensor(self.ori_adj.data, dtype=torch.float)
        size = torch.Size(self.ori_adj.shape)
        self.ori_adj = torch.sparse_coo_tensor(indices, values, size).to(device)

        if num_hidden is not None:
            self.num_hidden = num_hidden
        if num_feature is not None:
            self.num_feature = num_feature

        M = calc_A_hat(adj_matrix)
        tempM = M
        for ti in range(self.npow):
            M = M @ tempM
        indices_np = np.array([M.nonzero()[0], M.nonzero()[1]])
        indices = torch.tensor(indices_np)
        values = torch.tensor(M.data, dtype=torch.float)
        size = torch.Size(M.shape)
        self.A = torch.sparse_coo_tensor(indices, values, size).to(device)
        self.A = self.A.unsqueeze(0)

        self.fc1 = nn.Sequential(
            nn.Linear(niter, 1),
        )
        if num_hidden is None:
            self.Wk = nn.Linear(nclasses, nclasses * num_heads)
            self.Wq = nn.Linear(nclasses, nclasses * num_heads)
        else:
            self.Wk = nn.Linear(num_feature, num_hidden * num_heads)
            self.Wq = nn.Linear(num_feature, num_hidden * num_heads)

        self.linear1 = nn.Linear(self.niter * self.num_view, 1)
        self.linear2 = nn.Linear(self.niter_attn * self.num_view, 1)
        self.softmax = torch.nn.Softmax(dim=0)

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

        def reset_parameters(self):
            self.Wk.reset_parameters()
            self.Wq.reset_parameters()

    def forward(self, local_preds: torch.FloatTensor, idx: torch.LongTensor,
                origin_fea: torch.sparse.FloatTensor = None):
        preds = local_preds.float()
        if origin_fea is None:
            source_preds = preds.clone()
            query = self.Wq(preds).reshape(-1, self.num_heads, self.nclasses)
            key = self.Wk(source_preds).reshape(-1, self.num_heads, self.nclasses)
        else:
            source_preds = origin_fea.clone()
            query = self.Wq(origin_fea).reshape(-1, self.num_heads, self.num_hidden)
            key = self.Wk(source_preds).reshape(-1, self.num_heads, self.num_hidden)

        # =========================-------------revision_5,21,24'----------============================
        all_one_heads = torch.ones(self.num_heads).to(self.device)
        all_one = torch.ones(self.num_view).to(self.device)

        M__ = None
        tmp = None
        for i in range(0, self.niter):
            if i == 0:
                M__ = preds.unsqueeze(0)
                M__ = torch.einsum('v, lnd->vnd', all_one, M__)
                tmp = preds
                tmp = tmp.unsqueeze(0)
            else:
                QK_M = None
                QK_tmp = None
                for j in range(0, self.niter_attn):
                    if j == 0:
                        QK_M = torch.einsum('v, lnd->vnd', all_one_heads, tmp)
                        QK_tmp = tmp
                    else:
                        QK_tmp = torch.einsum('v, lnd->vnd', all_one_heads, QK_tmp)
                        QK_tmp = QK_tmp.permute(1, 0, 2)
                        QK_tmp = On_attention_conv(query, key, QK_tmp, self.npow_attn + 1)
                        QK_tmp = torch.mean(QK_tmp, dim=1).unsqueeze(0)
                        QK_M = torch.cat([QK_M, QK_tmp], dim=0)
                beta = self.linear2.weight.t().unsqueeze(1)
                tmp = torch.sum(beta * QK_M, dim=0)

                tmp = tmp.unsqueeze(0)
                tmp = torch.bmm(self.A, tmp)
                M__ = torch.cat([M__, tmp], dim=0)

        alph = self.linear1.weight.t().unsqueeze(1)
        preds = torch.sum(alph * M__, dim=0)

        # =========================-------------revision_5,21,24'----------============================

        # I_mat = torch.eye(self.A.shape[1]).to(self.device)
        # graph_a = torch.bmm(self.ori_adj.unsqueeze(0), torch.eye(self.A.shape[1]).to(self.device).unsqueeze(0)).squeeze(0)
        # # =======================---------revision 4.29.24'----------=======================
        # r = 4096
        # loss_Resi = 0
        # if graph_a.size(0) > r:
        #     M__ = None
        #     tmp = None
        #     for i in range(1, self.niter_attn, 1):
        #         if i == 1:
        #             M__ = query.clone()
        #             M__ = torch.mean(M__, dim=1).unsqueeze(0)
        #             tmp = query.clone()
        #             tmp = tmp.permute(1, 0, 2)
        #         else:
        #             tmp = torch.einsum('v, lnd->vnd', all_one_heads, tmp)
        #             tmp = tmp.permute(1, 0, 2)
        #             tmp = On_attention_conv(query, key, tmp, self.npow_attn + 1)
        #             tmp = torch.mean(tmp, dim=1).unsqueeze(0)
        #             tmp = torch.bmm(graph_a.unsqueeze(0), tmp)
        #             M__ = torch.cat([M__, tmp], dim=0)
        #
        #     GraphA_q = torch.sum(beta[1:] * M__, dim=0)
        #     GraphA_q = torch.einsum('v, nd->vnd', all_one_heads, GraphA_q)
        #
        #     # 计算需要迭代的次数，考虑到query的行数和r不能整除的情况
        #     num_iterations = (GraphA_q.size(0) + r - 1) // r
        #     for i in range(num_iterations):
        #         # 计算开始和结束的索引
        #         start_index = i * r
        #         end_index = min((i + 1) * r, GraphA_q.size(0))
        #         # 计算Attn
        #         attn_row = full_attention_conv(GraphA_q[start_index:end_index], key) + (beta[0]-1) * graph_a[start_index:end_index]
        #         loss_Resi += torch.norm(attn_row, p='fro') ** 2
        #     loss_Resi = torch.sqrt(loss_Resi)
        # else:
        #     # =======================---------revision 4.29.24'----------=======================
        #     beta = self.linear2.weight.t().unsqueeze(1)
        #     graph_b = beta[-1] * I_mat
        #     if self.npow_attn == 0:
        #         for i in range(self.niter_attn - 1, 0, -1):
        #             graph_b_vs = torch.einsum('v, nd->vnd', all_one_heads, graph_b)
        #             graph_b_vs = graph_b_vs.permute(1, 0, 2)
        #             ks_graphb_vs = torch.einsum("lhm,lhd->hmd", key, graph_b_vs)
        #             graph_tmp = torch.einsum("nhm,hmd->nhd", query, ks_graphb_vs)
        #             graph_b = beta[i - 1] * I_mat + torch.mean(graph_tmp, dim=1)
        #     else:
        #         q_vs = query.clone()
        #         Attn_qkq = On_attention_conv(query, key, q_vs, self.npow_attn)
        #         for i in range(self.niter_attn - 1, 0, -1):
        #             graph_b_vs = torch.einsum('v, nd->vnd', all_one_heads, graph_b)
        #             graph_b_vs = graph_b_vs.permute(1, 0, 2)
        #             ks_graphb_vs = torch.einsum("lhm,lhd->hmd", key, graph_b_vs)
        #             graph_tmp = torch.einsum("nhm,hmd->nhd", Attn_qkq, ks_graphb_vs)
        #             graph_b = beta[i - 1] * I_mat + torch.mean(graph_tmp, dim=1)
        #
        #     preds_graph = graph_a @ graph_b
        #     Resi = preds_graph.squeeze(0) - self.ori_adj
        #     Resi = torch.exp(- self.xi * (Resi) ** 2) * Resi
        #     loss_Resi = torch.norm(Resi, p='fro')

        # ==========================   revision 6.26.24' copy from 4090 version
        with torch.no_grad():
            # 创建单位矩阵的索引和数值
            indices = torch.arange(self.A.shape[1], dtype=torch.long).repeat(2, 1).to(self.device)
            values = torch.ones(self.A.shape[1], dtype=torch.float32).to(self.device)
            I_mat = torch.sparse_coo_tensor(indices, values, (self.A.shape[1], self.A.shape[1])).to(self.device)
            graph_a = self.ori_adj
            # =======================---------revision 4.29.24'----------=======================
            r = 1024
            loss_Resi = 0
            if graph_a.size(0) > r:
                beta = self.linear2.weight.t().unsqueeze(1)
                M__ = None
                tmp = None
                for i in range(1, self.niter_attn, 1):
                    if i == 1:
                        M__ = query.clone()
                        M__ = torch.mean(M__, dim=1).unsqueeze(0)
                        tmp = query.clone()
                        tmp = tmp.permute(1, 0, 2)
                    else:
                        tmp = torch.einsum('v, lnd->vnd', all_one_heads, tmp)
                        tmp = tmp.permute(1, 0, 2)
                        tmp = On_attention_conv(query, key, tmp, self.npow_attn + 1)
                        tmp = torch.mean(tmp, dim=1).unsqueeze(0)
                        tmp = torch.bmm(graph_a.unsqueeze(0), tmp)
                        M__ = torch.cat([M__, tmp], dim=0)

                GraphA_q = torch.sum(beta[1:] * M__, dim=0)
                GraphA_q = torch.einsum('v, nd->vnd', all_one_heads, GraphA_q)
                GraphA_q = GraphA_q.permute(1, 0, 2)

                # 计算需要迭代的次数，考虑到query的行数和r不能整除的情况
                num_iterations = (GraphA_q.size(0) + r - 1) // r
                for i in range(num_iterations):
                    # 计算开始和结束的索引
                    start_index = i * r
                    end_index = min((i + 1) * r, GraphA_q.size(0))
                    # 计算Attn
                    attn_row = full_attention_conv(GraphA_q[start_index:end_index], key) + (beta[0]-1) * slice_sparse_matrix(graph_a, start_index, end_index).unsqueeze(1)
                    loss_Resi += torch.norm(attn_row, p='fro') ** 2
                loss_Resi = torch.sqrt(loss_Resi)
            else:
                # =======================---------revision 4.29.24'----------=======================
                beta = self.linear2.weight.t().unsqueeze(1)
                graph_b = beta[-1] * I_mat
                if self.npow_attn == 0:
                    for i in range(self.niter_attn - 1, 0, -1):
                        graph_b_vs = torch.einsum('v, nd->vnd', all_one_heads, graph_b)
                        graph_b_vs = graph_b_vs.permute(1, 0, 2)
                        graph_b_vs = graph_b_vs.to_dense()
                        ks_graphb_vs = torch.einsum("lhm,lhd->hmd", key, graph_b_vs)
                        graph_tmp = torch.einsum("nhm,hmd->nhd", query, ks_graphb_vs)
                        graph_b = torch.mean(graph_tmp, dim=1) + beta[i - 1] * I_mat
                else:
                    q_vs = query.clone()
                    Attn_qkq = On_attention_conv(query, key, q_vs, self.npow_attn)
                    for i in range(self.niter_attn - 1, 0, -1):
                        graph_b_vs = torch.einsum('v, nd->vnd', all_one_heads, graph_b)
                        graph_b_vs = graph_b_vs.permute(1, 0, 2)
                        graph_b_vs = graph_b_vs.to_dense()
                        ks_graphb_vs = torch.einsum("lhm,lhd->hmd", key, graph_b_vs)
                        graph_tmp = torch.einsum("nhm,hmd->nhd", Attn_qkq, ks_graphb_vs)
                        graph_b = torch.mean(graph_tmp, dim=1) + beta[i - 1] * I_mat

                preds_graph = graph_a @ graph_b
                Resi = preds_graph.squeeze(0) - self.ori_adj
                Resi = torch.exp(- self.xi * (Resi) ** 2) * Resi
                loss_Resi = torch.norm(Resi, p='fro')

        return preds[idx], loss_Resi