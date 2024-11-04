import numpy as np
import scipy.sparse as sp
import torch
import os.path as osp
from deeprobust.graph.global_attack import Metattack, MinMax, BaseAttack
from deeprobust.graph.defense import GCN
from torch_geometric.utils import from_scipy_sparse_matrix, dense_to_sparse
import gc

def mask_to_index(mask, size):
    all_idx = np.arange(size)
    return all_idx[mask]

class Attacker:
    def __init__(self, args, data):
        self.attack_type = args.attack_type
        self.attack_ratio = args.attack_ratio
        self.dataset = args.dataset
        self.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        n = data.num_nodes
        self.train_index = mask_to_index(train_mask, n)
        self.val_index = mask_to_index(val_mask, n)
        self.test_index = mask_to_index(test_mask, n)
        self.adj = data.edge_index
        self.features, self.labels = data.x, data.y
        self.n_edge_mod = int(args.attack_ratio * (self.adj.sum() // 2))
        self.modified_adj = None

    def attack(self, RP):
        if self.attack_type == "DICE":
            self.modified_adj = self.DICE(RP)
        elif self.attack_type == "Random":
            self.modified_adj = self.Random(RP)
        elif self.attack_type == "Meta":
            self.modified_adj = self.Meta(RP)
        elif self.attack_type == "MinMax":
            self.modified_adj = self.MinMax(RP)
        else:
            raise NotImplementedError
        return self.modified_adj

    def DICE(self, RP):
        adj_name = "_".join(["DICE", self.dataset, str(int(self.attack_ratio * 100)), str(RP)]) + ".pt"
        adj_path = osp.join("./atk_data/atk_adj", adj_name)
        if osp.exists(adj_path):
            modified_adj = torch.load(adj_path)
        else:
            atk_model = DICE()
            atk_model.attack(self.adj, self.labels, n_perturbations=self.n_edge_mod, index_target=self.test_index)
            modified_adj = atk_model.modified_adj
            torch.save(modified_adj, adj_path)
        adj_attack, _ = from_scipy_sparse_matrix(modified_adj)
        return adj_attack

    def Random(self, RP):
        adj_name = "_".join(["Random", self.dataset, str(int(self.attack_ratio * 100)), str(RP)]) + ".pt"
        adj_path = osp.join("./atk_data/atk_adj", adj_name)
        if osp.exists(adj_path):
            modified_adj = torch.load(adj_path)
        else:
            atk_model = Random()
            atk_model.attack(self.adj, n_perturbations=self.n_edge_mod)
            modified_adj = atk_model.modified_adj
            torch.save(modified_adj, adj_path)
        adj_attack, _ = from_scipy_sparse_matrix(modified_adj)
        return adj_attack

    def Meta(self, RP):
        adj_name = "_".join(["Meta", self.dataset, str(int(self.attack_ratio * 100)), str(RP)]) + ".pt"
        adj_path = osp.join("./atk_data/atk_adj", adj_name)
        if osp.exists(adj_path):
            modified_adj = torch.load(adj_path, map_location=self.device)
        else:
        #     self.adj = torch.tensor(self.adj.todense(), dtype=torch.float32)
        #     idx_unlabeled = np.union1d(self.val_index, self.test_index)
        #     surrogate = GCN(nfeat=self.features.shape[1], nclass=self.labels.max().item() + 1,
        #                     nhid=16, dropout=0.5, weight_decay=5e-4, with_relu=False, with_bias=True,
        #                     device=self.device).to(self.device)
        #     surrogate.fit(self.features, self.adj, self.labels, self.train_index)
        #     # Setup Attack Model
        #     model = Metattack(surrogate, nnodes=self.adj.shape[0], feature_shape=self.features.shape,
        #                       attack_structure=True, attack_features=False, device=self.device, lambda_=0).to(
        #         self.device)
        #     # Attack
        #     model.attack(self.features.clone().cpu(), self.adj.clone().cpu(), self.labels.clone().cpu(),
        #                  self.train_index, idx_unlabeled,
        #                  n_perturbations=self.n_edge_mod,
        #                  ll_constraint=False)
        #     modified_adj = model.modified_adj
        #     modified_adj, _ = dense_to_sparse(modified_adj)
        #     torch.save(modified_adj, adj_path)
        # adj_attack = modified_adj.to(self.device)
        # return adj_attack
            features = self.features
            labels = self.labels
            train_index = self.train_index
            val_index = self.val_index
            test_index = self.test_index
            adj = self.adj

            batch_size = 8192
            n_batches = int(np.ceil(features.shape[0] / batch_size))
            surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
                            nhid=16, dropout=0.5, weight_decay=5e-4, with_relu=False, with_bias=True,
                            device=self.device).to(self.device)
            modified_adj_list = []

            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, features.shape[0])

                batch_features = features[start_idx:end_idx]
                batch_labels = labels[start_idx:end_idx]
                batch_train_index = train_index[(train_index >= start_idx) & (train_index < end_idx)] - start_idx
                idx_unlabeled = np.union1d(val_index, test_index)
                batch_idx_unlabeled = idx_unlabeled[
                                          (idx_unlabeled >= start_idx) & (idx_unlabeled < end_idx)] - start_idx
                batch_adj = adj[start_idx:end_idx, start_idx:end_idx]
                batch_adj_tensor = torch.tensor(batch_adj.todense(), dtype=torch.float32).to(self.device)

                surrogate.fit(batch_features, batch_adj_tensor, batch_labels, batch_train_index)

                model = Metattack(surrogate, nnodes=batch_adj_tensor.shape[0], feature_shape=batch_features.shape,
                                  attack_structure=True, attack_features=False, device=self.device, lambda_=0).to(
                    self.device)
                model.attack(batch_features.clone().cpu(), batch_adj_tensor.clone().cpu(), batch_labels.clone().cpu(),
                             batch_train_index, batch_idx_unlabeled, n_perturbations=self.n_edge_mod // n_batches,
                             ll_constraint=False)

                modified_adj = model.modified_adj
                modified_adj, _ = dense_to_sparse(modified_adj)
                modified_adj_list.append(modified_adj)
            modified_adj = torch.cat(modified_adj_list, dim=1)
            torch.save(modified_adj, adj_path)
        adj_attack = modified_adj.to(self.device)
        return adj_attack

    def MinMax(self, RP):
        adj_name = "_".join(["MM", self.dataset, str(int(self.attack_ratio * 100)), str(RP)]) + ".pt"
        adj_path = osp.join("./atk_data/atk_adj", adj_name)
        if osp.exists(adj_path):
            modified_adj = torch.load(adj_path, map_location=self.device)
        else:
            # Setup Victim Model
            # self.adj = torch.tensor(self.adj.todense(), dtype=torch.float32)
            # victim_model = GCN(nfeat=self.features.shape[1], nclass=self.labels.max().item() + 1,
            #                    nhid=16, dropout=0.5, weight_decay=5e-4, device=self.device).to(self.device)
            # victim_model.fit(self.features, self.adj, self.labels, self.train_index)
            # # Setup Attack Model
            # model = MinMax(model=victim_model, nnodes=self.adj.shape[0], loss_type='CE',
            #                device=self.device).to(self.device)
            # model.attack(self.features.clone().cpu(), self.adj.clone().cpu(), self.labels.clone().cpu(),
            #              self.train_index, n_perturbations=self.n_edge_mod)
            # modified_adj = model.modified_adj
            # modified_adj, _ = dense_to_sparse(modified_adj)
            # torch.save(modified_adj, adj_path)
            features = self.features
            labels = self.labels
            train_index = self.train_index
            adj = self.adj

            batch_size = 5000
            n_batches = int(np.ceil(features.shape[0] / batch_size))
            victim_model = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
                               nhid=16, dropout=0.5, weight_decay=5e-4, device=self.device).to(self.device)
            modified_adj_list = []

            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, features.shape[0])

                batch_features = features[start_idx:end_idx]
                batch_labels = labels[start_idx:end_idx]
                batch_train_index = train_index[(train_index >= start_idx) & (train_index < end_idx)] - start_idx

                batch_adj = adj[start_idx:end_idx, start_idx:end_idx]

                batch_adj_tensor = torch.tensor(batch_adj.todense(), dtype=torch.float32).to(self.device)

                victim_model.fit(batch_features, batch_adj_tensor, batch_labels, batch_train_index)
                attack_model = MinMax(model=victim_model, nnodes=end_idx - start_idx, loss_type='CE',
                                      device=self.device).to(self.device)
                attack_model.attack(batch_features.clone().cpu(), batch_adj_tensor.clone().cpu(),
                                    batch_labels.clone().cpu(), batch_train_index,
                                    n_perturbations=self.n_edge_mod // n_batches)

                # 获取和处理修改后的邻接矩阵
                modified_adj = attack_model.modified_adj
                modified_adj, _ = dense_to_sparse(modified_adj)
                modified_adj_list.append(modified_adj)
            modified_adj = torch.cat(modified_adj_list, dim=1)
            torch.save(modified_adj, adj_path)
        adj_attack = modified_adj
        return adj_attack


class DICE(BaseAttack):
    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=False, device='cpu'):
        super(DICE, self).__init__(model, nnodes, attack_structure=attack_structure,
                                   attack_features=attack_features, device=device)

        assert not self.attack_features, 'DICE does NOT support attacking features'

    def attack(self, ori_adj, labels, n_perturbations, index_target, **kwargs):
        modified_adj = ori_adj.tolil()

        remove_or_insert = np.random.choice(2, n_perturbations)
        n_remove = sum(remove_or_insert)

        indices = sp.triu(modified_adj).nonzero()
        # Remove edges of the same label
        possible_indices = [x for x in zip(indices[0], indices[1])
                            if labels[x[0]] == labels[x[1]] and x[0] in index_target or x[1] in index_target]

        remove_indices = np.random.permutation(possible_indices)[: n_remove]
        modified_adj[remove_indices[:, 0], remove_indices[:, 1]] = 0
        modified_adj[remove_indices[:, 1], remove_indices[:, 0]] = 0

        n_insert = n_perturbations - n_remove

        # sample edges to add
        added_edges = 0
        while added_edges < n_insert:
            n_remaining = n_insert - added_edges

            # sample random pairs
            candidate_edges = np.array([np.random.choice(ori_adj.shape[0], n_remaining),
                                        np.random.choice(ori_adj.shape[0], n_remaining)]).T

            # filter out existing edges, and pairs with the different labels
            # source node or target node are in the target_index set
            candidate_edges = set([(u, v) for u, v in candidate_edges if labels[u] != labels[v]
                                   and modified_adj[u, v] == 0 and modified_adj[v, u] == 0 and (
                                               u in index_target or v in index_target)])
            candidate_edges = np.array(list(candidate_edges))

            # if none is found, try again
            if len(candidate_edges) == 0:
                continue

            # add all found edges to your modified adjacency matrix
            modified_adj[candidate_edges[:, 0], candidate_edges[:, 1]] = 1
            modified_adj[candidate_edges[:, 1], candidate_edges[:, 0]] = 1
            added_edges += candidate_edges.shape[0]

        self.check_adj(modified_adj)
        self.modified_adj = modified_adj


class Random(BaseAttack):
    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=False, device='cpu'):
        super(Random, self).__init__(model, nnodes, attack_structure=attack_structure,
                                     attack_features=attack_features, device=device)

        assert not self.attack_features

    def attack(self, ori_adj, n_perturbations, **kwargs):
        modified_adj = ori_adj.tolil()

        remove_or_insert = np.random.choice(2, n_perturbations)
        n_remove = sum(remove_or_insert)

        nonzero = set(zip(*ori_adj.nonzero()))
        indices = sp.triu(modified_adj).nonzero()
        possible_indices = [x for x in zip(indices[0], indices[1])]

        remove_indices = np.random.permutation(possible_indices)[: n_remove]
        modified_adj[remove_indices[:, 0], remove_indices[:, 1]] = 0
        modified_adj[remove_indices[:, 1], remove_indices[:, 0]] = 0

        n_insert = n_perturbations - n_remove

        # sample edges to add
        added_edges = 0
        while added_edges < n_insert:
            n_remaining = n_insert - added_edges

            # sample random pairs
            candidate_edges = np.array([np.random.choice(ori_adj.shape[0], n_remaining),
                                        np.random.choice(ori_adj.shape[0], n_remaining)]).T

            # filter out existing edges
            candidate_edges = set([(u, v) for u, v in candidate_edges if
                                   modified_adj[u, v] == 0 and modified_adj[v, u] == 0])
            candidate_edges = np.array(list(candidate_edges))

            # if none is found, try again
            if len(candidate_edges) == 0:
                continue

            # add all found edges to your modified adjacency matrix
            modified_adj[candidate_edges[:, 0], candidate_edges[:, 1]] = 1
            modified_adj[candidate_edges[:, 1], candidate_edges[:, 0]] = 1
            added_edges += candidate_edges.shape[0]

        self.check_adj(modified_adj)
        self.modified_adj = modified_adj
