import os
from typing import Optional
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch as th
from torch_geometric.utils import dense_to_sparse

from torch_geometric_temporal.signal import StaticGraphTemporalSignal


class SZTaxiLoader:
    def __init__(self, raw_data_dir: Optional[str] = None) -> None:
        if raw_data_dir is None:
            raw_data_dir = os.path.join(os.getcwd(), "data", "sz_taxi")
        self.raw_data_dir = raw_data_dir
        self._read_web_data()

    def _read_web_data(self) -> None:
        os.makedirs(self.raw_data_dir, exist_ok=True)

        # read adjacency matrix
        adj_path = os.path.join(self.raw_data_dir, "sz_adj.csv")
        if not os.path.exists(adj_path):
            adj_url = "https://raw.githubusercontent.com/lehaifeng/T-GCN/refs/heads/master/data/sz_adj.csv"
            urlretrieve(adj_url, adj_path)
        adj_df = pd.read_csv(adj_path, header=None)
        A = np.array(adj_df, dtype=np.float32)

        # read node's features
        feat_path = os.path.join(self.raw_data_dir, "sz_speed.csv")
        if not os.path.exists(feat_path):
            feat_url = "https://raw.githubusercontent.com/lehaifeng/T-GCN/refs/heads/master/data/sz_speed.csv"
            urlretrieve(feat_url, feat_path)
        feat_df = pd.read_csv(feat_path)
        X = np.array(feat_df, dtype=np.float32)
        X = np.expand_dims(X.T, axis=1)

        # normalize
        max_val = np.max(X)
        X = X / float(max_val + 1e-16)

        self.X = th.from_numpy(X)
        self.A = th.from_numpy(A)

    def _get_edges_and_weights(self) -> None:
        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices.numpy()
        values = values.numpy()
        self.edges = edge_indices
        self.edge_weights = values

    def _generate_task(self, num_timesteps_in: int, num_timesteps_out: int) -> None:
        """Uses the node features of the graph and generates a feature/target
        relationship of the shape
        (num_nodes, num_node_features, num_timesteps_in) -> (num_nodes, num_timesteps_out)
        predicting the average traffic speed using num_timesteps_in to predict the
        traffic conditions in the next num_timesteps_out

        :param num_timesteps_in: number of timesteps the sequence model sees
        :param num_timesteps_out: number of timesteps the sequence model has to predict
        """
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(self.X.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]

        # Generate observations
        features, target = [], []
        for i, j in indices:
            features.append((self.X[:, :, i : i + num_timesteps_in]).numpy())
            target.append((self.X[:, 0, i + num_timesteps_in : j]).numpy())

        self.features = features
        self.targets = target

    def get_dataset(
        self, num_timesteps_in: int = 12, num_timesteps_out: int = 3
    ) -> StaticGraphTemporalSignal:
        """Returns data iterator for SZ-taxi dataset as an instance of the
        static graph temporal signal class.

        :param num_timesteps_in: number of timesteps the sequence model sees
        :param num_timesteps_out: number of timesteps the sequence model has to predict
        :return: SZ-taxi forecasting dataset.
        """
        self._get_edges_and_weights()
        self._generate_task(num_timesteps_in, num_timesteps_out)
        dataset = StaticGraphTemporalSignal(
            self.edges, self.edge_weights, self.features, self.targets
        )

        return dataset
