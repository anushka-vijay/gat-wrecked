import torch
import json
import zipfile
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.data import Data, IterableDataset
from ogb.nodeproppred import PygNodePropPredDataset

# --- 1. VARMISUSE STREAMING LOADER ---

class VarMisuseStreamingDataset(IterableDataset):
    def __init__(self, zip_path):
        super().__init__()
        self.zip_path = zip_path

    def __iter__(self):
        # We open the zip and stream JSON files one-by-one to save RAM
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            for file_info in z.infolist():
                if file_info.filename.endswith('.json'):
                    with z.open(file_info) as f:
                        yield self.parse_graph(json.load(f))

    def parse_graph(self, data):
        # Standard conversion to PyG Data object
        return Data(
            x=torch.tensor(data['node_features'], dtype=torch.float),
            edge_index=torch.tensor(data['edge_index'], dtype=torch.long),
            y=torch.tensor(data['label'], dtype=torch.long)
        )


# --- 3. THE FACTORY FUNCTION ---

def get_loader(name, path, batch_size=32, noise_ratio=0.0):
    """
    Returns the appropriate loader based on dataset name.
    """
    if name == 'varmisuse':
        dataset = VarMisuseStreamingDataset(path)
        return DataLoader(dataset, batch_size=batch_size)

    elif name in ['ogbn-arxiv', 'ogbn-proteins']:
        # Load from OGB Library
        dataset = PygNodePropPredDataset(name=name, root=path)
        data = dataset[0]
        

        # Use NeighborLoader for giant OGB graphs
        return NeighborLoader(
            data,
            num_neighbors=[15, 10], # Sample 2 hops
            batch_size=batch_size,
            input_nodes=data.train_mask if hasattr(data, 'train_mask') else None,
            shuffle=True
        )
    
    else:
        raise ValueError(f"Unknown dataset: {name}")