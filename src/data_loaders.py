import zipfile
import gzip
import json
import io
import torch
import random
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.data import Data, Dataset
from ogb.nodeproppred import PygNodePropPredDataset

# --- 1. VARMISUSE DATASET ---

class VarMisuseDataset(Dataset):
    def __init__(self, zip_path, subset_size=500, vocab=None, split='train'):
        """
        split (str): 'train', 'valid', or 'test'
        """
        super().__init__(None, None)
        self.graphs = []
        self.vocab = vocab if vocab is not None else {"<UNK>": 0}
        
        forbidden_projects = ['/commandline/', '/humanizer/', '/lean/']
        print(f"🚀 Initializing {split.upper()} Dataset. Extracting {subset_size} graphs...")
        
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Dynamically look for train, valid, or test folders
            target_folder = f'/graphs-{split}/'
            
            graph_files = [
                f for f in z.namelist() 
                if target_folder in f 
                and f.endswith('.gz')
                and not any(forbidden in f for forbidden in forbidden_projects)
            ]
            random.seed(42)
            random.shuffle(graph_files)
            
            for file_name in graph_files:
                if len(self.graphs) >= subset_size:
                    break
                    
                compressed_bytes = z.read(file_name)
                with gzip.open(io.BytesIO(compressed_bytes), 'rt', encoding='utf-8') as f:
                    raw_graphs_list = json.load(f)
                    
                for raw_graph in raw_graphs_list:
                    if len(self.graphs) >= subset_size:
                        break
                        
                    ctx = raw_graph['ContextGraph']
                    
                    # --- A. NODES (Features) ---
                    node_ids = [int(n) for n in ctx['NodeTypes'].keys()]
                    num_nodes = max(node_ids) + 1 if node_ids else 0
                    node_features = [0] * num_nodes 
                    
                    for str_node_id, node_type_string in ctx['NodeTypes'].items():
                        idx = int(str_node_id)
                        if node_type_string not in self.vocab:
                            # Only add to vocab if we are in training mode
                            if split == 'train':
                                self.vocab[node_type_string] = len(self.vocab)
                            else:
                                # For valid/test, map unseen words to <UNK> (0)
                                node_features[idx] = 0 
                                continue
                        node_features[idx] = self.vocab.get(node_type_string, 0)
                        
                    x = torch.tensor(node_features, dtype=torch.long)
                    
                    # --- B. EDGES (Connectivity) ---
                    src_list, dst_list, type_list = [], [], []
                    edge_types_dict = {name: i for i, name in enumerate(ctx['Edges'].keys())}
                    
                    for edge_type_str, edges in ctx['Edges'].items():
                        type_idx = edge_types_dict[edge_type_str]
                        if isinstance(edges, list):
                            for edge_pair in edges:
                                if len(edge_pair) >= 2:
                                    src_list.append(int(edge_pair[0]))
                                    dst_list.append(int(edge_pair[1]))
                                    type_list.append(type_idx)
                                    
                    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
                    edge_attr = torch.tensor(type_list, dtype=torch.long)

                    # --- C. TASK LABELS ---
                    slot_node = torch.tensor([raw_graph['SlotDummyNode']], dtype=torch.long)
                    candidate_nodes = []
                    correct_idx = 0 
                    
                    for i, candidate in enumerate(raw_graph['SymbolCandidates']):
                        if isinstance(candidate, dict):
                            node_id = candidate.get('SymbolDummyNode', candidate.get('NodeId'))
                            candidate_nodes.append(int(node_id))
                            if candidate.get('IsCorrect', False) == True:
                                correct_idx = i
                        else:
                            candidate_nodes.append(int(candidate))
                            
                    candidates = torch.tensor(candidate_nodes, dtype=torch.long)
                    y = torch.tensor([correct_idx], dtype=torch.long)
                    
                    data = Data(
                        x=x, edge_index=edge_index, edge_attr=edge_attr,
                        slot_node=slot_node, candidates=candidates, y=y
                    )
                    self.graphs.append(data)
                    
        print(f"✅ Successfully loaded {len(self.graphs)} pure tensor graphs!")

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]


# --- 2. THE FACTORY FUNCTION ---

def get_loader(name, path, batch_size=32, split='train', subset_size=1000, vocab=None):
    """
    Returns the appropriate loader based on dataset name.
    """
    if name == 'varmisuse':
        dataset = VarMisuseDataset(
            zip_path=path, 
            subset_size=subset_size, 
            vocab=vocab, 
            split=split
        )
        
        # Only shuffle the training data
        is_train = (split == 'train')
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train)
        
        # We return the loader AND the dataset so you can extract the vocab for the next split!
        return loader, dataset

    elif name in ['ogbn-arxiv', 'ogbn-proteins']:
        # Load from OGB Library
        dataset = PygNodePropPredDataset(name=name, root=path)
        data = dataset[0]
        
        # Use NeighborLoader for giant OGB graphs
        loader = NeighborLoader(
            data,
            num_neighbors=[15, 10], # Sample 2 hops
            batch_size=batch_size,
            input_nodes=data.train_mask if hasattr(data, 'train_mask') else None,
            shuffle=True
        )
        return loader, dataset
    else:
        raise ValueError(f"Unknown dataset: {name}")
