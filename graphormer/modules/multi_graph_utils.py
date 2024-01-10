import numpy as np
import pandas as pd
import networkx as nx
import torch
import copy
import itertools

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env
from matbench.bench import MatbenchBenchmark

OFFSET_LIST = [
    [-1, -1, -1],
    [-1, -1, 0],
    [-1, -1, 1],
    [-1, 0, -1],
    [-1, 0, 0],
    [-1, 0, 1],
    [-1, 1, -1],
    [-1, 1, 0],
    [-1, 1, 1],
    [0, -1, -1],
    [0, -1, 0],
    [0, -1, 1],
    [0, 0, -1],
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, -1],
    [0, 1, 0],
    [0, 1, 1],
    [1, -1, -1],
    [1, -1, 0],
    [1, -1, 1],
    [1, 0, -1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, -1],
    [1, 1, 0],
    [1, 1, 1],
]

def lattice_params_to_matrix(lengths, angles):
    """
    Batched torch version to compute lattice matrix from params.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/core/lattice.py#L311

    
    :param lengths: torch.Tensor of shape (N, 3), unit A
    :param angles: torch.Tensor of shape (N, 3), unit degree
    :return: torch.Tensor of shape (N, 3, 3)
    """
    angles_r = torch.deg2rad(angles) # transform to radian
    coses = torch.cos(angles_r) 
    sins = torch.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1]) 
    # Sometimes rounding errors result in values slightly > 1.
    val = torch.clamp(val, -1., 1.)
    gamma_star = torch.arccos(val)

    vector_a = torch.stack([ 
        lengths[:, 0] * sins[:, 1],
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 0] * coses[:, 1]], dim=1)
    vector_b = torch.stack([
        -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
        lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
        lengths[:, 1] * coses[:, 0]], dim=1)
    vector_c = torch.stack([
        torch.zeros(lengths.size(0), device=lengths.device),
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 2]], dim=1)  

    matrix = torch.stack([vector_a, vector_b, vector_c], dim=1) 

    return matrix

def frac_to_cart_coords(
    frac_coords,
    lengths,
    angles,
    num_atoms,
    cal_extend_pos=False,
    cell_offsets=None,
    edge_index=None,
    num_bonds=None
):
    """
    Convert fractional coordinates to cartesian coordinates.
    if cal_extend_pos is True, then calculate the cartesian coordinates of the extended atoms

    :param frac_coords: torch.Tensor of shape (N, 3)
    :param lengths: torch.Tensor of shape (N, 3)
    :param angles: torch.Tensor of shape (N, 3)
    :param num_atoms: torch.Tensor of shape (N,)
    :param cal_extend_pos: bool
    :param cell_offsets: torch.Tensor of shape (N, 3)
    :param edge_index: torch.Tensor of shape (2, M)
    :param num_bonds: torch.Tensor of shape (N,), the number of bonds for each batch
    :return: pos:torch.Tensor of shape (N, 3), the cartesian coordinates of the atoms
    """
    lattice = lattice_params_to_matrix(lengths, angles) 
    lattice_nodes = torch.repeat_interleave(lattice, num_atoms, dim=0) # every atom in different batch_size has the same lattice
    frac_coords = frac_coords.unsqueeze(1)  # [N, 1, 3]
    pos = torch.bmm(frac_coords, lattice_nodes)  # [N, 1, 3]
    pos = pos.squeeze(1)  # [N, 3]
    if cal_extend_pos:
        assert cell_offsets is not None
        assert edge_index is not None
        assert num_bonds is not None
        edge_index = edge_index[0]

        extend_pos = torch.index_select(pos, 0, edge_index)

        lattice_nodes = torch.repeat_interleave(lattice, num_bonds, dim=0)
        pos_offset = torch.bmm(cell_offsets.unsqueeze(1), lattice_nodes)
        extend_pos = extend_pos + pos_offset.squeeze(1)
        return pos, extend_pos
        


    return pos 

def multi_graph(data, radius,device):
    """
    Compute the multi-graph for a batch of crystals.

    :param data: Batch of Data
    :param radius: float, the radius of the sphere
    :param device: torch.device
    :return: edge_index: torch.Tensor of shape (2, M), the edge index of the graph, where M is the number of edges
             cell_offsets: torch.Tensor of shape (M, 3), the cell offsets of the graph for extended atoms
             neighbors: torch.Tensor of shape (N,), the number of neighbors for each batch
    """
    cart_coords = frac_to_cart_coords(
        data.frac_coords, data.lengths, data.angles, data.num_atoms
        )
    return pbc_check(
        cart_coords, 
        data.lengths, 
        data.angles, 
        data.num_atoms, 
        radius,
        -1, # not finished
        device
    )

def extend_atom(index2, batch_size, device):
    """
    Get the index and position of the extended atoms.

    :param index2: torch.Tensor of shape (M,), the index of the atoms
    :param batch_size: int, the number of batches
    :param device: torch.device
    :return: unit_cell_batch: torch.Tensor of shape (batch_size, 3, 27), the unit cell offsets for each batch
                unit_cell_per_atom: torch.Tensor of shape (M, 3), the unit cell offsets for each atom
                num_cells: int, the number of cells
    """
    unit_cell = torch.tensor(OFFSET_LIST, device=device).float() # 27 possible unit cell offsets
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(
        len(index2), 1, 1
    ) # every pair of atoms has 27 possible unit cell offsets 
    unit_cell = torch.transpose(unit_cell, 0, 1) # transpose for batched matrix multiplication
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(
        batch_size, -1, -1
    ) # [batch_size, 3, 27] 
    return unit_cell_batch, unit_cell_per_atom, num_cells



def pbc_check(cart_coords, lengths, angles, num_atoms,
                     radius, max_num_neighbors, device):
    """
    Compute the multi-graph for a batch of crystals.

    :param cart_coords: torch.Tensor of shape (N, 3), the cartesian coordinates of the atoms
    :param lengths: torch.Tensor of shape (N, 3)
    :param angles: torch.Tensor of shape (N, 3)
    :param num_atoms: torch.Tensor of shape (N,)
    :param radius: float, the radius of the sphere
    :param max_num_neighbors: int, the maximum number of neighbors
    :param device: torch.device
    :return: edge_index: torch.Tensor of shape (2, M), the edge index of the graph, where M is the number of edges,,from j to i
                cell_offsets: torch.Tensor of shape (M, 3), the cell offsets of the graph for extended atoms
                neighbors: torch.Tensor of shape (N,), the number of neighbors for each batch
    """
    
    batch_size = len(num_atoms)

    num_atoms_per_batch = num_atoms # shape: (batch_size,)
    num_atoms_per_batch_sqr = (num_atoms_per_batch ** 2).long() # atom pairs per batch

    # index offset between batches
    index_offset = (
        torch.cumsum(num_atoms_per_batch, dim=0) - num_atoms_per_batch 
    )

    index_offset_expand = torch.repeat_interleave( 
        index_offset, num_atoms_per_batch_sqr
    ) # index offset between batches for each atom pair

    num_atoms_per_batch_expand = torch.repeat_interleave(
        num_atoms_per_batch, num_atoms_per_batch_sqr
    ) # the number of atoms per batches for each atom pair

    num_atom_pairs = torch.sum(num_atoms_per_batch_sqr)
    index_sqr_offset = (
        torch.cumsum(num_atoms_per_batch_sqr, dim=0) - num_atoms_per_batch_sqr
    ) 
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_batch_sqr
    ) 
    atom_count_sqr = ( 
        torch.arange(num_atom_pairs, device=device) - index_sqr_offset
    ) # for example, [0,1,2,3,4,5,6,7,8,0,1,2,3] for two batches,showing the index of atom pairs in each batch

    # Compute the indices for the pairs of atoms (using division and mod)
    index1 = (
        (atom_count_sqr // num_atoms_per_batch_expand)
    ).long() + index_offset_expand # first atom index for each atom pair
    index2 = (
        atom_count_sqr % num_atoms_per_batch_expand
    ).long() + index_offset_expand  # second atom index for each atom pair
    # Get the positions for each atom
    pos1 = torch.index_select(cart_coords, 0, index1) 
    pos2 = torch.index_select(cart_coords, 0, index2) 

    # lattice matrix
    lattice = lattice_params_to_matrix(lengths, angles) # [batch_size, 3, 3]


    unit_cell_batch, unit_cell_per_atom, num_cells = extend_atom(
        index2, batch_size, device
    ) # unit_cell_batch is the 27 possible unit cell offsets for each batch

    # Compute the x, y, z positional offsets for each cell in each batch
    data_cell = torch.transpose(lattice, 1, 2) # transpose for batched matrix multiplication
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch) # [batch_size, 3, 27]
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_batch_sqr, dim=0
    ) 

    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells) # shape: (num_atom_pairs, 3, 27) pos1 is fixed which presents the center of the cell
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1) # shape: (num_atom_pairs * 27,)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1) 

    pos2 = pos2 + pbc_offsets_per_atom  # offset

    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1) # [num_atom_pairs,27]

    atom_distance_sqr = atom_distance_sqr.view(-1) 

    mask_within_radius = torch.le(atom_distance_sqr, radius * radius) 
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001) 
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(
        unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3) 

    num_neighbors = torch.zeros(len(cart_coords), device=device)
    num_neighbors.index_add_(0, index1, torch.ones(len(index1), device=device)) # use index1 to index num_neighbors, and add 1 to each index
    num_neighbors = num_neighbors.long()
    max_num_nb = torch.max(num_neighbors).long() # 


    _max_neighbors = copy.deepcopy(num_neighbors)
    if max_num_neighbors > 0:
        _max_neighbors[
            _max_neighbors > max_num_neighbors
        ] = max_num_neighbors  
    _num_neighbors = torch.zeros(len(cart_coords) + 1, device=device).long() 
    _natoms = torch.zeros(num_atoms.shape[0] + 1, device=device).long() # batch_size+1
    _num_neighbors[1:] = torch.cumsum(_max_neighbors, dim=0) 
    _natoms[1:] = torch.cumsum(num_atoms, dim=0) 
    num_neighbors_image = (
        _num_neighbors[_natoms[1:]] - _num_neighbors[_natoms[:-1]]
    ) 

    
    if (
        max_num_nb <= max_num_neighbors
        or max_num_neighbors <= 0
    ):
        return torch.stack((index2, index1)), unit_cell, num_neighbors_image
    

def get_extend_atom_index_and_pos(edge_index, pos,num_atoms):
    """
    Get the index and position of the extended atoms.

    :param edge_index: torch.Tensor of shape (2, M), the edge index of the graph, where M is the number of edges
    :param pos: torch.Tensor of shape (N, 3), the cartesian coordinates of the atoms
    :param num_atoms: torch.Tensor of shape (N,), the number of atoms for each batch
    :return: atom: list of torch.Tensor, the index of the atoms for each batch
                pos2: list of torch.Tensor, the cartesian coordinates of the atoms for each batch
    """
    edge_index = edge_index[0]
    combined = torch.cat((edge_index.unsqueeze(1), pos), dim=1)

    combined_unique, indices = torch.unique(combined, dim=0, return_inverse=True)

    expend_index = torch.cumsum(num_atoms, dim=0)

   
    bins = torch.searchsorted(expend_index-1, combined_unique[:, 0])
    
    splits = [combined_unique[bins == i] for i in range(len(expend_index))]

    atom = [split[:,0].long() for split in splits]
    pos2 = [split[:,1:] for split in splits]

    return atom, pos2

def pad_1d(samples, fill=0, multiplier=8):
    """
    Pad a list of 1D tensors to the same length.

    :param samples: list of torch.Tensor
    :param fill: int, the value to fill
    :param multiplier: int, the multiplier
    :return: torch.Tensor of shape (N, max_len, *samples[0].shape[1:])
    """
    max_len = max(x.size(0) for x in samples)
    max_len = (max_len + multiplier - 1) // multiplier * multiplier
    n_samples = len(samples)
    out = torch.full(
        (n_samples, max_len, *samples[0].shape[1:]), fill, dtype=samples[0].dtype
    )
    for i in range(n_samples):
        x_len = samples[i].size(0)
        out[i][:x_len] = samples[i]
    return out

def pad_pos(samples,lengths,fill=0):
    """
    Pad a list of tensors to the same length.

    :param samples: list of torch.Tensor
    :param lengths: int, the length to pad
    :param fill: int, the value to fill
    :return: torch.Tensor of shape (N, max_len, 3)
    """

    out = torch.full(
        (len(samples), lengths, 3), fill, dtype=samples[0].dtype
    )
    for i in range(len(samples)):
        x_len = samples[i].size(0)
        out[i][:x_len] = samples[i]
    return out



#---------------------------------------for test---------------------------------------
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch.utils.data import Dataset
import random
def build_crystal_graph(crystal,scale_length=False):
    """
    """
    graph_arrays = {}

    graph_arrays["frac_coords"] = crystal.frac_coords
    atom_types = crystal.atomic_numbers
    lattice_parameters = crystal.lattice.parameters
    graph_arrays["lattice_matrix"] = crystal.lattice.matrix
    lengths = lattice_parameters[:3]
    angles = lattice_parameters[3:]

    graph_arrays["atom_types"] = np.array(atom_types)
    graph_arrays["lengths"], graph_arrays["angles"] = np.array(lengths), np.array(angles)
    graph_arrays["num_atoms"] = graph_arrays["atom_types"].shape[0]

    if scale_length:
        graph_arrays["scale_length"] = lengths / float(graph_arrays["num_atoms"])**(1/3)

    return graph_arrays


def preprocess(data:pd.DataFrame,num_workers):

    def process_one(row):
        material_id = row.name
        crystal = row['structure']
        property = row[-1]
        graph_arrays = build_crystal_graph(crystal)
        result_dict = {
            'mp_id': material_id,
            'graph_arrays': graph_arrays,
            'property': property,
        }
        return result_dict
    
    result = data.apply(process_one, axis=1)
    result = list(result)

    return result

class CrystDataset(Dataset):
    def __init__(self,data,pad_fill=0):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        data_dict = self.data[index]

        graph_dict = data_dict['graph_arrays']

        data = Data(
            frac_coords=torch.Tensor(graph_dict["frac_coords"]),
            atom_types=torch.LongTensor(graph_dict["atom_types"]),
            lengths=torch.Tensor(graph_dict["lengths"]).view(1, -1),
            angles=torch.Tensor(graph_dict["angles"]).view(1, -1),
            num_atoms=graph_dict["num_atoms"],
            num_nodes=graph_dict["num_atoms"],  # special attribute used for batching in pytorch geometric
            y=torch.Tensor([data_dict["property"]]).view(1, -1),
        )

        return data

    def __len__(self):
        return len(self.data)

    def collater(self, samples):
        batch = Batch.from_data_list(samples)
        return batch



def test(dataset):
    crystal_data = dataset[0]
    data = Batch.from_data_list([crystal_data])

    radius = random.randint(3,10)

    edge_index, cell_offsets, neighbors = multi_graph(
        data, radius, data.num_atoms.device
    )
    base_atom_total_j = []
    base_atom_total_i = []

    for i in range(len(edge_index[0])):
        base_atom_total_j.append(data.atom_types[edge_index[0][i].item()])
        base_atom_total_i.append(data.atom_types[edge_index[1][i].item()])

    for i in range(len(dataset)):
        crystal_data = dataset[i]
        atom_total_j = []
        atom_total_i = []
        data = Batch.from_data_list([crystal_data])
        edge_index, cell_offsets, neighbors = multi_graph(
            data, radius, data.num_atoms.device
        )
        atom_total_j = []
        atom_total_i = []
        for j in range(len(edge_index[0])):
            atom_total_j.append(data.atom_types[edge_index[0][j].item()])
            atom_total_i.append(data.atom_types[edge_index[1][j].item()])
        
        
        assert base_atom_total_i == atom_total_i
        assert base_atom_total_j == atom_total_j

if __name__ == '__main__':
    from pymatgen.core.operations import SymmOp
    mb = MatbenchBenchmark(autoload=False)
    task = mb.tasks
    for task in mb.tasks:
        task.load()
        fold = 0
        test_inputs, test_outputs = task.get_test_data(fold, include_target=True)

        for i in range(50):
            structure = test_inputs[i]
            dataset = []
            for j in range(20):
                x_a = random.uniform(0, 1)
                x_b = random.uniform(0, 1)
                x_c = random.uniform(0, 1)
                op = SymmOp.from_rotation_and_translation(((1, 0, 0), (0, 1, 0), (0, 0, 1)), translation_vec = (x_a, x_b, x_c))
                dataset.append(structure.copy())
                structure.apply_operation(op)

            test_outputs = test_outputs[:len(dataset)]
            dataset = pd.Series(dataset)
            dataset.index = test_outputs.index
            dataset = dataset.to_frame()
            dataset.columns = ['structure']
            test_data = pd.concat([dataset,test_outputs.to_frame()],axis=1)
            test_data = preprocess(test_data,0)
            test_dataset = CrystDataset(test_data)
            test(test_dataset)
        break

    print("Translation Test pass!!!")