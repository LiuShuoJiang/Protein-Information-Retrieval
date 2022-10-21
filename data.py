import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
import numpy as np
import Bio.SeqIO
from esm.inverse_folding.util import load_structure, extract_coords_from_structure, load_coords, BatchConverter

database_path = 'autodl-tmp/swissprot_pdb_v2/'
device = torch.device('cuda')
# device = torch.device('cpu')


class MyProteinDataset(Dataset):
    def __init__(self, names, lines):
        self.names = names
        self.lines = lines

    def __len__(self):
        return len(self.names)

    def get_pairs(self, path, line):
        lines = (line + 1) // 2
        index2 = random.randint(0, lines - 1)
        protein_list = list(Bio.SeqIO.parse(path, "fasta"))
        protein_name1 = database_path + protein_list[0].name
        protein_name2 = database_path + protein_list[index2 * 2].name
        # structure1 = load_structure(protein_name1)
        # structure2 = load_structure(protein_name2)
        coordinates1 = list(load_coords(protein_name1, chain=None))
        coordinates2 = list(load_coords(protein_name2, chain=None))
        
        if len(coordinates1[1]) > 512:
            coordinates1[0] = coordinates1[0][:512, :]
            coordinates1[1] = coordinates1[1][:512]
        
        if len(coordinates2[1]) > 512:
            coordinates2[0] = coordinates2[0][:512, :]
            coordinates2[1] = coordinates2[1][:512]

        return (coordinates1[0], None, coordinates1[1]), (coordinates2[0], None, coordinates2[1])

    def __getitem__(self, index):
        protein1, protein2 = self.get_pairs(self.names[index], self.lines[index])
        return protein1, protein2

    def get_batch_indices(self, batch_size):
        batches = []
        iters = len(self.names) // batch_size

        for i in range(iters):
            buffer = random.sample(range(len(self.names)), batch_size)
            batches.append(buffer)

        return batches


class MyBatchConverter(BatchConverter):
    def __call__(self, raw_batch, device=device):
        self.alphabet.cls_idx = self.alphabet.get_idx("<cath>")
        batch1 = []
        batch2 = []
        for (coords1, confidence1, seq1), (coords2, confidence2, seq2) in raw_batch:
            if confidence1 is None:
                confidence1 = 1.
            if isinstance(confidence1, float) or isinstance(confidence1, int):
                confidence1 = [float(confidence1)] * len(coords1)
            if seq1 is None:
                seq1 = 'X' * len(coords1)
            batch1.append(((coords1, confidence1), seq1))

            if confidence2 is None:
                confidence2 = 1.
            if isinstance(confidence2, float) or isinstance(confidence2, int):
                confidence2 = [float(confidence2)] * len(coords2)
            if seq2 is None:
                seq2 = 'X' * len(coords2)
            batch2.append(((coords2, confidence2), seq2))

        coords_and_confidence1, strs1, tokens1 = super().__call__(batch1)
        coords_and_confidence2, strs2, tokens2 = super().__call__(batch2)

        # pad beginning and end of each protein due to legacy reasons
        coords1 = [
            F.pad(torch.tensor(cd), (0, 0, 0, 0, 1, 1), value=100000)
            for cd, _ in coords_and_confidence1
        ]
        confidence1 = [
            F.pad(torch.tensor(cf), (1, 1), value=-1.)
            for _, cf in coords_and_confidence1
        ]
        coords1 = self.collate_dense_tensors(coords1, pad_v=np.nan)
        confidence1 = self.collate_dense_tensors(confidence1, pad_v=-1.)
        if device is not None:
            coords1 = coords1.to(device)
            confidence1 = confidence1.to(device)
            tokens1 = tokens1.to(device)
        padding_mask1 = torch.isnan(coords1[:, :, 0, 0])
        coord_mask1 = torch.isfinite(coords1.sum(-2).sum(-1))
        confidence1 = confidence1 * coord_mask1 + (-1.) * padding_mask1

        coords2 = [
            F.pad(torch.tensor(cd), (0, 0, 0, 0, 1, 1), value=np.inf)
            for cd, _ in coords_and_confidence2
        ]
        confidence2 = [
            F.pad(torch.tensor(cf), (1, 1), value=-1.)
            for _, cf in coords_and_confidence2
        ]
        coords2 = self.collate_dense_tensors(coords2, pad_v=np.nan)
        confidence2 = self.collate_dense_tensors(confidence2, pad_v=-1.)
        if device is not None:
            coords2 = coords2.to(device)
            confidence2 = confidence2.to(device)
            tokens2 = tokens2.to(device)
        padding_mask2 = torch.isnan(coords2[:, :, 0, 0])
        coord_mask2 = torch.isfinite(coords2.sum(-2).sum(-1))
        confidence2 = confidence2 * coord_mask2 + (-1.) * padding_mask2

        return (coords1, confidence1, strs1, tokens1, padding_mask1), \
               (coords2, confidence2, strs2, tokens2, padding_mask2)

    def from_lists(self, coords_list, confidence_list=None, seq_list=None, device=None):
        """
        Args:
            coords_list: list of length batch_size, each item is a list of
            floats in shape L x 3 x 3 to describe a backbone
            confidence_list: one of
                - None, default to the highest confidence
                - list of length batch_size, each item is a scalar
                - list of length batch_size, each item is a list of floats of
                    length L to describe the confidence scores for the backbone
                    with values between 0. and 1.
            seq_list: either None or a list of strings
        Returns:
            coords: Tensor of shape batch_size x L x 3 x 3
            confidence: Tensor of shape batch_size x L
            strs: list of strings
            tokens: LongTensor of shape batch_size x L
            padding_mask: ByteTensor of shape batch_size x L
        """
        batch_size = len(coords_list)
        if confidence_list is None:
            confidence_list = [None] * batch_size
        if seq_list is None:
            seq_list = [None] * batch_size
        raw_batch = zip(coords_list, confidence_list, seq_list)
        return self.__call__(raw_batch, device)

    @staticmethod
    def collate_dense_tensors(samples, pad_v):
        """
        Takes a list of tensors with the following dimensions:
            [(d_11,       ...,           d_1K),
             (d_21,       ...,           d_2K),
             ...,
             (d_N1,       ...,           d_NK)]
        and stack + pads them into a single tensor of:
        (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
        """
        if len(samples) == 0:
            return torch.Tensor()
        if len(set(x.dim() for x in samples)) != 1:
            raise RuntimeError(
                f"Samples has varying dimensions: {[x.dim() for x in samples]}"
            )
        (device,) = tuple(set(x.device for x in samples))  # assumes all on same device
        max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
        result = torch.empty(
            len(samples), *max_shape, dtype=samples[0].dtype, device=device
        )
        result.fill_(pad_v)
        for i in range(len(samples)):
            result_i = result[i]
            t = samples[i]
            result_i[tuple(slice(0, k) for k in t.shape)] = t
        return result
