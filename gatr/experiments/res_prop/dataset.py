# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
# Adapted by Ruibin Liu, ruibinliuphd@gmail.com
import numpy as np
import torch

class ResPropDataset(torch.utils.data.Dataset):
    """Protein/DNA/RNA residue property prediction dataset.

    Loads data generated with generate_res_prop_dataset.py from disk.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the npz file with the dataset to be loaded.
    subsample : None or float
        If not None, defines the fraction of the dataset to be used. For instance, `subsample=0.1`
        uses just 10% of the samples in the dataset.
    label_type: torch.float32
        This should be chosen as to the loss function. Simple examples below:
            1. Binary classification & last layer Sigmoid & BCELoss --> torch.float32
            2. Binary classification & last layer Linear(n, 1) & BCEWithLogitsLoss --> torch.float32, better than 1
            3. k-class classification & last layer Linear(n, k) & CrossEntropyLoss --> torch.long
                apply torch.F.softmax to the output/logits `F.softmax(logits, dim=1)` to get probabilities for each class
                apply torch.max to the output/logits `_, preds = torch.max(logits, 1)` to get predicted class
            4. Regression & last layer Linear(n, 1) & MSELoss() --> torch.float32
    """ # noqa

    def __init__(self, filename, subsample=None, label_type=torch.float32):
        super().__init__()

        self.x, self.y = self._load_data(filename, subsample, label_type=label_type)

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.x)

    def __getitem__(self, idx):
        """Returns the `idx`-th sample from the dataset."""
        return self.x[idx], self.y[idx]

    @staticmethod
    def _load_data(filename, subsample=None, label_type=torch.float32):
        """Loads data from file and converts to input and output tensors."""
        # Load data from file
        npz = np.load(filename, "r")
        atoms, coords, labels, start_indices = (
            npz["atoms"],
            npz["coords"],
            npz["labels"],
            npz["start_indices"],
        )

        # Separate into residues by 'start_indices', pad to a same size, and concatenate them.
        atoms = ResPropDataset._separate_pad_concate(atoms, start_indices)
        coords = ResPropDataset._separate_pad_concate(coords, start_indices)

        # Convert to tensors
        atoms = torch.from_numpy(atoms).to(torch.float32)
        labels = torch.from_numpy(labels).to(label_type)
        coords = torch.from_numpy(coords).to(torch.float32)

        print(atoms.shape, labels.shape, coords.shape)
        # Concatenate into inputs and outputs
        x = torch.cat((atoms, coords), dim=2)
        y = labels

        # Subsample
        if subsample is not None and subsample < 1.0:
            n_original = len(x)
            n_keep = int(round(subsample * n_original))
            assert 0 < n_keep <= n_original
            x = x[:n_keep]
            y = y[:n_keep]

        return x, y
    
    @staticmethod
    def _separate_pad_concate(residue_data, residue_start_indices):
        """
        Here are the steps to process the residue_data consisting of residues with varying sizes.
            1. Separate each residue using 'residue_start_indices'.
            2. Pad each residue to the same size as the largest one.
            3. Concatenate all residues
        """
        # Separate 
        object_orign_sizes = residue_start_indices[1:] - residue_start_indices[:-1]
        padding_sizes = np.max(object_orign_sizes) - object_orign_sizes
        padded = []
        for i_s, i_e, i_p in zip(residue_start_indices, residue_start_indices[1:], padding_sizes):
            array = residue_data[i_s:i_e]
            padded.append(np.pad(array, ((0, i_p), (0,0)), mode='constant', constant_values=0))
        return np.stack(padded)

