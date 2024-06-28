#!/usr/bin/env python3
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
# Adapted by Ruibin Liu, ruibinliuphd@gmail.com

from pathlib import Path

import hydra
import numpy as np
import pandas as pd

from gatr.experiments.res_prop import ResPropPreparer

def generate_dataset(filename, preparer, num_samples, seed):
    """Samples residues, extract from PDB files, and stores the results at `filename`."""
    # assert not Path(filename).exists()
    pdb_paths, chain_ids, resids, subunits, labels = preparer.sample(
        num_samples, seed=seed, center=True,
    )
    info_df = pd.DataFrame({
        "pdb_path": pdb_paths,
        "chain_id": chain_ids,
        "resid": resids,
        "label": labels,
        }
    )
    info_df.to_csv(str(filename) + '_info.csv', mode='w', index=False)
    coords = [subunit[['x_coord', 'y_coord', 'z_coord']].values for subunit in subunits]
    coords = np.concatenate(coords, axis=0)
    atoms = [subunit[['atom']].values for subunit in subunits]
    start_indices = np.cumsum([0] + [atom.shape[0] for atom in atoms])
    # use atoms[start_indices[i]:start_indices[i+1]] to get ith's sample's atoms
    atoms = np.concatenate(atoms, axis=0)
    np.savez(
        filename,
        coords=coords,
        atoms=atoms,
        start_indices=start_indices,
        labels=labels,
    )


def generate_datasets(path, residues_file_path, seed=None):
    """Generates a canonical set of datasets for the res_prop problem, stores them in `path`."""
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)

    print(f"Creating gravity dataset in {str(path)}")

    preparer = ResPropPreparer(residues=residues_file_path)
    generate_dataset(path / "train.npz", preparer, 2, seed=seed)
    generate_dataset(path / "val.npz", preparer, 1, seed=seed)
    # generate_dataset(path / "eval.npz", preparer, 5000, seed=seed)
    # generate_dataset(path / "extra.npz", preparer, 5000, seed=seed)

    print("Done, have a nice day!")


@hydra.main(config_path="../config", config_name="res_prop", version_base=None)
def main(cfg):
    """Entry point for res_prop dataset generation."""
    data_dir = cfg.data.data_dir
    generate_datasets(data_dir, seed=cfg.seed, residues_file_path=cfg.data.residues_file_path)


if __name__ == "__main__":
    main()
