# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
# Adapted by Ruibin Liu, ruibinliuphd@gmail.com
import scipy
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import special_ortho_group as sog
from moldf import read_pdb

periodic_table = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
    'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
    'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
    'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
    'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110,
    'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
}
_element_atom_number = lambda x: periodic_table.get(x, 0) # Currently just embeding by atom numbers

class ResPropPreparer:
    """Preparer for the res_prop dataset.

    Each sample consists of the subunit of a protein/DNA/RNA residue dataframe.

    The data is generated in the following way:
    1. Read the PDB file for the structure containing the residue into a Pandas DataFrame.
    2. Check whether the residue is indeed in the structure; if not, log and skip.
    3. Get the subunit of the residue based on provided or default cutting criteria.

    Parameters
    ----------
    residues : str | Path | pd.DataFrame
        All the residues for the current problem. If given a str or Path, it should be the
        path to a CSV file, which can be read into a pd.DataFrame. The DataFrame should
        contain at least the following columns:
            pdb_path: full file path to the PDB file which the residue belongs to.
            chain_id: chain identifier to locate the residue in the PDB file.
            resid: residue number.
            label: data label for the residue.
        and the following optional columns:
            model_id: which model id to use; the first will be used if NOT/WRONGLY provided. 
            alt_id: which alternative_id to use; 'A'/default will be used if NOT/WRONGLY provided.
            residue_name: residue name to double check whether it is the desired residue;
            atom_name: atom name to select as subunit center if provided; otherwise
                the center of geometry is selected as the subunit center.
            group_name: an identifier to group residues so that the residues in a same 'group_name'
                will be in a same sub dataset. If not provided, residues are randomly splitted into
                different sub dataset (train/val/test).
    
    cut_off: float | None
        Distance cutoff to generate the residue subunit. If none, the whole PDB structure is used.

    cut_method: str | None
        Either 'cube' or 'sphere' should be provide; otherwise or None, ignored.
        If 'cube', the 'cut_off' means the distance to to each side. If 'sphere', the 'cut_off'
        means the sphere radius.
    
    atom_mapping: func
        Mapping function to convert each atom identity into a value.
        By default, the atomic number is used. 
    """

    def __init__(
        self,
        residues,
        cut_off=None,
        cut_method=None,
        atom_mapping=_element_atom_number,
    ):
        self.residues = self._residues_to_df(residues)
        self.cut_off= cut_off
        self.cut_method = cut_method
        self.atom_mapping = atom_mapping
        self.prepared_residues = []

    def _residues_to_df(self, residues):
        """Turn and sanitize provided 'residues' to pd.DataFrame.

        Parameters
        ----------
        residues : str | Path | pd.DataFrame
            residues from init.

        Returns
        -------
        df : pd.DataFrame
            residues as DataFrame.
        """
        if isinstance(residues, pd.DataFrame):
            df = residues
        else:
            df = pd.read_csv(residues)

        required_cols = ['pdb_path', 'chain_id', 'resid', 'label']
        for required_col in required_cols:
            if required_col not in df.columns:
                raise ValueError(f"Col {required_col} is required but not provided.")
        
        self.optional_cols = []
        for optional_col in ['model_id', 'alt_id', 'residue_name', 'atom_name', 'group_name']:
            if optional_col in df.columns:
                self.optional_cols.append(optional_col)
        
        return df[required_cols + self.optional_cols]

        
    def sample(self, num_samples, seed=None,
        n_rot=0, n_trans=0, center=False, reflect=False,
        trans_std=10.0,
    ):
        """Samples from res_prop problem.

        Parameters
        ----------
        num_samples : int
            Number of ORIGINAL (and GROUPED) samples. This means if 'group_name' is provided,
            'num_samples' means number of groups; and also means if extra structures are generated
            by rotaiton/translation, total number of samples will be different.
        n_rot: int
            Number of (extra) rotations to perform around the subunit center.
            Just returning the original by default.
        n_trans: int
            Number of (extra) translations to perfrom for the whole subunit.
            Just returning the original by default. 
        center: bool
            Whether to put the center of the subunit as (0, 0, 0).
            Just returning the original by default.
        reflect: bool
            Whether to reflecting by the yz plane (just reversing the x coordinates).
            Just returning the original by default.
        seed: None | int
            Numpy random RNG Generator seed; use one for reproducible results.
        trans_std: 10.0
            Random translation amount std; default is 10.0
        
        The operation order is center -> reflect -> rotation -> translation.
        """
        # 0. Check 'group_name'
        groups = []
        if 'group_name' not in self.optional_cols:
            self.residues['group_name'] = self.residues.index
        groups = self.residues['group_name'].unique().tolist()
        rng = np.random.default_rng(seed)
        rng.shuffle(groups)
        
        # 1. Read the PDB file
        pdb_paths = []
        chain_ids = []
        resids = []
        subunits = []
        labels = []
        n_sampled = 0
        for group in groups:
            if group in self.prepared_residues:
                continue
            grouped_residues = self.residues[self.residues.group_name == group]
            for _, row in grouped_residues.iterrows():
                pdb_path = row['pdb_path']
                chain_id = row['chain_id']
                resid = row['resid']
                label = row['label']
                pdb = read_pdb(pdb_file=pdb_path, category_names=['_atom_site'])['_atom_site']
                pdb = pdb[pdb.record_name.str.strip() == 'ATOM']
                
                # 2. Check residue existence
                target_res = pdb[(pdb.chain_id == chain_id) & (pdb.residue_number == resid)]
                if target_res.empty:
                    print(f"{pdb_path} has no: {chain_id} {resid}")
                    continue
                if 'residue_name' in self.optional_cols:
                    residue_name = row['residue_name']
                    target_res = target_res[target_res.residue_name.strip() == residue_name]
                    if target_res.empty:
                        print(f"{pdb_path} has no: {chain_id} {resid} as {residue_name}")
                        continue
                if 'nmr_model' in pdb.columns:
                    models = pdb.nmr_model.tolist()
                    if 'model_id' in self.optional_cols and row['model_id'] in models:
                        target_res = target_res[target_res.nmr_model == row['model_id']]
                        pdb = pdb[pdb.nmr_model == row['model_id']]
                    else:
                        target_res = target_res[target_res.nmr_model == models[0]]
                        pdb = pdb[pdb.nmr_model == models[0]]
                alt_locs = target_res.alt_loc.unique().tolist()
                if 'alt_id' in self.optional_cols or len(alt_locs) > 1:
                    alt_id = row['alt_id']
                    if alt_id in alt_locs:
                        target_res = target_res[target_res.alt_loc.isin([alt_id, ' '])]
                        pdb = pdb[pdb.alt_loc.isin([alt_id, ' '])]
                    else:
                        target_res = target_res[target_res.alt_loc.isin([alt_locs[0], ' '])]
                        pdb = pdb[pdb.alt_loc.isin([alt_locs[0], ' '])]
                if 'atom_name' in self.optional_cols:
                    atom_name = row['atom_name']
                    atom_names = target_res.atom_name.strip().tolist()
                    if atom_name in atom_names:
                        target_res = target_res[target_res.atom_name.strip() == atom_name]
                    else:
                        msg = f"Warning: {atom_name} not in {pdb_path} {chain_id} {resid}"
                        msg += "center of geometry is used instead."
                        print(msg)
                target_res_coords = target_res[['x_coord', 'y_coord', 'z_coord']].values
                tc = np.mean(target_res_coords, axis=0)

                # 3. Get subunit
                if self.cut_method not in ['cube', 'sphere'] or cut_off is None:
                    subunit = pdb.reset_index(drop=True)
                elif self.cut_method == 'cube':
                    cond = (np.abs(pdb.x_coord - tc[0]) <= cut_off
                        & np.abs(pdb.y_coord - tc[1]) <= cut_off
                        & np.abs(pdb.z_coord - tc[2]) <= cut_off
                    )
                    subunit = pdb[cond].reset_index(drop=True)
                elif self.cut_method == 'sphere':
                    kd_tree = scipy.spatial.KDTree(pdb[['x_coord', 'y_coord', 'z_coord']].values)
                    subunit_idx = kd_tree.query_ball_point(tc, r=cut_off, p=2.0)
                    subunit = pdb.iloc(subunit_idx).reset_index(drop=True)
                subunit['atom'] = subunit.apply(
                    lambda x: self.atom_mapping(x.element_symbol.strip()),
                    axis=1,
                )
            
                # 4. Get a sample data
                if center:
                    # Centering
                    subunit['x_coord'] = subunit['x_coord'] - tc[0]
                    subunit['y_coord'] = subunit['y_coord'] - tc[1]
                    subunit['z_coord'] = subunit['z_coord'] - tc[2]
                if reflect:
                    # Reflection over the yz plane
                    subunit['x_coord'] = -subunit['x_coord']
                pdb_paths.append(pdb_path)
                chain_ids.append(chain_id)
                resids.append(resid)
                subunits.append(subunit)
                labels.append(label)
                if n_rot > 0:
                    # Rotations from Haar measure
                    rotations = sog(3).rvs(size=n_rot, random_state=rng)
                    x = subunit[['x_coord', 'y_coord', 'z_coord']].values()
                    nx = np.einsum("ni,mij->mnj", x, rotations)
                    for i in range(n_rot):
                        pdb_paths.append(pdb_path)
                        chain_ids.append(chain_id)
                        resids.append(resid)
                        subunit['x_coord'] = nx[i][0, :]
                        subunit['y_coord'] = nx[i][1, :]
                        subunit['z_coord'] = nx[i][2, :]
                        subunits.append(subunit.copy())
                        labels.append(label)
                if n_trans > 0:
                    # Translations
                    shifts = rng.normal(scale=trans_std, size=(n_trans, 1, 3))
                    nx = subunit[['x_coord', 'y_coord', 'z_coord']].values() + shifts
                    for i in range(n_trans):
                        pdb_paths.append(pdb_path)
                        chain_ids.append(chain_id)
                        resids.append(resid)
                        subunit['x_coord'] = nx[i][0, :]
                        subunit['y_coord'] = nx[i][1, :]
                        subunit['z_coord'] = nx[i][2, :]
                        subunits.append(subunit.copy())
                        labels.append(label)

            self.prepared_residues.append(group)
            # Check enough is sampled
            n_sampled += 1
            if n_sampled >= num_samples:
                print(f"Has sampled {n_sampled} (grouped) residues. Congrats!")
                break

        return pdb_paths, chain_ids, resids, subunits, labels



