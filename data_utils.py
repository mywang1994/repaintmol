import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from biopython import PDB
import os
from tqdm import tqdm
from configs import Config
import warnings
# Add for protein fixing
try:
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile
except ImportError:
    PDBFixer = None
    PDBFile = None

class ComplexPointCloud:
    """Protein-ligand complex point cloud class"""
    def __init__(self, pdb_id: str, 
                 protein_coords: torch.Tensor, protein_features: torch.Tensor,
                 ligand_coords: torch.Tensor, ligand_features: torch.Tensor,
                 pocket_residues: List[PDB.Residue],
                 ligand_mol: Optional[Chem.Mol] = None):
        self.pdb_id = pdb_id
        self.protein_coords = protein_coords  # Protein atom coordinates
        self.protein_features = protein_features  # Protein atom features
        self.ligand_coords = ligand_coords  # Ligand atom coordinates
        self.ligand_features = ligand_features  # Ligand atom features
        self.pocket_residues = pocket_residues  # All protein residues
        self.ligand_mol = ligand_mol  # RDKit Mol object for ligand
    
    @classmethod
    def from_pdb(cls, pdb_file: str) -> 'ComplexPointCloud':
        """Create a complex point cloud from a PDB file (protein and ligand together)"""
        parser = PDB.PDBParser()
        structure = parser.get_structure('complex', pdb_file)
        
        # Separate protein and ligand
        protein_atoms = []
        ligand_atoms = []
        all_protein_residues = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_resname() in Config.LIGAND_RESIDUES:
                        ligand_atoms.extend(residue.get_atoms())
                    else:
                        protein_atoms.extend(residue.get_atoms())
                        all_protein_residues.append(residue)
        
        # Ligand heavy atom coordinates and features
        ligand_coords = []
        ligand_features = []
        for atom in ligand_atoms:
            if atom.element != 'H':
                coord = atom.get_coord()
                ligand_coords.append(coord)
                ligand_features.append(cls._get_ligand_atom_features(atom))
        ligand_coords = torch.tensor(ligand_coords, dtype=torch.float32)
        ligand_features = torch.tensor(ligand_features, dtype=torch.float32)
        
        # Use all protein residues as pocket_residues
        pocket_residues = all_protein_residues
        
        # Protein atom coordinates and features
        protein_coords = []
        protein_features = []
        for atom in protein_atoms:
            if atom.element != 'H':
                coord = atom.get_coord()
                protein_coords.append(coord)
                protein_features.append(cls._get_protein_atom_features(atom))
        protein_coords = torch.tensor(protein_coords, dtype=torch.float32)
        protein_features = torch.tensor(protein_features, dtype=torch.float32)
        
        return cls(
            pdb_id=os.path.basename(pdb_file).split('.')[0],
            protein_coords=protein_coords,
            protein_features=protein_features,
            ligand_coords=ligand_coords,
            ligand_features=ligand_features,
            pocket_residues=pocket_residues,
            ligand_mol=None
        )

    @classmethod
    def from_protein_ligand(cls, protein_pdb_file: str, ligand_sdf_file: str) -> 'ComplexPointCloud':
        """Create a complex point cloud from separate protein PDB and ligand SDF files, with validation and fixing."""
        # --- Protein fixing ---
        fixed_pdb_path = protein_pdb_file + '.fixed.pdb'
        if PDBFixer is not None and PDBFile is not None:
            try:
                fixer = PDBFixer(filename=protein_pdb_file)
                fixer.findMissingResidues()
                fixer.findMissingAtoms()
                fixer.addMissingAtoms()
                fixer.addMissingHydrogens()
                PDBFile.writeFile(fixer.topology, fixer.positions, open(fixed_pdb_path, 'w'))
                protein_pdb_to_use = fixed_pdb_path
            except Exception as e:
                warnings.warn(f"PDBFixer failed: {e}, using original file.")
                protein_pdb_to_use = protein_pdb_file
        else:
            warnings.warn("pdbfixer not installed, skipping protein fixing.")
            protein_pdb_to_use = protein_pdb_file
        # Read protein
        parser = PDB.PDBParser()
        structure = parser.get_structure('protein', protein_pdb_to_use)
        protein_atoms = []
        all_protein_residues = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    protein_atoms.extend(residue.get_atoms())
                    all_protein_residues.append(residue)
        # --- Ligand fixing ---
        mol = Chem.MolFromMolFile(ligand_sdf_file, removeHs=False)
        if mol is None:
            raise ValueError(f"Cannot parse ligand: {ligand_sdf_file}")
        mol = Chem.AddHs(mol, addCoords=True)
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol)
        ligand_coords = []
        ligand_features = []
        for atom in mol.GetAtoms():
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            ligand_coords.append([pos.x, pos.y, pos.z])
            ligand_features.append(cls._get_ligand_atom_features_rdkit(atom))
        ligand_coords = torch.tensor(ligand_coords, dtype=torch.float32)
        ligand_features = torch.tensor(ligand_features, dtype=torch.float32)
        # Use all protein residues as pocket_residues
        pocket_residues = all_protein_residues
        # Protein atom coordinates and features
        protein_coords = []
        protein_features = []
        for atom in protein_atoms:
            if atom.element != 'H':
                coord = atom.get_coord()
                protein_coords.append(coord)
                protein_features.append(cls._get_protein_atom_features(atom))
        protein_coords = torch.tensor(protein_coords, dtype=torch.float32)
        protein_features = torch.tensor(protein_features, dtype=torch.float32)
        # Clean up temporary fixed file
        if os.path.exists(fixed_pdb_path):
            try:
                os.remove(fixed_pdb_path)
            except Exception:
                pass
        return cls(
            pdb_id=os.path.basename(protein_pdb_file).replace('_pocket10.pdb', ''),
            protein_coords=protein_coords,
            protein_features=protein_features,
            ligand_coords=ligand_coords,
            ligand_features=ligand_features,
            pocket_residues=pocket_residues,
            ligand_mol=mol
        )

    @staticmethod
    def _get_ligand_atom_features(atom: PDB.Atom) -> np.ndarray:
        """Get ligand atom features from BioPython atom"""
        atom_type = atom.element
        type_feature = np.zeros(len(Config.LIGAND_ATOM_TYPES))
        if atom_type in Config.LIGAND_ATOM_TYPES:
            type_feature[Config.LIGAND_ATOM_TYPES.index(atom_type)] = 1
        properties = np.array([
            atom.get_occupancy(),
            atom.get_bfactor()
        ])
        return np.concatenate([type_feature, properties])

    @staticmethod
    def _get_ligand_atom_features_rdkit(atom: Chem.Atom) -> np.ndarray:
        """Get ligand atom features from RDKit atom"""
        atom_type = atom.GetSymbol()
        type_feature = np.zeros(len(Config.LIGAND_ATOM_TYPES))
        if atom_type in Config.LIGAND_ATOM_TYPES:
            type_feature[Config.LIGAND_ATOM_TYPES.index(atom_type)] = 1
        properties = np.array([
            atom.GetDegree(),
            atom.GetTotalNumHs(),
            atom.GetFormalCharge(),
            atom.IsAromatic()
        ])
        return np.concatenate([type_feature, properties])

    @staticmethod
    def _get_protein_atom_features(atom: PDB.Atom) -> np.ndarray:
        """Get protein atom features"""
        atom_type = atom.get_name()
        type_feature = np.zeros(len(Config.PROTEIN_ATOM_TYPES))
        if atom_type in Config.PROTEIN_ATOM_TYPES:
            type_feature[Config.PROTEIN_ATOM_TYPES.index(atom_type)] = 1
        properties = np.array([
            atom.get_occupancy(),
            atom.get_bfactor()
        ])
        return np.concatenate([type_feature, properties])

class ScaffoldPointCloud:
    """Scaffold point cloud class"""
    def __init__(self, pdb_id: str, scaffold_smiles: str,
                 scaffold_coords: torch.Tensor, scaffold_features: torch.Tensor,
                 pocket_coords: torch.Tensor, pocket_features: torch.Tensor):
        self.pdb_id = pdb_id
        self.scaffold_smiles = scaffold_smiles
        self.scaffold_coords = scaffold_coords
        self.scaffold_features = scaffold_features
        self.pocket_coords = pocket_coords
        self.pocket_features = pocket_features
    
    @classmethod
    def from_complex(cls, complex_cloud: ComplexPointCloud) -> 'ScaffoldPointCloud':
        """Create scaffold point cloud from complex point cloud"""
        ligand_mol = complex_cloud.ligand_mol
        scaffold = MurckoScaffold.GetScaffoldForMol(ligand_mol)
        if scaffold.GetNumConformers() == 0:
            AllChem.EmbedMolecule(scaffold)
        conf = scaffold.GetConformer()
        scaffold_coords = []
        scaffold_features = []
        for atom in scaffold.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            scaffold_coords.append([pos.x, pos.y, pos.z])
            atom_type = atom.GetSymbol()
            type_feature = np.zeros(len(Config.SCAFFOLD_ATOM_TYPES))
            if atom_type in Config.SCAFFOLD_ATOM_TYPES:
                type_feature[Config.SCAFFOLD_ATOM_TYPES.index(atom_type)] = 1
            scaffold_features.append(type_feature)
        scaffold_coords = torch.tensor(scaffold_coords, dtype=torch.float32)
        scaffold_features = torch.tensor(scaffold_features, dtype=torch.float32)
        pocket_coords = complex_cloud.protein_coords
        pocket_features = complex_cloud.protein_features
        scaffold_smiles = Chem.MolToSmiles(scaffold)
        return cls(
            pdb_id=complex_cloud.pdb_id,
            scaffold_smiles=scaffold_smiles,
            scaffold_coords=scaffold_coords,
            scaffold_features=scaffold_features,
            pocket_coords=pocket_coords,
            pocket_features=pocket_features
        )

class ComplexDataset:
    """Protein-ligand complex dataset"""
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.complexes = []
        self.scaffolds = []
    
    def load_data(self):
        """Load data from directory with separated protein and ligand files"""
        pdb_files = [f for f in os.listdir(self.data_dir) if f.endswith('_pocket10.pdb')]
        for pdb_file in tqdm(pdb_files, desc="Loading complexes"):
            base = pdb_file.replace('_pocket10.pdb', '')
            ligand_file = os.path.join(self.data_dir, base + '.sdf')
            protein_file = os.path.join(self.data_dir, pdb_file)
            if not os.path.exists(ligand_file):
                warnings.warn(f"Ligand file {ligand_file} not found, skipping.")
                continue
            try:
                complex_cloud = ComplexPointCloud.from_protein_ligand(protein_file, ligand_file)
                self.complexes.append(complex_cloud)
                # Extract scaffold point cloud
                try:
                    scaffold_cloud = ScaffoldPointCloud.from_complex(complex_cloud)
                    self.scaffolds.append(scaffold_cloud)
                except Exception as e:
                    warnings.warn(f"Scaffold extraction failed for {base}: {e}")
            except Exception as e:
                print(f"Error processing {base}: {str(e)}")
                continue
    
    def __len__(self) -> int:
        return len(self.complexes)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        complex_cloud = self.complexes[idx]
        result = {
            'pdb_id': complex_cloud.pdb_id,
            'complex': {
                'protein_coords': complex_cloud.protein_coords,
                'protein_features': complex_cloud.protein_features,
                'ligand_coords': complex_cloud.ligand_coords,
                'ligand_features': complex_cloud.ligand_features,
                'pocket_residues': complex_cloud.pocket_residues
            }
        }
        if idx < len(self.scaffolds):
            scaffold_cloud = self.scaffolds[idx]
            result['scaffold'] = {
                'scaffold_smiles': scaffold_cloud.scaffold_smiles,
                'scaffold_coords': scaffold_cloud.scaffold_coords,
                'scaffold_features': scaffold_cloud.scaffold_features,
                'pocket_coords': scaffold_cloud.pocket_coords,
                'pocket_features': scaffold_cloud.pocket_features
            }
        return result

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Batch collate function，自动生成 mask"""
    complex_data = {
        'protein_coords': torch.stack([item['complex']['protein_coords'] for item in batch]),
        'protein_features': torch.stack([item['complex']['protein_features'] for item in batch]),
        'ligand_coords': torch.stack([item['complex']['ligand_coords'] for item in batch]),
        'ligand_features': torch.stack([item['complex']['ligand_features'] for item in batch]),
        'pocket_residues': [item['complex']['pocket_residues'] for item in batch]
    }
    scaffold_data = None
    mask = None
    if 'scaffold' in batch[0]:
        scaffold_data = {
            'scaffold_smiles': [item['scaffold']['scaffold_smiles'] for item in batch],
            'scaffold_coords': torch.stack([item['scaffold']['scaffold_coords'] for item in batch]),
            'scaffold_features': torch.stack([item['scaffold']['scaffold_features'] for item in batch]),
            'pocket_coords': torch.stack([item['scaffold']['pocket_coords'] for item in batch]),
            'pocket_features': torch.stack([item['scaffold']['pocket_features'] for item in batch])
        }
        # 生成 mask: 只要特征有不同就mask=1
        ligand_features = complex_data['ligand_features']  # [B, M, D]
        scaffold_features = scaffold_data['scaffold_features']  # [B, M, D]
        mask = (ligand_features != scaffold_features).any(dim=-1).float()  # [B, M]
    result = {
        'pdb_id': [item['pdb_id'] for item in batch],
        'complex': complex_data
    }
    if scaffold_data is not None:
        result['scaffold'] = scaffold_data
    if mask is not None:
        result['mask'] = mask
    return result
