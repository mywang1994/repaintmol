import torch

class Config:
    # Data related config
    DATA_DIR = "data"
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # Model related config
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HIDDEN_DIM = 256
    NUM_HEADS = 8
    NUM_LAYERS = 6
    DROPOUT = 0.1
    
    # Training related config (global default)
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    WARMUP_STEPS = 1000
    
    # Training config for scaffold generator
    SCAFFOLD_TRAIN = {
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'batch_size': 32,
        'scheduler': 'cosine',
    }
    # Training config for inpainting model
    INPAINTING_TRAIN = {
        'learning_rate': 5e-5,
        'num_epochs': 150,
        'batch_size': 16,
        'scheduler': 'cosine',
    }
    
    # Diffusion model config
    DIFFUSION_STEPS = 1000
    BETA_START = 0.0001
    BETA_END = 0.02
    TEMPLATE_WEIGHT = 0.1
    POCKET_WEIGHT = 0.1
    SIGMA = 0.1
    
    # Scaffold loss weights
    LAMBDA_VQ = 1.0
    LAMBDA_TPL = 0.1
    LAMBDA_ELEC = 0.1  # Electrostatic/vdw energy loss weight
    
    # VQ-VAE config
    NUM_EMBEDDINGS = 512
    EMBEDDING_DIM = 64
    
    # Reaction template config
    REACTION_TEMPLATES = {
        'amidation': {
            'name': 'Amidation Reaction',
            'smarts': '[C:1](=[O:2])-[OH:3].[N:4]>>[C:1](=[O:2])-[N:4]',
            'fragments': {
                'acid': {'atoms': [1, 2, 3], 'type': 'carboxylic_acid'},
                'amine': {'atoms': [4], 'type': 'amine'}
            },
            'bonds': [(1, 4)],
            'angle_constraints': None
        },
        'friedel_crafts': {
            'name': 'Friedel-Crafts Alkylation',
            'smarts': '[c:1].[C:2]-[Cl:3]>>[c:1]-[C:2]',
            'fragments': {
                'aromatic': {'atoms': [1], 'type': 'aromatic'},
                'alkyl': {'atoms': [2, 3], 'type': 'alkyl_halide'}
            },
            'bonds': [(1, 2)],
            'angle_constraints': None
        },
        'click_chemistry': {
            'name': 'Click Chemistry (Azide-Alkyne Cycloaddition)',
            'smarts': '[N:1]=[N:2]=[N:3].[C:4]#[C:5]>>[N:1]1-[N:2]=[N:3]-[C:4]=[C:5]-1',
            'fragments': {
                'azide': {'atoms': [1, 2, 3], 'type': 'azide'},
                'alkyne': {'atoms': [4, 5], 'type': 'alkyne'}
            },
            'bonds': [(1, 4), (3, 5)],
            'angle_constraints': {
                'type': 'ring_closure',
                'atoms': [1, 4, 5, 3],
                'target_angle': 90.0,
                'weight': 0.5
            }
        }
    }
    
    # Protein pocket config
    # POCKET_CUTOFF = 3.5  # Å
    POCKET_RESIDUE_TYPES = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
        "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
        "THR", "TRP", "TYR", "VAL"
    ]
    
    # Ligand residue types
    LIGAND_RESIDUES = [
        "UNL", "UNX", "LIG", "DRG", "INH", "SUB", "LST", "MOL",
        "SM", "SM2", "SM3", "SM4", "SM5", "SM6", "SM7", "SM8",
        "SM9", "SM10", "SM11", "SM12", "SM13", "SM14", "SM15"
    ]
    
    # Ligand atom types
    LIGAND_ATOM_TYPES = [
        "C", "N", "O", "S", "P", "F", "Cl", "Br", "I",
        "B", "Si", "Se", "As"
    ]
    
    # Protein atom types
    PROTEIN_ATOM_TYPES = [
        "N", "CA", "C", "O", "CB", "CG", "CD", "CE", "CZ",
        "ND1", "ND2", "NE", "NE1", "NE2", "NH1", "NH2",
        "NZ", "OD1", "OD2", "OE1", "OE2", "OG", "OG1",
        "OH", "SD", "SG"
    ]
    
    # Scaffold atom types
    SCAFFOLD_ATOM_TYPES = [
        "C", "N", "O", "S", "P"
    ]
    
    # Save path
    SAVE_DIR = "checkpoints"
    LOG_DIR = "logs"
    
    # Atom charge config
    ATOM_CHARGES = {
        'C': 0.0,
        'N': -0.5,
        'O': -0.7,
        'S': -0.3,
        'P': 0.5,
        'F': -0.3,
        'Cl': -0.2,
        'Br': -0.1,
        'I': 0.0,
        'B': 0.3,
        'Si': 0.4,
        'Se': -0.2,
        'As': 0.2
    }
    
    # Feature encoding config
    ATOM_FEATURE_DIM = 32
    BOND_FEATURE_DIM = 16
    RESIDUE_FEATURE_DIM = 64 