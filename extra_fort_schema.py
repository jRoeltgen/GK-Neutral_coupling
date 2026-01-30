MOMENT_MAP = {
    '1': 'particle',
    '2': 'momentum',
    '3': 'energy',
    '4': 'N/A'
}

COLLISION_MAP = {
    '0': 'atom-plasma',
    '1': 'molecule-plasma',
    '2': 'testion-plasma',
    '3': 'photon-plasma',
    '4': 'plasma-plasma',
    '5': 'N/A'
}

PARTICLE_CLASS_MAP = {
    '0': 'ELECTRONS',
    '1': 'atoms',
    '2': 'molecules',
    '3': 'test_ions',
    '4': 'photons',
    '5': 'bulk_ions',
    '6': 'N/A'
}

# map from particle class to pseudo species
PSEUDO_SPECIES = {
    "atoms": "ATOMS",
    "molecules": "MOLECULES",
    "test_ions": "TEST IONS"
}