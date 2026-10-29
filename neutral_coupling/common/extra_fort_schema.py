from types import MappingProxyType
MOMENT_MAP_WRITABLE = {
    '1': 'particle',
    '2': 'momentum',
    '3': 'energy',
    '4': 'N/A'
}

COLLISION_MAP_WRITABLE = {
    '0': 'atom-plasma',
    '1': 'molecule-plasma',
    '2': 'testion-plasma',
    '3': 'photon-plasma',
    '4': 'plasma-plasma',
    '5': 'N/A'
}

#Particle class is lower cased
#Derived/assumed species are all capitalized (e.g. ATOMS)
PARTICLE_CLASS_MAP_WRITABLE = {
    '0': 'electrons',
    '1': 'atoms',
    '2': 'molecules',
    '3': 'test_ions',
    '4': 'photons',
    '5': 'bulk_ions',
    '6': 'N/A'
}

# map from particle class to pseudo species
PSEUDO_SPECIES_WRITABLE = {
    "atoms": "ATOMS",
    "molecules": "MOLECULES",
    "test_ions": "TEST IONS"
}

PSEUDO_SPECIES = MappingProxyType(PSEUDO_SPECIES_WRITABLE)
PARTICLE_CLASS_MAP = MappingProxyType(PARTICLE_CLASS_MAP_WRITABLE)
COLLISION_MAP = MappingProxyType(COLLISION_MAP_WRITABLE)
MOMENT_MAP = MappingProxyType(MOMENT_MAP_WRITABLE)
