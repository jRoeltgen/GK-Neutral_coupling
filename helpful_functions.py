def print_sources_tree(sources, indent=0):
    """
    Recursively prints keys of a nested dict as a tree.
    """
    spacer = "  " * indent
    if isinstance(sources, dict):
        for key, val in sources.items():
            print(f"{spacer}- {key}")
            print_sources_tree(val, indent + 1)
    else:
        # reached the leaf (probably the array)
        print(f"{spacer}  (leaf: {type(sources).__name__}, shape={getattr(sources, 'shape', 'N/A')})")

def print_sources_tree_flat(sources, path=None):
    """
    Prints all paths in sources with their leaf type and shape in a single line.
    """
    from collections.abc import Mapping

    if path is None:
        path = []

    if isinstance(sources, Mapping):
        for key, val in sources.items():
            print_sources_tree_flat(val, path + [str(key)])
    else:
        # leaf node: probably an array
        leaf_type = type(sources).__name__
        leaf_shape = getattr(sources, "shape", "N/A")
        print(f"{' / '.join(path)} -> {leaf_type}, shape={leaf_shape}")
