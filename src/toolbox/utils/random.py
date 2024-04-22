def make_seed(*args):
    """Copied from webdataset."""
    seed = 0
    for arg in args:
        seed = (seed * 31 + hash(arg)) & 0x7FFFFFFF
    return seed
