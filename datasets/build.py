from utils import registry


DATASETS = registry.Registry('dataset')


def build_dataset_from_cfg(args, cfg, default_args = None):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        cfg (eDICT): 
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    return DATASETS.build(args, cfg, default_args = default_args)