from utils import registry


MODELS = registry.Registry('models')


def build_model_from_cfg(args, cfg, **kwargs):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        cfg (eDICT): 
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    return MODELS.build(args, cfg, **kwargs)


