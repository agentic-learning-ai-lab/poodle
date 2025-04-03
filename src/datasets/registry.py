from collections import defaultdict
import sys

_module_to_datasets = defaultdict(set)  # dict of sets to check membership of dataset in module
_dataset_to_module = {}  # mapping of dataset names to module names
_dataset_entrypoints = {}  # mapping of dataset names to entrypoint fns

def register_dataset(fn):
    # lookup containing module
    mod = sys.modules[fn.__module__]
    module_name_split = fn.__module__.split('.')
    module_name = module_name_split[-1] if len(module_name_split) else ''

    # add dataset to __all__ in module
    dataset_name = fn.__name__

    if hasattr(mod, '__all__'):
        mod.__all__.append(dataset_name)
    else:
        mod.__all__ = [dataset_name]

    # add entries to registry dict/sets
    _dataset_entrypoints[dataset_name] = fn
    _dataset_to_module[dataset_name] = module_name
    _module_to_datasets[module_name].add(dataset_name)
    return fn


def is_dataset(dataset_name):
    """ Check if a dataset name exists
    """
    return dataset_name in _dataset_entrypoints


def is_dataset_in_modules(dataset_name, module_names):
    """Check if a dataset exists within a subset of modules
    Args:
        dataset_name (str) - name of dataset to check
        module_names (tuple, list, set) - names of modules to search in
    """
    assert isinstance(module_names, (tuple, list, set))
    return any(dataset_name in _module_to_datasets[n] for n in module_names)


def dataset_entrypoint(dataset_name):
    """Fetch a dataset entrypoint for specified dataset name
    """
    return _dataset_entrypoints[dataset_name]


def create_dataset(
        dataset_name,
        **kwargs):
    """Create a dataset

    Args:
        dataset_name (str): name of dataset to instantiate

    Keyword Args:
        **: other kwargs are dataset specific
    """
    # Parameters that aren't supported by all datasets or are intended to only override dataset defaults if set
    # should default to None in command line args/cfg. Remove them if they are present and not set so that
    # non-supporting datasets don't break and default args remain in effect.
    kwargs = {k: v for k, v in kwargs.items()}

    if is_dataset(dataset_name):
        create_fn = dataset_entrypoint(dataset_name)
        dataset = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown dataset (%s)' % dataset_name)
    
    return dataset