from collections import defaultdict
import sys

_module_to_routines = defaultdict(set)  # dict of sets to check membership of routine in module
_routine_to_module = {}  # mapping of routine names to module names
_routine_entrypoints = {}  # mapping of routine names to entrypoint fns

def register_routine(fn):
    # lookup containing module
    mod = sys.modules[fn.__module__]
    module_name_split = fn.__module__.split('.')
    module_name = module_name_split[-1] if len(module_name_split) else ''

    # add routine to __all__ in module
    routine_name = fn.__name__

    if hasattr(mod, '__all__'):
        mod.__all__.append(routine_name)
    else:
        mod.__all__ = [routine_name]

    # add entries to registry dict/sets
    _routine_entrypoints[routine_name] = fn
    _routine_to_module[routine_name] = module_name
    _module_to_routines[module_name].add(routine_name)
    return fn


def is_routine(routine_name):
    """ Check if a routine name exists
    """
    return routine_name in _routine_entrypoints


def is_routine_in_modules(routine_name, module_names):
    """Check if a routine exists within a subset of modules
    Args:
        routine_name (str) - name of routine to check
        module_names (tuple, list, set) - names of modules to search in
    """
    assert isinstance(module_names, (tuple, list, set))
    return any(routine_name in _module_to_routines[n] for n in module_names)


def routine_entrypoint(routine_name):
    """Fetch a routine entrypoint for specified routine name
    """
    return _routine_entrypoints[routine_name]


def get_routine(
        routine_name,
        **kwargs):
    """Create a routine

    Args:
        routine_name (str): name of routine to instantiate

    Keyword Args:
        **: other kwargs are routine specific
    """
    # Parameters that aren't supported by all routines or are intended to only override routine defaults if set
    # should default to None in command line args/cfg. Remove them if they are present and not set so that
    # non-supporting routines don't break and default args remain in effect.
    kwargs = {k: v for k, v in kwargs.items()}

    if is_routine(routine_name):
        create_fn = routine_entrypoint(routine_name)
        return create_fn
        routine = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown routine (%s)' % routine_name)
    
    return routine