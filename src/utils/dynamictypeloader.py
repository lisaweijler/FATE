from typing import Type
from functools import reduce, partial, update_wrapper

from src.utils.datastructures import GenericConfig


def load_type_dynamically_from_fqn(type_fqn: str) -> Type:
    if type_fqn is None: # usefule for optional stuff like lr_scheduler
        return None
    path = type_fqn.split(".")
    module_path = ".".join(path[0:-1])
    type_name = path[-1]

    return load_type_dynamically(module_path, type_name)


def load_type_dynamically(module_path: str, type_name: str) -> Type:
    try:
        module = __import__(module_path, fromlist=[type_name])

        current_type = getattr(module, type_name)
    except Exception as ex:
        print(ex)
        raise TypeError(
            f"exception while dynamically loading type: '{module_path}.{type_name}'") from ex

    return current_type




def init_obj(object_config: GenericConfig, *args, **kwargs):
    """
    Finds a function handle with the name given as 't' in config, and returns the
    instance initialized with corresponding arguments given.

    `object = config.init_obj('name', module, a, b=1)`
    is equivalent to
    `object = module.name(a, b=1)`
    returns None if object_config is none
    """
    if object_config is None:
        return None
    module_name = object_config.module # e.g. torch-geometric....
    module_args = object_config.args
    module_kwargs = object_config.kwargs
    
    module = load_type_dynamically_from_fqn(module_name)

    module_args += args
    module_kwargs.update(kwargs)

    return module(*module_args, **module_kwargs)

def init_ftn(function_config: GenericConfig):
        """
        Finds a function handle with the name given as 'module' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        
        Use update_wrapper, to pass on the __name__ and __doc__ string of original function to
        partial function.
        Alternative would be to define named tuple and save name and fct there. 

        Return None if function_config is none
        """
        if function_config is None:
            return None
        function_name = function_config.module # e.g. torch-geometric....
        function_args = function_config.args
        function_kwargs = function_config.kwargs

        ftn = load_type_dynamically_from_fqn(function_name)
        partial_ftn = partial(ftn, *function_args, **function_kwargs)

        return update_wrapper(partial_ftn, ftn)
