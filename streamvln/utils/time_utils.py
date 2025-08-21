import time
from contextlib import contextmanager

@contextmanager
def timing_context(name, instance, results_attr_name='timing_results'):
    """
    Context manager for measuring execution time with optional storage.
    
    Args:
        name (str): Name of the block being measured.
        instance (object): Object to store timing results.
        results_attr_name (str): Name of the attribute to store timing results on the instance.
    """
    # initialize the dict
    if not hasattr(instance, results_attr_name):
        setattr(instance, results_attr_name, {})
    
    # get dict
    results_dict = getattr(instance, results_attr_name)
    
    # initialize the list with the given name
    if name not in results_dict:
        results_dict[name] = []

    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # store the elapsed time in the dict
    results_dict[name].append(elapsed_time)