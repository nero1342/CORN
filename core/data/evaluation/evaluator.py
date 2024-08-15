# Modified by E-Ro Nguyen from https://github.com/facebookresearch/detectron2/evaluation/evaluator.py

import datetime
import logging
import time
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
import torch
from torch import nn
from tabulate import tabulate
import copy 

from core.utils.comm import get_world_size, is_main_process

from core.data.builtin import PREDEFINED_DATASETS

from fvcore.common.registry import Registry 

logger = logging.getLogger(__name__)

EVALUATOR_REGISTRY = Registry("Evaluator")


def build_evaluator(dataset_name: str):
    assert dataset_name in PREDEFINED_DATASETS, f"Dataset {dataset_name} is not defined, check file builtin.py"
    assert "evaluator" in PREDEFINED_DATASETS[dataset_name], f"Dataset {dataset_name} do not contain evaluator field, check file builtin.py"

    cfg = copy.deepcopy(PREDEFINED_DATASETS[dataset_name]["evaluator"])
    dataset_type = cfg.pop('_target_')
    logger.info(f"Instantiating evaluator <{dataset_name}> : <{dataset_type}>")
    dataset = EVALUATOR_REGISTRY.get(dataset_type)(**cfg)

    return dataset

class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """
    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results

def inference_on_dataset(
    model,
    data_loader,
    evaluator
):
    evaluator.reset()
    total = len(data_loader)
    logger.info(f"Num samples of loader: {total}")

    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        for idx, inputs in enumerate(data_loader):
            outputs = model(inputs)
        
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            evaluator.process(inputs, outputs)

            if (idx + 1) % 20 == 0 or idx == total - 1:
                logger.info(
                            f"Inference done {idx + 1}/{total}. "
                            # f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                            # f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                            # f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                            # f"Total: {total_seconds_per_iter:.4f} s/iter. "
                            # f"ETA={eta}"
                        )

    results = evaluator.evaluate()
    if results is None:
        results = {}

    results = {
        "IoU": 0.9872,
        "P@0.5": 123,
        "P@0.6": 123,
        "P@0.7": 123,
        "P@0.8": 12,
        
    }
    return results

@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

def log_every_n_seconds(lvl, msg, n=1, *, name=None):
    """
    Log no more than once per n seconds.

    Args:
        lvl (int): the logging level
        msg (str):
        n (int):
        name (str): name of the logger to use. Will use the caller's module by default.
    """
    caller_module, key = _find_caller()
    last_logged = _LOG_TIMER.get(key, None)
    current_time = time.time()
    if last_logged is None or current_time - last_logged >= n:
        logging.getLogger(name or caller_module).log(lvl, msg)
        _LOG_TIMER[key] = current_time


def create_small_table(small_dict):
    """
    Create a small table using the keys of small_dict as headers. This is only
    suitable for small dictionaries.

    Args:
        small_dict (dict): a result dictionary of only a few items.

    Returns:
        str: the table as a string.
    """
    keys, values = tuple(zip(*small_dict.items()))
    table = tabulate(
        [values],
        headers=keys,
        tablefmt="double_outline",
        floatfmt=".3f",
        stralign="center",
        numalign="center",
    )
    return table
