import logging 
from tqdm import tqdm 
import io 
import time
from .base import HookBase

from core.utils import comm

logger = logging.getLogger(__name__)


class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ''
    def __init__(self,logger,level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO
    def write(self,buf):
        self.buf = buf.strip('\r\n\t ')
    def flush(self):
        self.logger.log(self.level, self.buf)


class EvalHook(HookBase):
    def __init__(self, period: int, eval_func):
        super().__init__()

        self.period = period
        self._num_iter = 0
        self._func = eval_func
    
    def _do_eval(self):
        results = self._func()
        
        if results:
            assert isinstance(
                results, dict
            ), "Eval function must return a dict. Got {} instead.".format(results)

            for k, v in results.items():
                try:
                    v = float(v)
                except Exception as e:
                    raise ValueError(
                        "[EvalHook] eval_function should return a nested dict of float. "
                        "Got '{}: {}' instead.".format(k, v)
                    ) from e

            self.storage.put_scalars(**results, smoothing_hint=False)
        
        # Evaluation may take different time among workers.
        # A barrier make them start the next iteration together.
        # comm.synchronize()

    def after_step(self):
        self._num_iter += 1
        
        if self.period > 0 and self._num_iter % self.period == 0:
            logger.info("Eval!!\r")
            logger.info("Eval!!\r")
            self._do_eval()
            




