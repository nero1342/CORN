import logging 

from .evaluator import DatasetEvaluator, EVALUATOR_REGISTRY

logger = logging.getLogger(__name__)

@EVALUATOR_REGISTRY.register()
class ClassificationEvaluator(DatasetEvaluator):
    def __init__(self):
        super().__init__()

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):

        # for inp, out in zip(inputs, outputs):
            
        pass 
    
    def evaluate(self):
        pass 
    