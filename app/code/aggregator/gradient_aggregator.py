import torch
import numpy as np
from nvflare.apis.shareable import Shareable
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.aggregator import Aggregator

class GradientAggregator(Aggregator):
    def __init__(self):
        self.gradients_list = []

    def accept(self, shareable: Shareable, fl_ctx: FLContext):
        if "gradients" not in shareable:
            return False  # Reject invalid results        
        # Accept collects the gradients from client nodes and stores them in a list.
        gradients = shareable["gradients"]
        
        if not gradients:
            return False  # Reject empty gradients
        
        self.gradients_list.append(gradients)
        return True

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        # Perform gradient aggregation by averaging the gradients across clients and sends the aggregated gradients back to the clients.
        num_clients = len(self.gradients_list)
        aggregated_gradients = self.average_gradients(self.gradients_list)

        # Clear the gradients list for the next round
        self.gradients_list = []

        # Return the aggregated gradients
        result = Shareable()
        result["aggregated_gradients"] = aggregated_gradients
        return result

    def average_gradients(self, gradients_list):
        # Convert gradients to numpy arrays and perform averaging
        n = len(gradients_list)
        sum_arrays = [np.zeros_like(arr) for arr in gradients_list[0]]
        
        for gradients in gradients_list:
            for i, grad in enumerate(gradients):
                sum_arrays[i] += grad
        
        average_arrays = [grad / n for grad in sum_arrays]
        return average_arrays
