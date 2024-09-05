import torch
from nvflare.apis.shareable import Shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.aggregator import Aggregator

class GradientAggregator(Aggregator):
    def accept(self, shareable: Shareable, fl_ctx: FLContext):
        # Accept gradients from a site and store them for later aggregation
        # Stores gradients from each site (local node) into a list.
        gradients = shareable["gradients"]
        if not hasattr(self, "gradients_list"):
            self.gradients_list = []
        self.gradients_list.append(gradients)
    
    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        # Aggregate gradients from all sites (average them) and returns the aggregated result.
        num_clients = len(self.gradients_list)
        aggregated_gradients = [
            torch.mean(torch.stack(grads), dim=0)
            for grads in zip(*self.gradients_list)
        ]

        # Return the aggregated gradients
        result = Shareable()
        result["aggregated_gradients"] = aggregated_gradients

        # Clear the gradients list for the next round
        self.gradients_list = []
        
        return result
