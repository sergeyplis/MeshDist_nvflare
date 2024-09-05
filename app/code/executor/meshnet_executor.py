import torch
from nvflare.apis.executor import Executor
from nvflare.apis.shareable import Shareable
from nvflare.apis.fl_constant import FLContextKey
from .meshnet import MeshNet  

class MeshNetExecutor(Executor):
    def __init__(self):
        super().__init__()
        self.model = MeshNet()  # Initiat the MeshNet model

    def execute(self, task_name: str, shareable: Shareable, fl_ctx):
        if task_name == "train_and_get_gradients":
            # Local training -- need review
            data = self.load_data()  # Function to load local node data
            gradients = self.train_and_get_gradients(data)  # Training and gradient calculation
            
            # Return the calculated gradients
            outgoing_shareable = Shareable()
            outgoing_shareable["gradients"] = gradients
            return outgoing_shareable

        elif task_name == "accept_aggregated_gradients":
            # Accept and apply the aggregated gradients from the server
            aggregated_gradients = shareable["aggregated_gradients"]
            self.apply_gradients(aggregated_gradients)
            return Shareable()

    def train_and_get_gradients(self, data):
        # Apply training and gradient calculation,  basic code to try and need adjustment. 
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters())
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = self.model(data["inputs"])
            loss = loss_fn(outputs, data["labels"])
            loss.backward()
            optimizer.step()

        # Extract gradients basic code to try
        gradients = [param.grad.data for param in self.model.parameters()]
        return gradients

    def apply_gradients(self, aggregated_gradients):
        # Apply the aggregated gradients to the model parameters
        with torch.no_grad():
            for param, grad in zip(self.model.parameters(), aggregated_gradients):
                param.grad = grad
                param -= param.grad  # Gradient descent 

    def load_data(self):
        # To be coded .....
        pass
