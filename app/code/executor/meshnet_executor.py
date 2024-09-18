import torch
import os
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from .meshnet import MeshNet  # Import the MeshNet model
from .loader import Scanloader  # Import the Scanloader for MRI data
from .dist import GenericLogger  # Import GenericLogger
import torch.cuda.amp as amp

class MeshNetExecutor(Executor):
    def __init__(self):
        super().__init__()
        # Initialize the MeshNet model
        # Model Initialization: The MeshNet model is initialized with input channels, number of classes, and the configuration file (modelAE.json).
        # Construct the absolute path to the modelAE.json file
        config_file_path = os.path.join(os.path.dirname(__file__), "modelAE.json")
        self.model = MeshNet(in_channels=1, n_classes=3, channels=32, config_file=config_file_path)        

        # self.model = MeshNet(in_channels=1, n_classes=3, channels=32, config_file="modelAE.json")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = torch.nn.CrossEntropyLoss()

        # Initialize data loader (assuming mindboggle.db is the database file)
        # We load the MRI data using the Scanloader class from loader.py, which reads data from an SQLite database.
        db_file_path = os.path.join(os.path.dirname(__file__), "mindboggle.db")
        self.data_loader = Scanloader(db_file=db_file_path, label_type='GWlabels', num_cubes=1)
        self.trainloader, self.validloader, self.testloader = self.data_loader.get_loaders()

        # Initializes the logger to write logs to a file named meshnet_executor.log.
        self.logger = GenericLogger(log_file_path='meshnet_executor.log')        

        self.current_iteration = 0

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:

        if task_name == "train_and_get_gradients":
            # Perform local training and return gradients
            # This function trains the model on a single batch of data and returns the gradients.
            gradients = self.train_and_get_gradients()
            outgoing_shareable = Shareable()
            outgoing_shareable["gradients"] = gradients
            return outgoing_shareable

        elif task_name == "accept_aggregated_gradients":
            # Accept aggregated gradients and apply them to the model
            aggregated_gradients = shareable["aggregated_gradients"]

            # Applies the aggregated gradients from the central node to update the model.
            self.apply_gradients(aggregated_gradients)
            return Shareable()

    def train_and_get_gradients(self):
        # Perform one iteration of training and return the gradients
        self.model.train()
        image, label = self.get_next_train_batch()
        image, label = image.to(self.device), label.to(self.device)

        self.optimizer.zero_grad()

        # Initialize gradient scaler for mixed precision training
        scaler = amp.GradScaler()

        # Mixed precision training
        with amp.autocast():
            output = self.model(image)
            loss = self.criterion(output, label)

        # Backward pass with scaled gradients
        scaler.scale(loss).backward()

        # Step with scaled optimizer
        scaler.step(self.optimizer)
        scaler.update()

        # Log loss and training information
        self.logger.log_message(f"Iteration {self.current_iteration}: Loss = {loss.item()}")

        # Extract gradients
        gradients = [param.grad.clone().cpu().numpy() for param in self.model.parameters()]
        self.current_iteration += 1
        return gradients

    def get_next_train_batch(self):
        # Get the next batch of data from the trainloader
        for batch_id, (image, label) in enumerate(self.trainloader):
            if batch_id == self.current_iteration % len(self.trainloader):
                return image, label

    def apply_gradients(self, aggregated_gradients):
        # Apply aggregated gradients to the model parameters
        with torch.no_grad():
            for param, grad in zip(self.model.parameters(), aggregated_gradients):
                param.grad = torch.tensor(grad).to(self.device)
            self.optimizer.step()


        # Log the gradient application step
        self.logger.log_message("Aggregated gradients applied to the model.")
