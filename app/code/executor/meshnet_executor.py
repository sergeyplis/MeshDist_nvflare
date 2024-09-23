import torch
import os
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from .meshnet import MeshNet  # Import the MeshNet model
from .meshnet import enMesh_checkpoint
from .loader import Scanloader  # Import the Scanloader for MRI data
from .dist import GenericLogger  # Import GenericLogger
import torch.cuda.amp as amp
from torch.utils.checkpoint import checkpoint  # For layer checkpointing

class MeshNetExecutor(Executor):
    def __init__(self):
        super().__init__()
        # Model Initialization
        config_file_path = os.path.join(os.path.dirname(__file__), "modelAE.json")
        self.model = enMesh_checkpoint(in_channels=1, n_classes=3, channels=1, config_file=config_file_path)

        # Check if GPU availabel
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Ensure model parameters require gradients
        for param in self.model.parameters():
            param.requires_grad = True

        # Optimizer and criterion setup
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.001)
        # I guess adam also can be used 
        self.criterion = torch.nn.CrossEntropyLoss()

        # amp for mixed precision to overcome memory limitation for now 
        self.scaler = amp.GradScaler()

        # Data Loader with min batch size to save memory
        db_file_path = os.path.join(os.path.dirname(__file__), "mindboggle.db")
        self.data_loader = Scanloader(db_file=db_file_path, label_type='GWlabels', num_cubes=1)
        self.trainloader, self.validloader, self.testloader = self.data_loader.get_loaders(batch_size=1)  

        # Logger can be found for example with: MeshDist_nvflare/simulator_workspace/simulate_job/app_site-1 and app_site-2
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
            gradients = self.train_and_get_gradients()
            outgoing_shareable = Shareable()
            outgoing_shareable["gradients"] = gradients
            return outgoing_shareable

        elif task_name == "accept_aggregated_gradients":
            aggregated_gradients = shareable["aggregated_gradients"]
            self.apply_gradients(aggregated_gradients)
            return Shareable()

    def train_and_get_gradients_old(self):
        self.model.train()
        image, label = self.get_next_train_batch()
        image, label = image.to(self.device), label.to(self.device)

        # Ensure input requires gradients
        image.requires_grad = True

        self.optimizer.zero_grad()

        # Mixed precision and checkpointing
        with amp.autocast():
            # Ensure checkpoint works with requires_grad
            output = self.model(image)  # To avoid using checkpointing for now

            # Fix shape dim and ensure label is in long type
            label = label.squeeze(1)
            loss = self.criterion(output, label.long())

        # Scale the loss before backward pass with amp
        self.scaler.scale(loss).backward()

        # Update the optimizer with scaled gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Log loss
        self.logger.log_message(f"Iteration {self.current_iteration}: Loss = {loss.item()}")

        # Extract gradients
        gradients = [param.grad.clone().cpu().numpy() for param in self.model.parameters() if param.grad is not None]
        self.current_iteration += 1

        # Clear GPU memory cache to free memory
        torch.cuda.empty_cache()

        return gradients

    def train_and_get_gradients(self):
        self.model.train()
        
        # Initialize accumulators for the loss and gradients
        total_loss = 0.0
        gradient_accumulator = [torch.zeros_like(param).to(self.device) for param in self.model.parameters()]
        
        # Training loop for one epoch (full pass through the dataset)
        for batch_id, (image, label) in enumerate(self.trainloader):
            image, label = image.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()

            # Mixed precision and checkpointing
            with amp.autocast():
                output = self.model(image)
                label = label.squeeze(1)
                loss = self.criterion(output, label.long())
            
            # Accumulate loss
            total_loss += loss.item()

            # Scale the loss and backward pass
            self.scaler.scale(loss).backward()

            # Accumulate gradients
            for i, param in enumerate(self.model.parameters()):
                if param.grad is not None:
                    gradient_accumulator[i] += param.grad.clone()

            # Update optimizer
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Clear GPU cache
            torch.cuda.empty_cache()
        
        # Log the average loss per epoch
        average_loss = total_loss / len(self.trainloader)
        self.logger.log_message(f"Epoch {self.current_iteration}: Loss = {average_loss}")
        
        # Extract accumulated gradients
        gradients = [grad.clone().cpu().numpy() for grad in gradient_accumulator if grad is not None]
        
        # Increment the iteration count
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

        # Clear GPU memory cache after applying gradients
        torch.cuda.empty_cache()

        # Log the gradient application step
        self.logger.log_message("Aggregated gradients applied to the model.")
