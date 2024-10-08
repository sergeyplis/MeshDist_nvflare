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
from .dice import faster_dice  # Import GenericLogger
import torch.cuda.amp as amp
from torch.utils.checkpoint import checkpoint  # For layer checkpointing
from .paths import get_data_directory_path, get_output_directory_path
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)


class MeshNetExecutor(Executor):
    def __init__(self):
        super().__init__()
        # Model Initialization
        config_file_path = os.path.join(os.path.dirname(__file__), "modelAE.json")

        self.model = enMesh_checkpoint(in_channels=1, n_classes=3, channels=1, config_file=config_file_path)

        # Check if GPU available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Ensure model parameters require gradients
        for param in self.model.parameters():
            param.requires_grad = True

        # Optimizer and criterion setup
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.001)
        # You could also try Adam if needed 
        self.criterion = torch.nn.CrossEntropyLoss()

        # AMP for mixed precision to overcome memory limitations
        self.scaler = torch.amp.GradScaler()  # No need to specify 'cuda', it's inferred automatically

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
        # Get the correct data directory path
        db_file_path = os.path.join(get_data_directory_path(fl_ctx), "mindboggle.db")
        
        # Initialize Data Loader with dynamic path
        self.data_loader = Scanloader(db_file=db_file_path, label_type='GWlabels', num_cubes=1)
        self.trainloader, self.validloader, self.testloader = self.data_loader.get_loaders(batch_size=1)

        if task_name == "train_and_get_gradients":
            gradients = self.train_and_get_gradients()
            outgoing_shareable = Shareable()
            outgoing_shareable["gradients"] = gradients
            return outgoing_shareable

        elif task_name == "accept_aggregated_gradients":
            aggregated_gradients = shareable["aggregated_gradients"]
            self.apply_gradients(aggregated_gradients, fl_ctx)
            return Shareable()

    def train_and_get_gradients(self):
        self.model.train()
        
        # Initialize accumulators for the loss and gradients
        total_loss = 0.0
        total_dice_score = 0.0
        gradient_accumulator = [torch.zeros_like(param).to(self.device) for param in self.model.parameters()]
        
        # Training loop for one epoch (full pass through the dataset)
        for batch_id, (image, label) in enumerate(self.trainloader):
            image, label = image.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()

            # Mixed precision and checkpointing
            with torch.amp.autocast(device_type='cuda'):
                # Using checkpointing for memory efficiency
                output = torch.utils.checkpoint.checkpoint(self.model, image, use_reentrant=False)
                label = label.squeeze(1)
                loss = self.criterion(output, label.long())
            
            # Accumulate loss
            total_loss += loss.item()

            # Calculate Dice score for the batch
            pred = torch.argmax(output, dim=1)
            dice_score = faster_dice(pred, label, labels=[0, 1, 2])  # 3-class segmentation
            total_dice_score += dice_score.mean().item()            

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
        
        # Log the average loss and Dice score per epoch
        average_loss = total_loss / len(self.trainloader)
        average_dice = total_dice_score / len(self.trainloader)
        self.logger.log_message(f"Epoch {self.current_iteration}: Loss = {average_loss}, Dice = {average_dice}")
        
        
        # Extract accumulated gradients
        gradients = [grad.clone().cpu().numpy() for grad in gradient_accumulator if grad is not None]
        
        # Increment the iteration count
        self.current_iteration += 1

        return gradients

    def apply_gradients(self, aggregated_gradients, fl_ctx):
        # Apply aggregated gradients to the model parameters
        with torch.no_grad():
            for param, grad in zip(self.model.parameters(), aggregated_gradients):
                param.grad = torch.tensor(grad).to(self.device)
            self.optimizer.step()

        # Clear GPU memory cache after applying gradients
        torch.cuda.empty_cache()

        # Log the gradient application step
        self.logger.log_message("Aggregated gradients applied to the model.")

        # Get the output directory path
        output_dir = get_output_directory_path(fl_ctx)
        
        # Save the model
        model_save_path = os.path.join(output_dir, f"model_epoch_{self.current_iteration}.pth")
        torch.save(self.model.state_dict(), model_save_path)
        
        # Log the model saving step
        self.logger.log_message(f"Model saved at {model_save_path}")        
