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
from .dice import faster_dice  # Import Dice score calculation
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
        
        # TODO Make sure the initial random model weights are the same across sites!
        self.model = enMesh_checkpoint(in_channels=1, n_classes=3, channels=5, config_file=config_file_path)

        # Check if GPU available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Ensure model parameters require gradients
        for param in self.model.parameters():
            param.requires_grad = True

        # Optimizer and criterion setup
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = torch.nn.CrossEntropyLoss()

        # AMP for mixed precision to overcome memory limitations
        # self.scaler = torch.amp.GradScaler()  # No need to specify 'cuda', it's inferred automatically

        # Logger can be found for example with: MeshDist_nvflare/simulator_workspace/simulate_job/app_site-1 and app_site-2
        self.logger = GenericLogger(log_file_path='meshnet_executor.log')
        self.current_epoch = 0

        # Epochs and aggregation interval
        self.total_epochs = 2  # Set the total number of epochs
        # self.aggregation_interval = 1  # Aggregation occurs every 5 epochs (you can modify this)

        self.dice_threshold = 0.9  # Set the Dice score threshold

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
        self.site_name = fl_ctx.get_prop(FLContextKey.CLIENT_NAME)


        if task_name == "train_and_get_gradients":
            self.logger.log_message(f"{self.site_name}-train_and_get_gradients called ")
            gradients = self.train_and_get_gradients()
            outgoing_shareable = Shareable()
            outgoing_shareable["gradients"] = gradients
            return outgoing_shareable

        elif task_name == "accept_aggregated_gradients":
            self.logger.log_message(f"{self.site_name}-accept_aggregated_gradients called ")
            aggregated_gradients = shareable["aggregated_gradients"]
            self.apply_gradients(aggregated_gradients, fl_ctx)
            return Shareable()

    # def train_and_get_gradients_old(self):
    #     for epoch in range(self.total_epochs):

    #         self.logger.log_message(f"Starting Epoch {epoch}/{self.total_epochs}, Aggregation Interval: {self.aggregation_interval}")
    #         self.model.train()
            
    #         # Initialize accumulators for the loss and gradients
    #         total_loss = 0.0
    #         gradient_accumulator = [torch.zeros_like(param).to(self.device) for param in self.model.parameters()]
            
    #         # Training loop for one epoch (full pass through the dataset)
    #         for batch_id, (image, label) in enumerate(self.trainloader):
    #             image, label = image.to(self.device), label.to(self.device)
    #             self.optimizer.zero_grad()

    #             # Mixed precision and checkpointing
    #             with torch.amp.autocast(device_type='cuda'):
    #                 output = torch.utils.checkpoint.checkpoint(self.model, image, use_reentrant=False)
    #                 label = label.squeeze(1)
    #                 loss = self.criterion(output, label.long())

    #             total_loss += loss.item()

    #             # Scale loss and backward pass
    #             self.scaler.scale(loss).backward()

    #             # Accumulate gradients
    #             for i, param in enumerate(self.model.parameters()):
    #                 if param.grad is not None:
    #                     gradient_accumulator[i] += param.grad.clone()

    #             self.scaler.step(self.optimizer)
    #             self.scaler.update()

    #             torch.cuda.empty_cache()

    #         # Log the average loss per epoch
    #         average_loss = total_loss / len(self.trainloader)
    #         dice_score = self.calculate_dice(self.trainloader)
    #         self.logger.log_message(f"Site {self.site_name} - Epoch {epoch}: Loss = {average_loss}, Dice = {dice_score}")

    #         # Call aggregation based on your set aggregation_interval
    #         if (epoch + 1) % self.aggregation_interval == 0:
    #             # Perform model aggregation here
    #             return [grad.clone().cpu().numpy() for grad in gradient_accumulator if grad is not None]

    #     return []

    # def train_and_get_gradients_new(self):
    #     for epoch in range(self.total_epochs):
    #         # self.logger.log_message(f"Starting Epoch {epoch+1}/{self.total_epochs}, Aggregation Interval: {self.aggregation_interval}")
    #         self.model.train()

    #         # Initialize accumulators for the loss and gradients
    #         total_loss = 0.0
    #         gradient_accumulator = [torch.zeros_like(param).to(self.device) for param in self.model.parameters()]

    #         # Training loop for one epoch (full pass through the dataset)
    #         for batch_id, (image, label) in enumerate(self.trainloader):
    #             image, label = image.to(self.device), label.to(self.device)
    #             self.optimizer.zero_grad()

    #             # Mixed precision and checkpointing
    #             with torch.amp.autocast(device_type='cuda'):
    #                 output = torch.utils.checkpoint.checkpoint(self.model, image, use_reentrant=False)
    #                 label = label.squeeze(1)
    #                 loss = self.criterion(output, label.long())

    #             total_loss += loss.item()

    #             # Scale loss and backward pass
    #             self.scaler.scale(loss).backward()

    #             # Accumulate gradients
    #             for i, param in enumerate(self.model.parameters()):
    #                 if param.grad is not None:
    #                     gradient_accumulator[i] += param.grad.clone()

    #             self.scaler.step(self.optimizer)
    #             self.scaler.update()

    #             torch.cuda.empty_cache()

    #         # Log the average loss per epoch
    #         average_loss = total_loss / len(self.trainloader)
    #         dice_score = self.calculate_dice(self.trainloader)
    #         self.logger.log_message(f"Site {self.site_name} - Epoch {epoch+1}: Loss = {average_loss}, Dice = {dice_score}")

    #         # Check if it's time to perform aggregation
    #         if (epoch + 1) % self.aggregation_interval == 0:
    #             # Return the gradients after completing the specified aggregation interval
    #             self.logger.log_message(f"Performing aggregation after epoch {epoch+1}")
    #             return [grad.clone().cpu().numpy() for grad in gradient_accumulator if grad is not None]

    #     return []


    # def setup_site_logger(self):
    #     site_id = os.getenv('FL_SITE_ID', 'site_unknown')  # Use environment variable or other means to set site ID
    #     log_dir = f'logs/{site_id}'
    #     os.makedirs(log_dir, exist_ok=True)
    #     log_filename = os.path.join(log_dir, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    #     logging.basicConfig(
    #         filename=log_filename,
    #         level=logging.INFO,
    #         format='%(asctime)s - %(levelname)s - %(message)s',
    #         datefmt='%Y-%m-%d %H:%M:%S',
    #     )

    #     self.logger = logging.getLogger()
    #     self.logger.info("Logging started")


    def train_and_get_gradients(self):
        self.model.train()

        # Initialize accumulators for the loss and gradients
        total_loss = 0.0

        # Training loop for one epoch (full pass through the dataset)
        for batch_id, (image, label) in enumerate(self.trainloader):
            image, label = image.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()

            # Mixed precision and checkpointing
            with torch.amp.autocast(device_type='cuda'):
                output = torch.utils.checkpoint.checkpoint(self.model, image, use_reentrant=False)
                label = label.squeeze(1)
                loss = self.criterion(output, label.long())

            # Accumulate loss
            total_loss += loss.item()

            # Scale loss and backward pass
            loss.backward()
            self.optimizer.step()
            
        # Log the average loss and Dice score per epoch
        average_loss = total_loss / len(self.trainloader)
        dice_score = self.calculate_dice(self.trainloader)
        self.logger.log_message(f"{self.site_name} - Epoch {self.current_epoch}: Loss = {average_loss}, Dice = {dice_score}")

        # Return the gradients after completing the specified aggregation interval
        self.logger.log_message(f"{self.site_name} Preparing payload after an iteration in epoch {self.current_epoch}")
        # Accumulate gradients
        gradients = []
        for i, param in enumerate(self.model.parameters()):
            if param.grad is not None:
                gradient_accumulator.append(param.grad.clone().cpu().numpy())
        
        # # Increment the iteration count
        #-- self.current_epoch += 1

        return gradients

    def calculate_dice(self, loader):
        dice_total = 0.0
        for image, label in loader:
            image, label = image.to(self.device), label.to(self.device)
            with torch.no_grad():
                output = self.model(image)
                output_label = torch.argmax(output, dim=1)
                dice_score = faster_dice(output_label, label.squeeze(1), labels=[0, 1, 2])
                dice_total += dice_score.mean().item()
        return dice_total / len(loader)

    def apply_gradients(self, aggregated_gradients, fl_ctx):
        # Apply aggregated gradients to the model parameters
        self.optimizer.zero_grad()
        for param, grad in zip(self.model.parameters(), aggregated_gradients):
            param.grad = torch.tensor(grad).to(self.device)
        self.optimizer.step()

        # Clear GPU memory cache after applying gradients
        # torch.cuda.empty_cache()

        # Log the gradient application step
        self.logger.log_message(f"{self.site_name} Aggregated gradients applied to the model.")

        # Get the output directory path
        output_dir = get_output_directory_path(fl_ctx)
        
        # Save the model
        model_save_path = os.path.join(output_dir, f"model_epoch_{self.current_epoch}.pth")
        torch.save(self.model.state_dict(), model_save_path)
        
        # Log the model saving step
        self.logger.log_message(f"Model saved at {model_save_path}")
        self.current_epoch += 1
