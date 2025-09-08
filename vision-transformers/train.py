import torch
from torch import nn, optim
import wandb
import logging
import sys
import os

from utils import save_experiment, save_checkpoint
from data import prepare_data
from vit import ViTForClassification

# Configure logging
def setup_logger(log_level='INFO'):
    log_level_dict = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    level = log_level_dict.get(log_level.upper(), logging.INFO)
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging to file and console
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/vision_transformer.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger('vision_transformer')

logger = logging.getLogger('vision_transformer')

config = {
    "patch_size": 4,  # Input image size: 32x32 -> 8x8 patches
    "hidden_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 4 * 48, # 4 * hidden_size
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": 10, # num_classes of CIFAR10
    "num_channels": 3,
    "qkv_bias": True,
    "use_faster_attention": True,
}

class Trainer:
    def __init__(self, model, optimizer, loss_fn, exp_name, device, use_wandb=False, log_model=False):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device
        self.use_wandb = use_wandb
        self.log_model = log_model
        logger.info(f"Trainer initialized with device: {device}, wandb: {use_wandb}, log_model: {log_model}")
        logger.info(f"Model structure: {model.__class__.__name__}")
        logger.info(f"Optimizer: {optimizer.__class__.__name__}, Learning rate: {optimizer.param_groups[0]['lr']}")

    def train(self, trainloader, testloader, epochs, save_model_every_n_epochs=0):
        # keep track of losses and accuracies
        train_losses, test_losses, accuracies = [], [], []
        
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Train dataset size: {len(trainloader.dataset)}, Test dataset size: {len(testloader.dataset)}")
        logger.info(f"Train batches: {len(trainloader)}, Test batches: {len(testloader)}")

        # train the model
        for i in range(epochs):
            logger.info(f"Starting epoch {i+1}/{epochs}")
            train_loss = self.train_epoch(trainloader)
            logger.info(f"Evaluating model after epoch {i+1}")
            accuracy, test_loss = self.evaluate(testloader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)

            # Log metrics
            metrics = {
                "epoch": i + 1,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "accuracy": accuracy
            }
            
            logger.info(f"Epoch: {i+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
            
            if self.use_wandb:
                logger.debug("Logging metrics to wandb")
                wandb.log(metrics)
            
            if save_model_every_n_epochs > 0 and (i+1) % save_model_every_n_epochs == 0 and i+1 != epochs:
                logger.info(f"Saving checkpoint at epoch {i+1}")
                save_checkpoint(self.exp_name, self.model, i+1)
                
                # Log model checkpoint to wandb if enabled
                if self.use_wandb and self.log_model:
                    checkpoint_path = f"experiments/{self.exp_name}/model_{i+1}.pt"
                    logger.debug(f"Logging model checkpoint to wandb: {checkpoint_path}")
                    artifact = wandb.Artifact(f"model-{self.exp_name}-{i+1}", type="model")
                    artifact.add_file(checkpoint_path)
                    wandb.log_artifact(artifact)

        # save the experiment
        logger.info("Saving final experiment results")
        save_experiment(self.exp_name, config, self.model, train_losses, test_losses, accuracies)
        
        # Log final model to wandb if enabled
        if self.use_wandb and self.log_model:
            final_checkpoint_path = f"experiments/{self.exp_name}/model_final.pt"
            logger.debug(f"Logging final model to wandb: {final_checkpoint_path}")
            artifact = wandb.Artifact(f"model-{self.exp_name}-final", type="model")
            artifact.add_file(final_checkpoint_path)
            wandb.log_artifact(artifact)
        
        logger.info("Training completed")

    def train_epoch(self, trainloader):
        """ Train the model """
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        logger.info(f"Training epoch with {len(trainloader)} batches")
        
        for batch_idx, batch in enumerate(trainloader):
            # move the batch to the device
            batch = [t.to(self.device) for t in batch]
            images, labels = batch
            
            logger.debug(f"Processing batch {batch_idx+1}/{len(trainloader)}, "
                        f"images shape: {images.shape}, labels shape: {labels.shape}")

            # zero the gradients
            self.optimizer.zero_grad()

            # calculate the loss
            logger.debug("Forward pass")
            logits, _ = self.model(images)
            loss = self.loss_fn(logits, labels)

            # backpropagate loss
            logger.debug("Backward pass")
            loss.backward()

            # update model params
            self.optimizer.step()

            total_loss += loss.item() * len(images)
            batch_count += 1
            
            # Log batch-level metrics
            if self.use_wandb and batch_count % 10 == 0:  # Log every 10 batches
                # Calculate batch accuracy
                predictions = torch.argmax(logits, dim=1)
                batch_accuracy = torch.sum(predictions == labels).item() / len(labels)
                
                logger.debug(f"Batch {batch_idx+1}: Loss: {loss.item():.4f}, Accuracy: {batch_accuracy:.4f}")
                
                wandb.log({
                    "batch/loss": loss.item(),
                    "batch/accuracy": batch_accuracy,
                    "batch/learning_rate": self.optimizer.param_groups[0]["lr"],
                })
            
            # Print progress for larger datasets
            if batch_idx % 50 == 0 and batch_idx > 0:
                logger.info(f"Batch {batch_idx}/{len(trainloader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(trainloader.dataset)
        logger.info(f"Epoch completed, average loss: {avg_loss:.4f}")
        return avg_loss

    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        
        logger.info(f"Evaluating model with {len(testloader)} test batches")

        with torch.no_grad():
            for batch_idx, batch in enumerate(testloader):
                # move the batch to device
                batch = [t.to(self.device) for t in batch]
                images, labels = batch
                
                logger.debug(f"Evaluating batch {batch_idx+1}/{len(testloader)}")

                # get predictions
                logits, _ = self.model(images)

                # calculate the loss
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * len(images)

                # calculate the accuracy
                predictions = torch.argmax(logits, dim=1)
                batch_correct = torch.sum(predictions == labels).item()
                correct += batch_correct
                
                logger.debug(f"Batch {batch_idx+1}: Loss: {loss.item():.4f}, "
                           f"Correct: {batch_correct}/{len(labels)}")

        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        
        logger.info(f"Evaluation completed: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, "
                  f"Correct: {correct}/{len(testloader.dataset)}")
        
        return accuracy, avg_loss

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--device", type=str)
    parser.add_argument("--save-model-every", type=int, default=0)
    # wandb arguments
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="vision-transformers", help="wandb project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="wandb entity name")
    parser.add_argument("--wandb-log-model", action="store_true", help="Log model checkpoints to wandb")
    parser.add_argument("--wandb-watch", action="store_true", help="Watch model parameters and gradients")
    # logging arguments
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level")

    args = parser.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args

def main():
    args = parse_args()
    
    # Setup logging
    global logger
    logger = setup_logger(args.log_level)
    
    logger.info(f"Starting Vision Transformer training with experiment name: {args.exp_name}")
    logger.info(f"Command line arguments: {args}")
    
    # Check for CUDA availability
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    # training params
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    device = args.device
    save_model_every_n_epochs = args.save_model_every
    
    logger.info(f"Training parameters: batch_size={batch_size}, epochs={epochs}, lr={lr}, device={device}")

    # Initialize wandb if enabled
    if args.wandb:
        logger.info(f"Initializing wandb with project: {args.wandb_project}, entity: {args.wandb_entity}")
        wandb_config = {
            **config,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": lr,
            "optimizer": "AdamW",
            "weight_decay": 1e-2,
            "device": device,
        }
        try:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.exp_name,
                config=wandb_config
            )
            logger.info("wandb initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing wandb: {e}")
            logger.warning("Continuing without wandb")
            args.wandb = False

    # load the CIFAR10 Dataset
    logger.info("Loading CIFAR10 dataset")
    try:
        trainloader, testloader, classes = prepare_data(batch_size=batch_size)
        logger.info(f"Dataset loaded successfully. Classes: {classes}")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        logger.error("Exiting due to dataset loading failure")
        if args.wandb:
            wandb.finish()
        return

    # create model, optimizer, loss function and trainer
    logger.info("Creating model, optimizer, and loss function")
    model = ViTForClassification(config)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    
    logger.info(f"Model created: {model.__class__.__name__}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Watch model parameters and gradients if enabled
    if args.wandb and args.wandb_watch:
        logger.info("Watching model parameters and gradients with wandb")
        wandb.watch(model, log="all")
    
    trainer = Trainer(model, optimizer, loss_fn, args.exp_name, device=device, use_wandb=args.wandb, log_model=args.wandb_log_model)
    
    logger.info("Starting training")
    trainer.train(trainloader, testloader, epochs, save_model_every_n_epochs=save_model_every_n_epochs)
    logger.info("Training completed")
    
    # Close wandb run
    if args.wandb:
        logger.info("Finishing wandb run")
        wandb.finish()

if __name__ == "__main__":
    main()