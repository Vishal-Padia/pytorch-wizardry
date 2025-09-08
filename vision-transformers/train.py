import torch
from torch import nn, optim
import wandb

from utils import save_experiment, save_checkpoint
from data import prepare_data
from vit import ViTForClassification

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

    def train(self, trainloader, testloader, epochs, save_model_every_n_epochs=0):
        # keep track of losses and accuracies
        train_losses, test_losses, accuracies = [], [], []

        # train the model
        for i in range(epochs):
            train_loss = self.train_epoch(trainloader)
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
            
            print(f"Epoch: {i+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
            
            if self.use_wandb:
                wandb.log(metrics)
            
            if save_model_every_n_epochs > 0 and (i+1) % save_model_every_n_epochs == 0 and i+1 != epochs:
                print("\t Save checkpoint at epoch", i+1)
                save_checkpoint(self.exp_name, self.model, i+1)
                
                # Log model checkpoint to wandb if enabled
                if self.use_wandb and self.log_model:
                    checkpoint_path = f"experiments/{self.exp_name}/model_{i+1}.pt"
                    artifact = wandb.Artifact(f"model-{self.exp_name}-{i+1}", type="model")
                    artifact.add_file(checkpoint_path)
                    wandb.log_artifact(artifact)

        # save the experiment
        save_experiment(self.exp_name, config, self.model, train_losses, test_losses, accuracies)
        
        # Log final model to wandb if enabled
        if self.use_wandb and self.log_model:
            final_checkpoint_path = f"experiments/{self.exp_name}/model_final.pt"
            artifact = wandb.Artifact(f"model-{self.exp_name}-final", type="model")
            artifact.add_file(final_checkpoint_path)
            wandb.log_artifact(artifact)

    def train_epoch(self, trainloader):
        """ Train the model """
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in trainloader:
            # move the batch to the device
            batch = [t.to(self.device) for t in batch]
            images, labels = batch

            # zero the gradients
            self.optimizer.zero_grad()

            # calculate the loss
            logits, _ = self.model(images)
            loss = self.loss_fn(logits, labels)

            # backpropagate loss
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
                
                wandb.log({
                    "batch/loss": loss.item(),
                    "batch/accuracy": batch_accuracy,
                    "batch/learning_rate": self.optimizer.param_groups[0]["lr"],
                })

        return total_loss / len(trainloader.dataset)

    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        total_loss = 0
        correct = 0

        with torch.no_grad():
            for batch in testloader:
                # move the batch to device
                batch = [t.to(self.device) for t in batch]
                images, labels = batch

                # get predictions
                logits, _ = self.model(images)

                # calculate the loss
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * len(images)

                # calculate the accuracy
                predictions = torch.argmax(logits, dim=1)
                correct += torch.sum(predictions == labels).item()

        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        
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

    args = parser.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args

def main():
    args = parse_args()

    # training params
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    device = args.device
    save_model_every_n_epochs = args.save_model_every

    # Initialize wandb if enabled
    if args.wandb:
        wandb_config = {
            **config,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": lr,
            "optimizer": "AdamW",
            "weight_decay": 1e-2,
            "device": device,
        }
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.exp_name,
            config=wandb_config
        )

    # load the CIFAR10 Dataset
    trainloader, testloader, _ = prepare_data(batch_size=batch_size)

    # create model, optimizer, loss function and trainer
    model = ViTForClassification(config)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    
    # Watch model parameters and gradients if enabled
    if args.wandb and args.wandb_watch:
        wandb.watch(model, log="all")
    
    trainer = Trainer(model, optimizer, loss_fn, args.exp_name, device=device, use_wandb=args.wandb, log_model=args.wandb_log_model)
    trainer.train(trainloader, testloader, epochs, save_model_every_n_epochs=save_model_every_n_epochs)
    
    # Close wandb run
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
