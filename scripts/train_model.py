import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer
import numpy as np
from pathlib import Path
import logging
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from torch.cuda.amp import autocast, GradScaler

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class FastFlagDetectorModel(nn.Module):
    def __init__(self, num_labels=2):
        super().__init__()
        # Use DistilRoBERTa instead of RoBERTa
        self.distilroberta = AutoModel.from_pretrained('distilroberta-base')
        
        # Freeze embeddings and first 4 layers
        modules_to_freeze = [
            self.distilroberta.embeddings,
            *self.distilroberta.encoder.layer[:4]
        ]
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False
        
        # Efficient classifier head
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.distilroberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled_output)

class OptimizedTrainer:
    def __init__(self, max_length=128):
        logger.info("Initializing OptimizedTrainer...")
        self.device = 'cpu'
        self.max_length = max_length
        
        # Optimize CPU performance
        torch.set_num_threads(4)
        
        # Create directories
        self.model_dir = Path('models/bert_model/saved_model')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = FastFlagDetectorModel().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
        
        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler()
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def load_and_preprocess_data(self, batch_size=32):
        logger.info("Loading and preprocessing data...")
        try:
            # Load data
            data_dir = Path('data/processed')
            
            # Load and prepare training data
            train_data = self.load_data_efficiently(data_dir, 'train')
            test_data = self.load_data_efficiently(data_dir, 'test')
            
            # Create dataloaders with efficient batch size
            train_loader = DataLoader(
                train_data, 
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True
            )
            
            test_loader = DataLoader(
                test_data,
                batch_size=batch_size,
                pin_memory=True
            )
            
            return train_loader, test_loader
            
        except Exception as e:
            logger.error(f"Error in data loading: {e}")
            raise
            
    def load_data_efficiently(self, data_dir, prefix):
        # Load data with proper typing
        input_ids = torch.from_numpy(
            np.load(f"{data_dir}/{prefix}_input_ids.npy")
        ).long()
        
        attention_mask = torch.from_numpy(
            np.load(f"{data_dir}/{prefix}_attention_mask.npy")
        ).long()
        
        labels = torch.from_numpy(
            np.load(f"{data_dir}/{prefix}_labels.npy")
        ).long()
        
        return TensorDataset(input_ids, attention_mask, labels)
    
    def train(self, train_loader, val_loader, epochs=5, 
              learning_rate=5e-5, gradient_accumulation_steps=4):
        logger.info("Starting training...")
        
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate
        )
        criterion = nn.CrossEntropyLoss()
        
        best_val_accuracy = 0
        total_steps = len(train_loader) * epochs
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            optimizer.zero_grad()
            
            batch_iterator = tqdm(
                train_loader,
                desc=f"Training Epoch {epoch+1}/{epochs}",
                leave=True
            )
            
            for i, (input_ids, attention_mask, labels) in enumerate(batch_iterator):
                # Move data to device
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass with mixed precision
                with autocast():
                    outputs = self.model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    loss = loss / gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                if (i + 1) % gradient_accumulation_steps == 0:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * gradient_accumulation_steps
                
                # Update progress bar
                batch_iterator.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{total_loss/(i+1):.4f}"
                })
            
            # Validation phase
            val_loss, val_accuracy = self.evaluate(val_loader, criterion)
            
            # Save metrics
            self.train_losses.append(total_loss / len(train_loader))
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save_model('best_model.pth')
                logger.info(f"New best model saved! Accuracy: {val_accuracy:.4f}")
            
            logger.info(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
    def evaluate(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for input_ids, attention_mask, labels in dataloader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))
        return total_loss / len(dataloader), accuracy
    
    def save_model(self, filename):
        path = self.model_dir / filename
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def plot_metrics(self):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.title('Validation Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.close()

def main():
    try:
        # Initialize trainer
        trainer = OptimizedTrainer(max_length=128)
        
        # Load and preprocess data
        train_loader, val_loader = trainer.load_and_preprocess_data(batch_size=32)
        
        # Train model
        trainer.train(
            train_loader,
            val_loader,
            epochs=5,
            learning_rate=5e-5,
            gradient_accumulation_steps=4
        )
        
        # Plot metrics
        trainer.plot_metrics()
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in training: {e}")
        raise

if __name__ == "__main__":
    main()