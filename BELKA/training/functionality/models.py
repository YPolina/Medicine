import torch
import gc
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torchmetrics import AUROC, Precision, Recall, F1Score
from transformers import RobertaForSequenceClassification, AutoModel

class ClearMemoryCallback(Callback):
    """
    PyTorch Lightning callback to clear CUDA memory at training checkpoints
    """
    def on_epoch_end(self, trainer, pl_module):
        """
        Clear CUDA cache at the end of each epoch
        
        Args:
            trainer: The PyTorch Lightning trainer
            pl_module: The LightningModule being trained
        """
        torch.cuda.empty_cache()  

    def on_batch_end(self, trainer, pl_module):
        """
        Clear CUDA cache at the end of each batch
        
        Args:
            trainer: The PyTorch Lightning trainer
            pl_module: The LightningModule being trained
        """
        torch.cuda.empty_cache()

class CNNBinaryClassifierLightning(pl.LightningModule):
    """
    PyTorch Lightning module for binary classification using CNN architecture
    
    Args:
        hidden_dim (int): Number of channels in hidden convolutional layers. Default: 32
        kernel_size (int): Size of convolutional kernels. Default: 3
        output_dim (int): Dimension of output layer. Default: 1 (binary classification)
        stride (int): Stride for convolutional layers. Default: 1
        padding (int): Padding for convolutional layers. Default: 1
        learning_rate (float): Initial learning rate. Default: 2e-3
        max_length (int): Maximum length of input sequences. Default: 142
        dropout_rate (float): Dropout rate. Default: 0.1
    """
    def __init__(self, hidden_dim=32, kernel_size=3, output_dim=1, 
                 stride=1, padding=1, learning_rate=2e-3, max_length=142, dropout_rate=0.1):
        super(CNNBinaryClassifierLightning, self).__init__()
        """
        Initialize the CNN classifier with given hyperparameters
        """
        self.save_hyperparameters()

        self.max_length = max_length
        
        # Convolutional layers with BatchNorm
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=kernel_size, 
                               stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=64, kernel_size=kernel_size, 
                               stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm1d(64)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, output_dim)

        # Dropout and activation functions
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU() 
        self.sigmoid = nn.Sigmoid()

        # Loss function for binary classification
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Metrics
        self.auroc = AUROC(task="binary")
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.f1 = F1Score(task="binary")

    def forward(self, features):
        """
        Forward pass of CNN model
        
        Args:
            features (tensor): Encoded SMILES strings
        
        Returns:
            Tensor: Predicted logits
        """
        x = features.unsqueeze(1)

        # Convolutional layers
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))

        # Global average pooling
        x = self.global_avg_pool(x).squeeze(2)  # Shape: (batch_size, 64)

        # Fully connected layers
        x = self.dropout(self.activation(self.fc1(x)))
        logits = self.fc2(x).squeeze(1)  # Shape: (batch_size)

        return logits

    def training_step(self, batch, batch_idx):
        """
         Perform a single training step with metrics calculation and logging
        
        Args:
            batch (tuple): Batch of (features, labels)
            batch_idx (int): Index of current batch
            
        Returns:
            torch.Tensor: Computed loss value
            
        """
        features, labels = batch
        logits = self(features)
        loss = self.loss_fn(logits, labels.float())

        # Compute metrics
        probs = self.sigmoid(logits)
        auroc = self.auroc(probs, labels.int())
        precision = self.precision(probs, labels.int())
        recall = self.recall(probs, labels.int())
        f1 = self.f1(probs, labels.int())

        # Log metrics
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_auroc", auroc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_precision", precision, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_recall", recall, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_f1", f1, prog_bar=True, on_step=False, on_epoch=True)

        gc.collect()
        torch.cuda.empty_cache()

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step with metrics calculation and logging
        
        Args:
            batch (tuple): Batch of (features, labels)
            batch_idx (int): Index of current batch
            
        Returns:
            torch.Tensor: Computed loss value
        """
        features, labels = batch
        logits = self(features)
        loss = self.loss_fn(logits, labels.float())

        # Compute metrics
        probs = self.sigmoid(logits)
        auroc = self.auroc(probs, labels.int())
        precision = self.precision(probs, labels.int())
        recall = self.recall(probs, labels.int())
        f1 = self.f1(probs, labels.int())

        # Log metrics
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_auroc", auroc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_precision", precision, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_recall", recall, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_f1", f1, prog_bar=True, on_step=False, on_epoch=True)

        gc.collect()
        torch.cuda.empty_cache()
        
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler
        
        Returns:
            dict: Dictionary containing:
                - "optimizer": Initialized AdamW optimizer
                - "lr_scheduler": Dictionary with ReduceLROnPlateau scheduler configuration
                                  monitoring validation loss
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=2, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss", 
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def predict_step(self, batch, batch_idx):
        """
        Perform prediction on a single batch
        
        Args:
            batch (torch.Tensor): Input features
            batch_idx (int): Batch index
            
        Returns:
            torch.Tensor: Predicted probabilities (after sigmoid activation)
        """
        features = batch
        logits = self(features)
        probs = self.sigmoid(logits)
        return probs
    
class ChemBertBinaryClassifier(pl.LightningModule):
    """
    A PyTorch Lightning module for binary classification using a pretrained ChemBERTa model
    
    This class implements a chemical-aware BERT model fine-tuned for binary classification tasks,
    with frozen base layers (except last) and trainable classification head. Includes metrics
    tracking (AUROC, precision, recall, F1) and learning rate scheduling.

    Args:
        model_name (str): Name or path of the pretrained ChemBERTa model
                         Default: 'seyonec/PubChem10M_SMILES_BPE_450k'
        learning_rate (float): Initial learning rate for the optimizer. Default: 2e-5
        num_labels (int): Number of output labels

    Attributes:
        model (RobertaForSequenceClassification): The underlying ChemBERTa model
        loss_fn (nn.BCEWithLogitsLoss): Binary cross-entropy loss with logits
        auroc (AUROC): Area Under ROC Curve metric
        precision (Precision): Precision metric
        recall (Recall): Recall metric
        f1 (F1Score): F1 score metric
    """
    def __init__(self, model_name='seyonec/PubChem10M_SMILES_BPE_450k', learning_rate=2e-5, num_labels=1):
        super().__init__()
        """
        Initialize the ChemBERTa classifier with given hyperparameters
        
        Loads pretrained model, freezes all layers except last transformer layer and classifier,
        and initializes metrics and loss function
        """
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

        # Freeze all layers except the last 2
        for param in self.model.roberta.parameters():
            param.requires_grad = False
        for param in self.model.roberta.encoder.layer[-1].parameters(): 
            param.requires_grad = True
        for param in self.model.classifier.parameters(): 
            param.requires_grad = True

        # Loss function for binary classification
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Metrics
        self.auroc = AUROC(task="binary")
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.f1 = F1Score(task="binary")

    def forward(self, input_ids, attention_mask):
        """
        Perform forward pass through the model
        
        Args:
            input_ids (torch.Tensor): Tokenized input IDs tensor of shape (batch_size, seq_length)
            attention_mask (torch.Tensor): Attention mask tensor of shape (batch_size, seq_length)
            
        Returns:
            torch.Tensor: Model output logits of shape (batch_size,)
        """
        # Forward pass through the model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits.squeeze(1)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels'].float() 

        # Forward pass
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)

        # Compute metrics
        probs = torch.sigmoid(logits)  
        auroc = self.auroc(probs, labels.int())
        precision = self.precision(probs, labels.int())
        recall = self.recall(probs, labels.int())
        f1 = self.f1(probs, labels.int())

        # Log metrics
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_auroc", auroc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_precision", precision, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_recall", recall, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_f1", f1, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels'].float() 

        # Forward pass
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)

        # Compute metrics
        probs = torch.sigmoid(logits) 
        auroc = self.auroc(probs, labels.int())
        precision = self.precision(probs, labels.int())
        recall = self.recall(probs, labels.int())
        f1 = self.f1(probs, labels.int())

        # Log metrics
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_auroc", auroc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_precision", precision, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_recall", recall, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_f1", f1, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss", 
                "frequency": 1,
            },
        }

    def predict_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        # Forward pass
        logits = self(input_ids, attention_mask)
        probs = torch.sigmoid(logits) 
        return probs


    
class MolFormerClassifier(pl.LightningModule):
    """
    A PyTorch Lightning module for molecular property prediction using MolFormer
    
    This class implements a transformer-based molecular classifier using IBM's MolFormer model,
    with frozen base layers (except last) and a trainable classification head. Includes standard
    binary classification metrics and learning rate scheduling.

    Args:
        model_name (str): Name or path of the pretrained MolFormer model(by default 'ibm/MoLFormer-XL-both-10pct')
        learning_rate (float): Initial learning rate for the optimizer(by default 2e-5)
        num_labels (int): Number of output labels 

    Attributes:
        molformer (AutoModel): The underlying MolFormer transformer model
        classifier (nn.Linear): Custom classification head
        loss_fn (nn.BCEWithLogitsLoss): Binary cross-entropy loss with logits
        auroc (AUROC): Area Under ROC Curve metric
        precision (Precision): Precision metric
        recall (Recall): Recall metric
        f1 (F1Score): F1 score metric
    """
    def __init__(self, model_name='ibm/MoLFormer-XL-both-10pct', learning_rate=2e-5, num_labels=1):
        """
        Initialize the MolFormer classifier with given hyperparameters
        
        Loads pretrained model, freezes all layers except last transformer layer,
        adds custom classification head, and initializes metrics and loss function
        """
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # Load the pre-trained MolFormer model
        self.molformer = AutoModel.from_pretrained(model_name, trust_remote_code = True)
        
        for param in self.molformer.parameters():
            param.requires_grad = False

        # Unfreeze the last transformer layer
        for param in self.molformer.encoder.layer[-1].parameters():
            param.requires_grad = True

        # Add a custom classification head
        self.classifier = nn.Linear(self.molformer.config.hidden_size, num_labels)

        # Loss function for binary classification
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Metrics
        self.auroc = AUROC(task="binary")
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.f1 = F1Score(task="binary")

    def forward(self, input_ids, attention_mask):
        """
        Perform forward pass through the MolFormer model
        
        Args:
            input_ids (torch.Tensor): Tokenized input IDs tensor of shape (batch_size, seq_length)
            attention_mask (torch.Tensor): Attention mask tensor of shape (batch_size, seq_length)
            
        Returns:
            torch.Tensor: Model output logits of shape (batch_size,)
        """
        outputs = self.molformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use mean pooling over the sequence length
        pooled_output = outputs.last_hidden_state.mean(dim=1) 
        
        # Pass through the classifier
        logits = self.classifier(pooled_output).squeeze(1) 
        return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels'].float()  

        # Forward pass
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)

        # Compute metrics
        probs = torch.sigmoid(logits) 
        auroc = self.auroc(probs, labels.int())
        precision = self.precision(probs, labels.int())
        recall = self.recall(probs, labels.int())
        f1 = self.f1(probs, labels.int())

        # Log metrics
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_auroc", auroc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_precision", precision, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_recall", recall, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_f1", f1, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels'].float() 

        # Forward pass
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)

        # Compute metrics
        probs = torch.sigmoid(logits) 
        auroc = self.auroc(probs, labels.int())
        precision = self.precision(probs, labels.int())
        recall = self.recall(probs, labels.int())
        f1 = self.f1(probs, labels.int())

        # Log metrics
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_auroc", auroc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_precision", precision, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_recall", recall, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_f1", f1, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        # Optimizer 
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def predict_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        # Forward pass
        logits = self(input_ids, attention_mask)
        probs = torch.sigmoid(logits) 
        return probs

