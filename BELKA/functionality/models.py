import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoModel, AutoConfig, LlamaTokenizer, LlamaModel
from peft import get_peft_model, LoraConfig, TaskType
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd


class BaseBinaryClassifierLightning(pl.LightningModule):
    def __init__(self, learning_rate=2e-5):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()
        self.auroc = BinaryAUROC()

        self.train_loss = 0
        self.val_loss = 0
        self.train_auc = 0
        self.val_auc = 0

    def training_step(self, batch, batch_idx):
        features, labels = batch
        logits = self(features)
        loss = self.loss_fn(logits, labels)
        probs = torch.sigmoid(logits[:, 1])
        auc = self.auroc(probs, labels)

        self.log("train_loss", np.round(loss.detach().cpu().numpy(), 3), prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_auc", np.round(auc.detach().numpy(), 3), prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()

        with torch.no_grad():
            features, labels = batch
            logits = self(features)
            loss = self.loss_fn(logits, labels)
            probs = torch.sigmoid(logits[:, 1])
            auc = self.auroc(probs, labels)

        self.log("val_loss", np.round(loss, 3), prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_auc", np.round(auc, 3), prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
    
    def predict(self, batch):

        smiles_batch, _ = batch 
        logits = self(smiles_batch)
        probabilities = torch.sigmoid(logits)
        return probabilities

class CNNBinaryClassifierLightning(BaseBinaryClassifierLightning):
    def __init__(self, input_dim=142, hidden_dim=32, kernel_size=3, output_dim=1, 
                 stride=1, padding=1, learning_rate=2e-5, max_length = 142):
        super(CNNBinaryClassifierLightning, self).__init__(learning_rate)

        self.max_length = max_length
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=kernel_size, 
                               stride=stride, padding=padding)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=64, kernel_size=kernel_size, 
                               stride=stride, padding=padding)

        self.dropout = nn.Dropout(0.1)
        self.leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2) 

        self.fc1 = nn.Linear(64 * ((self.max_length // 2) // 2), 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, features):
        """
        Forward pass of CNN model
        
        Args:
            features (tensor): encoded SMILES strings
        
        Returns:
            Tensor: Predicted probability scores
        """
        #(batch_size, seq_len)
        x = features.unsqueeze(1)
        # add chanel dimension(batch_size, 1, seq_len)
        x = self.pool(self.leakyrelu(self.conv1(x)))
        x = self.pool(self.leakyrelu(self.conv2(x)))
        
        x = x.view(x.size(0), -1)
        #flatten
        x = self.dropout(self.leakyrelu(self.fc1(x)))
        logits = self.fc2(x)
        
        return logits

class ChemBertaBinaryClassifierLightning(BaseBinaryClassifierLightning):
    def __init__(self, model_name="seyonec/PubChem10M_SMILES_BPE_450k",
                 num_labels=2, 
                 dropout_rate=0.1,
                 learning_rate=2e-5):
        super().__init__(learning_rate)

        self.config = AutoConfig.from_pretrained(model_name)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.config.hidden_size, 128)
        self.fc2 = nn.Linear(128, num_labels)


    def forward(self, features):
        """
        Forward pass through ChemBERTa model

        Args:
            features (Tensor): Preprocessed feature tensor, shape (batch_size, feature_size)

        Returns:
            Tensor: Logits for binary classification
        """
        
        x = self.dropout(self.fc1(features))
        logits = self.fc2(x)

        return logits

    

class MolFormerXLBinaryClassifierLightning(BaseBinaryClassifierLightning):
    def __init__(self, model_name="ibm/MoLFormer-XL-both-10pct", 
                 hidden_dim=256, output_dim=1, 
                 dropout_rate=0.1, learning_rate=2e-5):
        super().__init__(learning_rate)

        self.config = AutoConfig.from_pretrained(model_name)

        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, features):
        """
        Forward pass through MolFormer model

        Args:
            features (Tensor): Preprocessed feature tensor, shape (batch_size, feature_size)

        Returns:
            Tensor: Logits for binary classification
        """
        x = self.dropout(self.fc1(features))
        logits = self.fc2(x)

        return logits
    
