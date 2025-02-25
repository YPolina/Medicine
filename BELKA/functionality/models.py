import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
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
        super(BaseBinaryClassifierLightning, self).__init__()
        
        self.learning_rate = learning_rate
        self.auroc = BinaryAUROC()
        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits.squeeze(), y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits.squeeze(), y)
        auroc = self.auroc(logits.squeeze(), y)
        return {"val_loss": loss, "val_auroc": auroc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_auroc = torch.stack([x['val_auroc'] for x in outputs]).mean()

        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("val_auroc", avg_auroc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3, verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }
    
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

    def CNN_features(self, smiles_batch):
        """
        Encodes SMILES strings into fixed-length numerical representations
        
        Args:
            smiles_batch (list of str): Batch of SMILES strings

        Returns:
            Tensor: Encoded and padded SMILES representations
        """
        enc_dict = {
            'l': 1, 'y': 2, '@': 3, '3': 4, 'H': 5, 'S': 6, 'F': 7, 'C': 8, 'r': 9, 's': 10, '/': 11, 'c': 12, 'o': 13,
            '+': 14, 'I': 15, '5': 16, '(': 17, '2': 18, ')': 19, '9': 20, 'i': 21, '#': 22, '6': 23, '8': 24, '4': 25, '=': 26,
            '1': 27, 'O': 28, '[': 29, 'D': 30, 'B': 31, ']': 32, 'N': 33, '7': 34, 'n': 35, '-': 36
        }
        
        def encode_smile(smile):
            encoded = [enc_dict.get(char, 0) for char in smile]  
            padded = encoded + [0] * (self.max_length - len(encoded))
            return np.array(padded[:self.max_length], dtype=np.uint8)

        smiles_enc = np.stack([encode_smile(smile) for smile in smiles_batch])
        
        return torch.tensor(smiles_enc, dtype=torch.float32)

    def forward(self, smiles_batch):
        """
        Forward pass of CNN model
        
        Args:
            smiles_batch (list of str): Batch of SMILES strings
        
        Returns:
            Tensor: Predicted probability scores
        """
        x = self.CNN_features(smiles_batch)
        #(batch_size, seq_len)
        x = x.unsqueeze(1)
        # add chanel dimension(batch_size, 1, seq_len)
        x = self.pool(self.leakyrelu(self.conv1(x)))
        x = self.pool(self.leakyrelu(self.conv2(x)))
        
        x = x.view(x.size(0), -1)
        #flatten
        x = self.dropout(self.leakyrelu(self.fc1(x)))
        logits = self.fc2(x)
        output = self.sigmoid(logits)

        return output

class ChemBertaBinaryClassifierLightning(BaseBinaryClassifierLightning):
    def __init__(self, model_name="seyonec/PubChem10M_SMILES_BPE_450k",
                 num_labels=2, 
                 dropout_rate=0.1,
                 learning_rate=2e-5):
        super().__init__(learning_rate)

        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        for param in self.model.parameters():
            param.requires_grad = False

        #unfreeze last layer for fine-tuning
        for param in self.model.encoder.layer[-1].parameters():  
            param.requires_grad = True

        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.config.hidden_size, 128)
        self.fc2 = nn.Linear(128, num_labels)


    def forward(self, smiles_batch):
        """
        Forward pass through ChemBERTa model

        Args:
            smiles_batch (list of str): Batch of SMILES strings

        Returns:
            Tensor: Logits for binary classification
        """
        tokens = self.tokenizer(smiles_batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
        tokens = {key: val for key, val in tokens.items()}

        outputs = self.model(**tokens)
        cls_embedding = outputs.last_hidden_state[:, 0]

        x = self.fc1(cls_embedding)
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits


class MolLlamaBinaryClassifierLightning(BaseBinaryClassifierLightning):
    def __init__(self, llama_model="meta-llama/Llama-2-7b-chat-hf", 
                 num_classes=2, embedding_dim=768,
                 num_heads=8, num_layers=2, 
                 use_lora=True, learning_rate=2e-5):
        super().__init__(learning_rate)

        self.llama = LlamaModel.from_pretrained(llama_model)

        if use_lora:
            config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=16, lora_alpha=32, lora_dropout=0.1,
                target_modules=["query", "key", "value", "dense"]
            )
            self.llama = get_peft_model(self.llama, config)

        self.tokenizer = LlamaTokenizer.from_pretrained(llama_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads)
        self.self_attention = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projection = nn.Linear(embedding_dim, self.llama.config.hidden_size)
        self.classifier = nn.Linear(self.llama.config.hidden_size, num_classes)

    def forward(self, smiles_batch):
        """
        Forward pass through MolLLaMA model

        Args:
            smiles_batch (list of str): Batch of SMILES strings

        Returns:
            Tensor: Logits for binary classification
        """
        tokens = self.tokenizer(smiles_batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
        tokens = {key: val for key, val in tokens.items()}

        outputs = self.model(input_ids=tokens["input_ids"])
        pooled_output = outputs.last_hidden_state.mean(dim=1)

        x = self.fc1(pooled_output)
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits
    

class MolFormerXLBinaryClassifierLightning(BaseBinaryClassifierLightning):
    def __init__(self, model_name="ibm/MoLFormer-XL-both-10pct", 
                 hidden_dim=256, output_dim=1, 
                 dropout_rate=0.1, learning_rate=2e-5, use_lora = True):
        super().__init__(learning_rate)

        self.molformer = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False, 
                r=16,
                lora_alpha=32, 
                lora_dropout=0.1,
                target_modules=["query", "key", "value", "dense"]
            )
            self.molformer = get_peft_model(self.molformer, lora_config)

        self.classifier = nn.Sequential(
            nn.Linear(self.molformer.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, smiles_batch):
        tokens = self.tokenizer(
            smiles_batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        tokens = {key: val for key, val in tokens.items()}

        outputs = self.molformer(**tokens)
        cls_embedding = outputs.last_hidden_state[:, 0]

        logits = self.classifier(cls_embedding).squeeze(-1)
        return logits
    
