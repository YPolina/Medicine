import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModel, AutoConfig
import torch.optim as optim
import lightning as L
from torchmetrics.classification import AUROC
from pytorch_lightning.callbacks import ModelCheckpoint

class CNNBinaryClassifier(nn.Module):

    def __init(self, input_dim = 142, hidden_dim = 32, kernel_size = 3, output_dim = 1, stride = 1, padding = 1):
        super(CNNBinaryClassifier, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, stride = stride, padding=padding)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=64, kernel_size=kernel_size, stride = stride, padding = padding)

        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool1d(kernel_size=2, stride = stride)
        self.fc1 = nn.Linear(64 * (input_dim // 4), 128)
        self.fc2 = nn. Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))

        return x
    
class ChemBertaBinaryClassifier(nn.Module):

    def __init__(self, model_name="seyonec/PubChem10M_SMILES_BPE_450k", num_labels=2, hidden_size=768, dropout_rate=0.1):
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        for param in self.model.parameters():
            param.requiers_grad = False

        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.config.hidden_size, 128)
        self.fc2 = nn.Linear(128, 1)

        self.loss = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):

        outputs = self.model(input_ids = input_ids, attention_mask = attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]

        x = self.relu(self.fc1(cls_embedding))
        logits = self.fc2(self.dropout(x))

        return logits

    def compute_loss(self, input_ids, attention_mask, label_ids):

        logits = self(input_ids, attention_mask).squeeze()
        loss = self.loss(logits, label_ids.float())

        return logits, loss



    
class ChemBertaLightningClassifier(L.LightningModule):

    def __init__(self, model_name="seyonec/PubChem10M_SMILES_BPE_450k", lr = 1e-4):
        super().__init__()
        self.model = ChemBertaBinaryClassifier(model_name)
        self.lr = lr  
        self.map = AUROC(task="binary")

    def forward(self, batch):

        return self.model(batch['input_ids'], batch['attention_mask'])
    
    def training_step(self, batch, batch_idx):

        loss, logits = self.model.compute_loss(batch['input_ids'], batch['attention_mask'], batch['label_ids'])
        self.log("training loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def on_train_batch_end(self, loss, batch, batch_idx):
        if batch_idx % 2 == 0:
            print(f"Batch {batch_idx}: Loss = {loss.item()}")

    def on_train_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            model_filename = f"ChemBert_{val_loss:.4f}.pth"
            torch.save(self.state_dict(), model_filename)
            print(f"Model saved as {model_filename}")

    def validation_step(self, batch, batch_idx):

        loss, logits = self.model.compute_loss(batch['input_ids'], batch['attention_mask'], batch['label_ids'])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.map.update(torch.sigmoid(logits), batch["label_ids"].long())

    def on_validation_epoch_end(self):

        val_map = self.map.compute()
        self.log("val_map", val_map, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.map.reset()

    def predict(self, batch):

        logits = self(batch['input_ids'], batch['attention_mask'])
        probabilities = torch.sigmoid(logits)  
        return probabilities
    
    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

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





            





        
