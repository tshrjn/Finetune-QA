import nlp
import torch
import transformers as tfs
import pytorch_lightning as pl

class QAModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.model = tfs.RobertaForQuestionAnswering.from_pretrained(hparams.qa_model)

    def forward(self, x):
        return self.model(**x)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs[0]
        return {'loss': loss, 'log': {'train_loss': loss}}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        pass