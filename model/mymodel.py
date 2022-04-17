from typing import Any, Optional

import pytorch_lightning as pl
import torch
# from pytorch_lightning import Trainer, seed_everything
# from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
# 自动停止
# https://pytorch-lightning.readthedocs.io/en/1.2.1/common/early_stopping.html
import torch.optim as optim
from transformers import AutoConfig, AutoModel, BertTokenizer, AutoModelForSequenceClassification
from torchmetrics.functional import accuracy, precision_recall
from torchmetrics import AUROC
import torch.nn as nn


class myModel(pl.LightningModule):
    """
    基础的命名实体
    简化版本丢弃矩阵相乘

    """

    def __init__(
            self, learning_rate=3e-4, T_max=5,
            optimizer_name="AdamW",
            pretrained="uer/chinese_roberta_L-2_H-128",
            batch_size=2,
            trainfile="./data/train.pkt",
            valfile="./data/val.pkt",
            testfile="./data/test.pkt", **kwargs):
        super().__init__()
        self.save_hyperparameters()
        print(self.hparams)
        # pretrained="/data_200t/chenyaozu/data/base_model/chinese_roberta_L-4_H-512/"
        config = AutoConfig.from_pretrained(pretrained)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        # config.num_labels=self.hparams.max_len
        config.output_attentions = True
        self.hparams.config = config
        # self.model = BertForPreTraining.from_pretrained(pretrained, config=config)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained, config=config)

        self.save_hyperparameters()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        x = self.model(input_ids, token_type_ids=token_type_ids.long(), attention_mask=attention_mask, labels=labels)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward

        input_ids, token_type_ids, attention_mask, label = batch
        x = self(input_ids, token_type_ids, attention_mask, label)

        acc = accuracy(x.logits.argmax(-1), label)
        auroc = AUROC(pos_label=1)
        myauroc = auroc(x.logits, label)
        metrics = {

            "train_acc": acc,
            "train_auroc": myauroc

        }
        # # print("metrics",metrics)
        self.log_dict(metrics)
        self.log('train_loss', x.loss)
        return x.loss

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward

        input_ids, token_type_ids, attention_mask, label = batch
        x = self(input_ids, token_type_ids, attention_mask, label)

        acc = accuracy(x.logits.argmax(-1), label)
        auroc = AUROC(pos_label=1)
        myauroc = auroc(x.logits, label)
        metrics = {

            "val_acc": acc,
            "val_auroc": myauroc,
            "val_loss":  x.loss

        }
        # # print("metrics",metrics)
        self.log_dict(metrics)
        return metrics

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        # training_step defined the train loop.
        # It is independent of forward
        input_ids, token_type_ids, attention_mask = batch

        input_ids, token_type_ids, attention_mask = input_ids.view(-1, self.hparams.max_len), token_type_ids.view(
            -1, self.hparams.max_len), attention_mask.view(-1, self.hparams.max_len)

        logits, outType, _ = self(input_ids, token_type_ids, attention_mask)
        # active_loss = attention_mask.view(-1) > -100
        # testf = open("data/test_pos_base_new_testpredict_step.txt", "a+")
        outdata = []
        with open("data/test_pos_base_predict_step.txt", "a+") as f:
            for i, (x, out_pos, out_type, masks) in enumerate(zip(input_ids.tolist(),
                                                                  logits.argmax(
                                                                      dim=-1).tolist(),
                                                                  outType.argmax(
                                                                      dim=-1).tolist(),
                                                                  attention_mask.tolist(),
                                                                  )):
                #             print(p,y)
                words = self.tokenizer.convert_ids_to_tokens(x)
                #                 print(words)
                word_dict = {}
                word_y_dict = {}

                for ii, (pit, tx, mask) in enumerate(zip(out_pos, out_type, masks)):
                    #                     if pit!=0 and  yit!=0 and mask!=0:
                    # if mask != 0:
                    #     testf.write(words[ii] + "," + str(int(yit)) + "," + str(int(ty)) +
                    #                 "\n")
                    if pit != 0 and mask != 0 and tx != 0:
                        word_y_dict[ii] = words[ii:ii + int(pit)]
                        word_y_dict[ii] = tx

                outdata.append({"words": words, "word_dict": word_dict, "word_y_dict": word_y_dict})
        return outdata

    def train_dataloader(self):
        train = torch.load(self.hparams.trainfile)
        return DataLoader(train, batch_size=int(self.hparams.batch_size), num_workers=24, pin_memory=True,
                          shuffle=True)

    def val_dataloader(self):
        val = torch.load(self.hparams.valfile)
        return DataLoader(val, batch_size=int(self.hparams.batch_size), num_workers=24, pin_memory=True)

    def test_dataloader(self):
        val = torch.load(self.hparams.testfile)
        return DataLoader(val, batch_size=int(self.hparams.batch_size), num_workers=24, pin_memory=True)

    def configure_optimizers(self):
        """优化器 自动优化器"""
        optimizer = getattr(optim, self.hparams.optimizer_name)(self.parameters(), lr=self.hparams.learning_rate)
        #         使用自适应调整模型
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=500,factor=0.8,verbose=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=2, eta_min=0,
                                                                         verbose=False)
        lr_scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
            'name': "lr_scheduler",
            'monitor': 'val_loss',  # 监听数据变化
            'strict': True,
        }
        #         return [optimizer], [lr_scheduler]
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
