import lightning.pytorch as pl
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Adafactor
from metric import MatchMetirc

class InstructABSA(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str = "allenai/tk-instruct-11b-def-pos-neg-expl",
    ):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
        )

        self.metirc = MatchMetirc()

    def training_step(self, batch, batch_idx):
        self.model.train()
        output = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        loss = output.loss
        print(loss)
        print(loss.requires_grad)
        self.log(
            "train/loss",
            loss,
        )
        return loss

    def evaluation_step(self, batch, batch_idx):
        output = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        pred = self.tokenizer.batch_decode(
            output[0],
            skip_special_tokens=True,
        )

        self.metric.update(batch["aspect"], pred)

    def on_evaluation_epoch_end(self):
        print(self.metric.compute())
        self.metirc.reset()

    def predict_step(self, batch, batch_idx):
        output = self.model.generate(
            input_ids=batch["input_ids"].to(self.device),
            attention_mask=batch["attention_mask"].to(self.device),
        )

        pred = self.tokenizer.batch_decode(
            output[0],
            skip_special_tokens=True,
        )

        return pred

    def configure_optimizers(self):
        return Adafactor(
            self.model.parameters(),
            lr=1e-3,
            scale_parameter=False,
            relative_step=False,
        )



