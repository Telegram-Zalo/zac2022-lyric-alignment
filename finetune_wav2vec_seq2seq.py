import warnings

warnings.filterwarnings("ignore")

import collections
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2Processor,
)

from transformers.trainer import (
    DistributedSamplerWithLoop,
    SequentialDistributedSampler,
    SequentialSampler,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.training_args import ParallelMode


from dataloaders import CustomDataset, DataCollatorCTCWithPadding
from models import Wav2Vec2ForCTCV2

# seed
torch.manual_seed(2022)
np.random.seed(2022)
random.seed(2022)


class CommonVoiceTrainer(Trainer):
    def _get_train_sampler(self):
        if isinstance(
            self.train_dataset, torch.utils.data.IterableDataset
        ) or not isinstance(self.train_dataset, collections.abc.Sized):
            return None

        if self.args.world_size <= 1:
            return RandomSampler(self.train_dataset)
        elif (
            self.args.parallel_mode == ParallelMode.TPU
            and not self.args.dataloader_drop_last
        ):
            # Use a loop for TPUs when drop_last is False to have all batches have the same size.
            return DistributedSamplerWithLoop(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
            )
        else:
            return DistributedSampler(
                self.train_dataset,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
            )

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _get_eval_sampler(self, eval_dataset):
        if self.args.local_rank != -1:
            return SequentialDistributedSampler(eval_dataset)
        else:
            return SequentialSampler(eval_dataset)

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


if __name__ == "__main__":
    processor = Wav2Vec2Processor.from_pretrained(
        "./pretrained/dragonSwing/wav2vec2-base-vietnamese/"
    )
    model = Wav2Vec2ForCTCV2.from_pretrained(
        "./pretrained/dragonSwing/wav2vec2-base-vietnamese/"
    )
    model.freeze_feature_extractor()
    ADD_STOP_END = False

    print("NEW VOCAB_SIZE: ", len(processor.tokenizer.get_vocab()))
    model.resize_lm_head(new_num_tokens=len(processor.tokenizer.get_vocab()))

    data_collator = DataCollatorCTCWithPadding(
        get_output_lengths_function=model._get_feat_extract_output_lengths,
        processor=processor,
        padding=True,
    )

    train_dataset = CustomDataset(
        f"train.csv",
        "./data/",
        processor=processor,
        is_train=True,
        n_words_range=(10, 40),
        column_names=["input_values", "labels"],
    )

    OUTPUT_DIR = os.path.join(
        "./checkpoints/",
        "dragonSwing/wav2vec2-base-vietnamese",
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        group_by_length=True,
        per_device_train_batch_size=4,
        num_train_epochs=50,
        fp16=True,
        gradient_checkpointing=False,
        save_steps=500,
        logging_steps=50,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=4,
        dataloader_num_workers=4,
        gradient_accumulation_steps=2,
    )

    trainer = CommonVoiceTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=processor.feature_extractor,
    )

    last_checkpoint = None

    if os.path.exists(OUTPUT_DIR):
        last_checkpoint = get_last_checkpoint(OUTPUT_DIR)

    if last_checkpoint:
        print(f"last_checkpoint: {last_checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        train_result = trainer.train()
