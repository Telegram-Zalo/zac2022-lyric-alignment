from torch.utils.data import Dataset
import librosa
import soundfile as sf
import os
import pandas as pd
import numpy as np
import re
import torch
from transformers import Wav2Vec2Processor
from typing import Dict, List, Optional, Union
import json
import math
import string


chars_to_ignore_regex = r"[" + string.punctuation + "]"

REMOVE_CHARACTOR_LIST = [
    "̃",
    "̀",
    "́",
    "”",
    "…",
    "̣",
    "“",
    "’",
    "\\",
]


def remove_special_characters(text, add_stop_end=False):
    text = re.sub(chars_to_ignore_regex, "", text).lower()
    text = "".join([c for c in text if c not in REMOVE_CHARACTOR_LIST])
    if add_stop_end:
        text = "<s>" + text + "</s>"
    return text


class CustomDataset(Dataset):
    def __init__(
        self,
        csv_file,
        root_dir,
        processor,
        is_train=False,
        n_words_range=(40, 70),
        column_names=None,
        sep="\t",
    ):
        self.data = pd.read_csv(os.path.join(root_dir, csv_file), sep=sep)
        self.processor = processor
        self.column_names = column_names
        self.n_words_range = n_words_range
        self.is_train = is_train
        self.target_samplerate = 16000

    def __len__(self):
        return len(self.data)

    def simple_trimp(self, np_audio):
        np_audio = librosa.effects.trim(np_audio, top_db=30)[0]
        return np_audio

    def speech_file_to_array_fn(self, batch):
        speech_array, sampling_rate = sf.read(
            batch["file"].replace("songs_denoise", "songs_denoise_16k")
        )
        if len(speech_array.shape) == 2:
            speech_array = speech_array[..., 0]
        if sampling_rate != self.target_samplerate:
            speech_array = librosa.resample(
                speech_array, orig_sr=sampling_rate, target_sr=self.target_samplerate
            )
        batch["speech"] = speech_array
        batch["sampling_rate"] = 16000  # for the sake of init model
        return batch

    def random_segment(self, batch):
        transcript, word_durations_gts = self.get_label(
            batch["file"].replace("songs_denoise", "labels").replace(".wav", ".json")
        )
        speech = batch["speech"]
        max_n_words = np.random.randint(self.n_words_range[0], self.n_words_range[1])
        if len(transcript) > max_n_words and self.is_train:
            start_idx_rand = np.random.randint(0, len(transcript) - max_n_words)
            transcript = transcript[start_idx_rand : start_idx_rand + max_n_words]
            speech_start_idx = word_durations_gts[start_idx_rand][0]
            speech_end_idx = word_durations_gts[start_idx_rand + max_n_words - 1][1]
            speech = speech[speech_start_idx:speech_end_idx]

        transcript = " ".join(transcript)
        batch["text"] = transcript
        batch["speech"] = self.simple_trimp(speech)
        return batch

    def remove_special_characters(self, batch):
        batch["text"] = remove_special_characters(batch["text"])
        return batch

    def prepare_dataset(self, batch, column_names=None):
        batch["input_values"] = (
            self.processor(batch["speech"], sampling_rate=batch["sampling_rate"])
            .input_values[0]
            .tolist()
        )

        with self.processor.as_target_processor():
            batch["labels"] = self.processor(batch["text"]).input_ids

        if column_names and isinstance(column_names, list):
            batch = {name: batch[name] for name in column_names}

        return batch

    def get_label(self, path):
        with open(path, "rb") as fp:
            label = json.load(fp)

        word_durations_gts = []
        transcript = []
        for segment in label:
            l = segment["l"]
            for words in l:
                s = math.floor((words["s"] / 1000.0) * self.target_samplerate)
                e = math.floor((words["e"] / 1000.0) * self.target_samplerate)
                word_durations_gts.append([s, e])
                transcript.append(words["d"])

        return transcript, word_durations_gts

    def get_label_segment(self, path):
        with open(path, "rb") as fp:
            label = json.load(fp)

        segment_durations_gts = []
        segment_transcript = []

        for segment in label:
            l = segment["l"]
            transcript = []
            for words in l:
                transcript.append(words["d"].strip())
            segment_transcript.append(" ".join(transcript))
            segment_durations_gts.append(
                [int(segment["s"] / 1000.0 * 16000), int(segment["e"] / 1000.0 * 16000)]
            )

        return np.array(segment_transcript), np.array(segment_durations_gts)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        batch = self.data.iloc[idx].copy()
        batch = batch.to_dict()
        batch = self.speech_file_to_array_fn(batch)
        if self.is_train and self.n_words_range != None:
            batch = self.random_segment(batch)
        batch = self.remove_special_characters(batch)
        batch = self.prepare_dataset(batch, self.column_names)
        return batch


# @dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
                        The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
                        Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
                        among:
                        * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                            sequence if provided).
                        * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                            maximum acceptable input length for the model if that argument is not provided.
                        * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                            different lengths).
        max_length (:obj:`int`, `optional`):
                        Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
                        Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
                        If set will pad the sequence to a multiple of the provided value.
                        This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
                        7.5 (Volta).
    """

    def __init__(
        self,
        get_output_lengths_function,
        processor: Wav2Vec2Processor,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        max_length_labels: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        pad_to_multiple_of_labels: Optional[int] = None,
    ):
        self.get_output_lengths_function = get_output_lengths_function
        self.processor = processor
        self.padding = padding
        self.max_length = max_length
        self.max_length_labels = max_length_labels
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_to_multiple_of_labels = pad_to_multiple_of_labels

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        # mel
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]

        # text
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        batch["input_ids"] = labels_batch["input_ids"]
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels  # [B, T]
        return batch
