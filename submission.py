import copy
import json

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm
from dataloaders import (
    remove_special_characters,
)
from tqdm import tqdm
import pandas as pd
from alignment import get_duration_from_emission
import glob
from typing import Dict, List, Optional, Tuple, Union
import os
import click
import logging
from denoiser import get_denoiser, run_denoiser
from transformers import Wav2Vec2Processor, Wav2Vec2Config
from models import Wav2Vec2ForCTCV2


@click.group()
def cli():
    logging.basicConfig(
        format="%(asctime)12s - %(levelname)s - %(message)s", level=logging.INFO
    )


def generate_pred(se, audio, labels, segments, char_durations):
    preds = []
    charactor_index = 0
    start_offet = se[0] // 320.0
    start = start_offet + segments[0].start
    if start <= 15:
        start = 0
    end = 0
    for seg_idx, segment in enumerate(labels):
        segment_l = segment["l"]
        pred_segment_l = []
        for word_segment_idx, word_segment in enumerate(segment_l):
            word = word_segment["d"]
            # must to apply remove_special_characters for word to prevent charactor shift
            # if len(remove_special_characters(word)) != len(word):
            # 	print(word, " ", remove_special_characters(word))
            len_word = len(remove_special_characters(word))
            word_duration = sum(
                char_durations[charactor_index : charactor_index + len_word + 1]
            )
            end = start + word_duration
            if seg_idx == len(labels) - 1 and word_segment_idx == len(segment_l) - 1:
                # always at the end of audio
                pred_segment_l.append(
                    {
                        "s": int(start * 320.0 / 16000.0 * 1000),
                        "e": len(audio) / 16000.0 * 1000,
                        "d": word,
                    }
                )
            else:
                pred_segment_l.append(
                    {
                        "s": int(start * 320.0 / 16000.0 * 1000),
                        "e": int(end * 320.0 / 16000.0 * 1000),
                        "d": word,
                    }
                )
            # update char index
            charactor_index = charactor_index + len_word + 1
            start = end

        preds.append({"s": 0, "e": 0, "l": pred_segment_l})
    return preds


def get_emission(model, processor, audio, tokens, device):
    if device == "cuda":
        float16 = True
    input_values = processor(
        audio.astype(np.float32),
        return_tensors="pt",
        padding="longest",
        sampling_rate=16000,
    ).input_values  # Batch size 1
    with torch.no_grad():
        input_values = input_values.to(device)
        if float16:
            input_values = input_values.half()
        model_outputs = model(
            input_values,
            labels=torch.from_numpy(np.array(tokens)[None, ...]).long().to(device),
        )
        logits = model_outputs.logits.float()
    emissions = torch.log_softmax(logits, dim=-1)
    emissions = emissions[0].cpu().detach()  # [T_mel, 98]
    return emissions, logits


def run(song_path, model, processor, separator, device, saved_path):
    if separator:
        audio, sr = run_denoiser(separator, song_path)

    if len(audio.shape) == 2:
        audio = audio[..., 0]

    if sr != 16000:
        audio = librosa.resample(audio, sr, 16000, res_type="kaiser_best")

    # step 1: trimp audio and return start, end index
    audio_trim, se = librosa.effects.trim(audio, top_db=30)

    # step 2: read json labels and handle it
    with open(
        song_path.replace("songs", "new_labels_json").replace(".wav", ".json"),
        "rb",
    ) as fp:
        labels = json.load(fp)

    transcript = []
    for segment in labels:
        l = segment["l"]
        for words in l:
            transcript.append(words["d"])

    transcript = " ".join(transcript)
    orig_transcript = copy.copy(transcript)
    orig_transcript = orig_transcript.split(" ")
    transcript = remove_special_characters(transcript, add_stop_end=False)

    # step 4: get char duration
    with processor.as_target_processor():
        tokens = processor(transcript.lower()).input_ids

    # step 3: run model and return emission matrix
    emissions, _ = get_emission(model, processor, audio_trim, tokens, device)

    char_durations, segments = get_duration_from_emission(
        emissions, tokens, transcript, blank_id=processor.tokenizer.pad_token_id
    )

    # step 5: generate pred
    preds = generate_pred(se, audio, labels, segments, char_durations)

    # step 6: save to file
    preds = json.dumps(preds, ensure_ascii=False)
    name = song_path.split("/")[-1].split(".")[0]
    with open(os.path.join(saved_path, f"{name}.json"), "w") as fp:
        fp.write(preds)


@cli.command("submission")
@click.option("--saved_path", default="/result/", show_default=True)
def main(saved_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2Processor.from_pretrained(
        "./pretrained/dragonSwing/wav2vec2-base-vietnamese/"
    )
    model = Wav2Vec2ForCTCV2(
        config=Wav2Vec2Config.from_pretrained(
            "./pretrained/dragonSwing/wav2vec2-base-vietnamese/"
        )
    )
    print("NEW VOCAB_SIZE: ", len(processor.tokenizer.get_vocab()))
    model.resize_lm_head(new_num_tokens=len(processor.tokenizer.get_vocab()))
    # load weight
    model.load_state_dict(
        torch.load(
            f"./checkpoints/dragonSwing/wav2vec2-base-vietnamese/checkpoint-5500/pytorch_model.bin",
            map_location="cpu",
        )
    )
    model.to(device)
    if device == "cuda":
        model.half()
    model.eval()
    # denoiser
    separator = get_denoiser(device=device)

    # convert model to device
    os.makedirs(saved_path, exist_ok=True)

    song_paths = glob.glob("./data/public_test/songs/*.wav")

    for song_path in tqdm(song_paths):
        run(
            song_path,
            model,
            processor,
            separator,
            device,
            saved_path=saved_path,
        )


if __name__ == "__main__":
    cli()
