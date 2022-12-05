import glob
import json
import logging
import math
import multiprocessing
import os

import click
import librosa
import soundfile as sf
from pqdm.processes import pqdm
from functools import partial
from denoiser import get_denoiser, run_denoiser
import torch
from tqdm import tqdm


@click.group()
def cli():
    logging.basicConfig(
        format="%(asctime)12s - %(levelname)s - %(message)s", level=logging.INFO
    )


def resampling(file, target_sr=16000, res_type="kaiser_best"):
    speech_array, sampling_rate = sf.read(file)
    if len(speech_array.shape) == 2:
        speech_array = speech_array[..., 0]
    if sampling_rate != target_sr:
        speech_array = librosa.resample(
            speech_array, orig_sr=sampling_rate, target_sr=target_sr, res_type=res_type
        )

    sf.write(
        file.replace("songs_denoise", f"songs_denoise_{int(target_sr/1000)}k"),
        speech_array,
        samplerate=target_sr,
    )


def run_denoise(train_files):
    os.makedirs("./data/train/songs_denoise", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    separator = get_denoiser(device=device)
    for song_path in tqdm(train_files):
        audio, sr = run_denoiser(separator, song_path)
        sf.write(song_path.replace("songs", "songs_denoise"), data=audio, samplerate=sr)


@cli.command("resampling-data")
@click.option("--wavs_path", default="./data/train/songs/", show_default=True)
@click.option("--target_sr", default=16000, show_default=True)
@click.option("--res_type", default="kaiser_best", show_default=True)
def resampling_data(wavs_path, target_sr, res_type):
    train_file = glob.glob(os.path.join(wavs_path, "*.wav"))
    run_denoise(train_file)
    train_file = glob.glob(
        os.path.join(wavs_path.replace("songs", "songs_denoise"), "*.wav")
    )
    os.makedirs("./data/train/songs_denoise_16k", exist_ok=True)
    pqdm(
        train_file,
        partial(resampling, target_sr=target_sr, res_type=res_type),
        n_jobs=multiprocessing.cpu_count() - 1,
    )


if __name__ == "__main__":
    cli()
