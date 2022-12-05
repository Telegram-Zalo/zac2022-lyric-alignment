## load pretrained model
import torch
import numpy as np
import time

import torch.hub
import torch
import torchaudio as ta

from demucs import pretrained
from demucs.apply import apply_model
from demucs.audio import convert_audio, save_audio


torch.hub.set_dir("./pretrained/demucs_models/")


def get_denoiser(device="cuda"):
    # Use a pre-trained model
    separator = pretrained.get_model(name="mdx").models[3]
    separator.to(device)
    separator.eval()
    return separator


def run_denoiser(separator, path):
    mix, sr = ta.load(str(path))
    src_rate = separator.samplerate  # 44100
    mix = mix.cuda()
    ref = mix.mean(dim=0)  # mono mixture
    mix = (mix - ref.mean()) / ref.std()
    mix = convert_audio(mix, src_rate, separator.samplerate, separator.audio_channels)

    # Separate
    with torch.no_grad():
        estimates = apply_model(separator, mix[None], overlap=0.25)[0]  # defalut 0.25

    estimates = estimates * ref.std() + ref.mean()  # estimates * std + mean
    return estimates[3].cpu().numpy()[0, ...], sr
