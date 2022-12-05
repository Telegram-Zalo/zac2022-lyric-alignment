import glob
import json
import logging
import os

import click
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import GroupKFold


@click.group()
def cli():
    logging.basicConfig(
        format="%(asctime)12s - %(levelname)s - %(message)s", level=logging.INFO
    )


def get_labels(path):
    with open(path, "rb") as fp:
        label = json.load(fp)

    word_durations_gts = []
    transcript = []
    for segment in label:
        l = segment["l"]
        for words in l:
            word_durations_gts.append([words["d"], words["s"], words["e"]])
            transcript.append(words["d"])

    transcript = " ".join(transcript)
    return transcript


@cli.command("train-valid-split")
@click.option("--labels_path", default="./data/train/labels", show_default=True)
@click.option("--train_size", default=0.8, show_default=True)
def split(labels_path, train_size):
    file = []
    text = []
    group = []
    for label_path in glob.glob(os.path.join(labels_path, "*.json")):
        transcript = get_labels(label_path)
        text.append(transcript)
        file.append(
            label_path.replace("labels", "songs_denoise").replace(".json", ".wav")
        )
        group.append(label_path.split("/")[-1].split("f")[0])

    file, text, group = shuffle(file, text, group, random_state=2022)

    csv = pd.DataFrame({"file": file, "text": text, "group": group})
    csv.to_csv(f"./data/train.csv", index=False, sep="\t")

    # public test
    file = []
    text = []
    for label_path in glob.glob(
        os.path.join("./data/public_test/json_lyrics", "*.json")
    ):
        transcript = get_labels(label_path)
        text.append(transcript)
        file.append(
            label_path.replace("json_lyrics", "songs_denoise").replace(".json", ".wav")
        )
    csv = pd.DataFrame({"file": file, "text": text})
    csv.to_csv(f"./data/public_test.csv", index=False, sep="\t")


if __name__ == "__main__":
    cli()
