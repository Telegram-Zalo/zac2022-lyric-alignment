import re

from dataloaders import remove_special_characters
import pandas as pd
import json
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Processor
from models import Wav2Vec2ForCTCV2


if __name__ == "__main__":
    # download and save pretrained
    processor = Wav2Vec2Processor.from_pretrained(
        "dragonSwing/wav2vec2-base-vietnamese"
    )
    model = Wav2Vec2ForCTCV2.from_pretrained("dragonSwing/wav2vec2-base-vietnamese")

    processor.save_pretrained("./pretrained/dragonSwing/wav2vec2-base-vietnamese/")
    model.save_pretrained("./pretrained/dragonSwing/wav2vec2-base-vietnamese/")

    train = pd.read_csv("./data/train.csv", sep="\t")
    public_test = pd.read_csv("./data/public_test.csv", sep="\t")

    all_data = pd.concat([train, public_test])
    texts = all_data["text"].map(lambda x: remove_special_characters(x)).tolist()
    vocab_list = []
    for text in texts:
        vocab_list.extend(list(text))
    vocab_list = list(set(vocab_list))

    old_vocab = json.load(
        open("./pretrained/dragonSwing/wav2vec2-base-vietnamese/vocab.json", "rb")
    )
    new_vocab = list(set(vocab_list) - set(old_vocab.keys()) - set([" "]))

    len_old_vocab = len(old_vocab)

    for k in range(0, len(new_vocab)):
        old_vocab[new_vocab[k]] = k + len_old_vocab

    vocab_dict = json.dumps(old_vocab, ensure_ascii=False)
    with open(
        "./pretrained/dragonSwing/wav2vec2-base-vietnamese/vocab.json", "w"
    ) as fp:
        fp.write(vocab_dict)
