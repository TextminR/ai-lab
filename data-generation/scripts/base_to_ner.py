import json

import nltk
from datasets import load_dataset
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)

dataset = load_dataset("textminr/ner_extended")
ner_tag_to_id = {"O": 0, "AUTHOR": 1, "DATE": 2}


def convert_row(row):
    prompt = row["prompt"]
    response = json.loads(row["response"])
    author = word_tokenize(response["author"])
    date = response["date"]

    tokens = word_tokenize(prompt)
    ner_ids = []
    ner_tags = []

    counter_author = 0
    counter_date = 0

    for token in tokens:
        if token in author and counter_author < len(author):
            ner_ids.append(ner_tag_to_id["AUTHOR"])
            ner_tags.append("AUTHOR")
            counter_author += 1
        elif token in date and counter_date < 1:
            ner_ids.append(ner_tag_to_id["DATE"])
            ner_tags.append("DATE")
            counter_date += 1
        else:
            ner_ids.append(ner_tag_to_id["O"])
            ner_tags.append("O")

    row["tokens"] = tokens
    row["ner_ids"] = ner_ids
    row["ner_tags"] = ner_tags

    return row


dataset_tokenized = dataset.map(convert_row).remove_columns(["prompt", "response"])
dataset_tokenized.push_to_hub("textminr/ner_tokenized")
