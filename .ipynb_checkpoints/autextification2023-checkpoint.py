"""
https://zenodo.org/record/7956207
MGT detection and attribution in 5 domains, 2 languages, 6 models
domains: tweets, reviews, wikihow, news, legal
languages: english, spanish
models: bloom 1b7, 3b, 7b, gpt-3 babbage, gpt-3 curie and text-davinci-003
"""

import datasets
import pandas as pd

DETECTION_LABELS = {"human": 0, "generated": 1}
ATTRIBUTION_LABELS = {"bloom-1b7": 0, "bloom-3b": 1, "bloom-7b1": 2, "babbage": 3, "curie": 4, "text-davinci-003": 5}

ATTRIBUTION_ANON2LABEL = {"A": "bloom-1b7", "B": "bloom-3b", "C": "bloom-7b1", "D": "babbage",  "E": "curie", "F": "text-davinci-003"}

raw_urls = {
    "detection": {
        "train": "data/train/subtask_1/{language}/train.tsv",
        "test": "data/test/subtask_1/{language}/test.tsv",
    },
    "attribution": {
        "train": "data/train/subtask_2/{language}/train.tsv",
        "test": "data/test/subtask_2/{language}/test.tsv",
    },
}


class AuTexTification(datasets.GeneratorBasedBuilder):
    """The AuTexTification dataset prepared for MGT detection and family attribution"""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="detection_en",
            description="This part of the dataset is for English MGT detection.",
        ),
        datasets.BuilderConfig(
            name="attribution_en",
            description="This part of the dataset is for English model attribution.",
        ),
        datasets.BuilderConfig(
            name="detection_es",
            description="This part of the dataset is for Spanish MGT detection.",
        ),
        datasets.BuilderConfig(
            name="attribution_es",
            description="This part of the dataset is for Spanish model attribution.",
        ),
    ]

    def _info(self):
        names = (
            DETECTION_LABELS
            if "detection" in self.config.name
            else ATTRIBUTION_LABELS
        )

        self.features = {
            "id": datasets.Value("int64"),
            "prompt": datasets.Value("string"),
            "text": datasets.Value("string"),
            "label": datasets.features.ClassLabel(
                names=list(names.keys())
            ),
            "domain": datasets.Value("string"),
        }
        if "detection" in self.config.name:
            self.features["model"] = datasets.Value("string")

        return datasets.DatasetInfo(
            description="AuTexTification dataset prepared for MGT detection and family attribution",
            features=datasets.Features(self.features),
        )

    def _split_generators(self, dl_manager):
        task, language = self.config.name.split("_")
        selected_urls = {split: url.format(language=language) for split, url in raw_urls[task].items()}
        
        paths = dl_manager.download_and_extract(selected_urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"path": paths["train"]}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"path": paths["test"]}
            ),
        ]

    def _generate_examples(self, path):
        data = pd.read_csv(path, sep="\t", usecols=self.features.keys())

        # de-anonymize
        if "detection" in self.config.name:
            data["model"] = data["model"].apply(lambda x: ATTRIBUTION_ANON2LABEL.get(x, x))
        else:
            data["label"] = data["label"].apply(lambda x: ATTRIBUTION_ANON2LABEL.get(x, x))

        for i in range(data.shape[0]):
            yield i, data.iloc[i].to_dict()
