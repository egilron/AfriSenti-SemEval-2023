from typing import Union, Any, Optional, TYPE_CHECKING, Iterable

from datasets import Dataset 
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer
import pandas as pd

def eval_dses(model, dses: Iterable[str]) -> Iterable[dict]:
    """Receives a list of Dataset test objects
    returns a list of dicts with metric as key.
    Does not keep track of any names for the test sets"""
    results: list = []
    for test_ds in dses:
        trainer = SetFitTrainer(
            model=model,
            eval_dataset=test_ds
        )
        results.append(trainer.evaluate())
    return results

if __name__ == "__main__":
    model = SetFitModel._from_pretrained("/home/egil/gits_wsl/afrisent/models/setfit_pt")
    df = pd.read_csv(
        "/home/egil/gits_wsl/afrisent/SubtaskA/train/formatted-train-data/pt/train.tsv" , sep='\t').dropna().rename(columns={"label":"label_text"})
    label_mapping = {'negative':0, 'neutral':1, 'positive':2}
    df["label"] = (df["label_text"]
        .map(label_mapping)
        .sample(frac=0.2, random_state=47))
    eval_ds = Dataset.from_pandas(df[["text", "label", "label_text"]], preserve_index=False)
    print(eval_dses(model, [eval_ds]))