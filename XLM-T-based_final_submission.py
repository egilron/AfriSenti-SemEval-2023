
# %%

from transformers import pipeline
import tqdm
import os, json, pathlib, toml, pickle, csv
import numpy as np
import pandas as pd
from pathlib import Path
from src.pp_data import *

# %% [markdown]
# ## Parameters




models_root = "/fp/projects01/ec30/egilron/tweet-afri/submit"

def single_lang():
    langs = ['dz', 'am', 'yo', 'twi', 'pcm', 'pt', 'ma', 'ha', 'ig', 'ts', 'kr']

    source_texts = {}
    for lang in langs:
        model_path = Path(models_root, lang)
        tokenizer_path = "/fp/projects01/ec30/egilron/tweet-afri/development"
        predict_path = f"afrisent-semeval-2023/SubtaskA/test/{lang}_test_participants.tsv"
        target_path = f"results/predictions_3001/{lang}_predictions_A.json"
        if not Path(target_path).is_file(): # Not redo those completed
            df_predict = preprocess_td(str(predict_path))
            preds = [] # one dict per tweet
            model = pipeline("text-classification", 
                model=str(model_path),
                tokenizer =tokenizer_path)
            preds = [] # one dict per tweet
            for id, row in df_predict.iterrows():
                try:
                    label = model(row["text"])[0]["label"] 
                except:
                    label = "neutral"
                    print("Prediction failed:", id)

                preds.append({"ID": id, "label": label})
            
            fieldnames = list(preds[0].keys())
            with open(target_path, 'w', newline='') as wf:
                writer = csv.DictWriter(wf, fieldnames=fieldnames,  delimiter="\t")
                writer.writeheader()
                writer.writerows(preds)



def multiling():
    # The test sets that will be predicted with the multilingual model
    test_langs = {"multilingual": "afrisent-semeval-2023/SubtaskB/test/multilingual_test_participants.tsv",
                    "sw": "afrisent-semeval-2023/SubtaskA/test/sw_test_participants.tsv"
    }

    source_texts = {}
    
    for lang, predict_path in test_langs.items():
        print("starting predictions for", lang)
        task = "B" if lang == "multilingual" else "A"
        model_path = Path(models_root, "multilingual")
        tokenizer_path = "/fp/projects01/ec30/egilron/tweet-afri/development"
        target_path = f"results/predictions_3001/{lang}_predictions_{task}.json"
        if not Path(target_path).is_file(): # Not redo those completed
            df_predict = preprocess_td(str(predict_path))
            preds = [] # one dict per tweet
            model = pipeline("text-classification", 
                model=str(model_path),
                tokenizer =tokenizer_path)
            preds = [] # one dict per tweet
            for id, row in df_predict.iterrows():
                try:
                    label = model(row["text"])[0]["label"] 
                except:
                    label = "neutral"
                    print("Prediction failed:", id)

                preds.append({"ID": id, "label": label})
            
            fieldnames = list(preds[0].keys())
            with open(target_path, 'w', newline='') as wf:
                writer = csv.DictWriter(wf, fieldnames=fieldnames,  delimiter="\t")
                writer.writeheader()
                writer.writerows(preds)

multiling()

exit()






predict_paths
# %%
# Conversion
tokenizers = {"xlmt_sent": "xlmt_sent/final",
            "xlmt": "xlmt/final"}

default_model = "xlmt_sent/checkpoint-13146"

best_models = {
'datasets/data_jan23/TaskA/dev/dz_dev.tsv': "xlmt_sent/checkpoint-15024",
 'datasets/data_jan23/TaskA/dev/ha_dev.tsv': "xlmt/final",
 'datasets/data_jan23/TaskA/dev/kr_dev.tsv': "xlmt_sent/checkpoint-15024",
'datasets/data_jan23/TaskA/dev/pcm_dev.tsv': "xlmt/final",
'datasets/data_jan23/TaskA/dev/pt_dev.tsv': "xlmt/final",
'datasets/data_jan23/TaskA/dev/sw_dev.tsv': "xlmt_sent/checkpoint-11268",
'datasets/data_jan23/TaskA/dev/ts_dev.tsv': "xlmt_sent/checkpoint-11268",
'datasets/data_jan23/TaskA/dev/twi_dev.tsv': "xlmt_sent/checkpoint-11268",
'datasets/data_jan23/TaskA/dev/yo_dev.tsv': "xlmt/checkpoint-9390",
'datasets/data_jan23/TaskC/ts_dev.tsv': "xlmt/checkpoint-9390",
 'datasets/data_jan23/TaskC/tg_dev.tsv': "xlmt/checkpoint-9390",
 'datasets/data_jan23/TaskC/or_dev.tsv': "xlmt/checkpoint-9390"
}




# Development
# Remaining task
predict_paths = [p for p in predict_paths if "multiling" in p]
print(len(predict_paths), "to process")   

model = None
MODEL_ROOT = Path("/home/egil/gits_wsl/afrisent/models")
SAVE_FOLDER = Path("results/predictions_1601")
for path_key in predict_paths:
    sub_path = Path(path_key.replace("datasets/data_jan23/", ""))
    task = sub_path.parts[0]
    tsv_path = SAVE_FOLDER.joinpath(sub_path)
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    model_subpath = best_models.get(path_key, default_model)
    model_path = str(MODEL_ROOT.joinpath(Path(model_subpath)))
    tokenizer_subpath = tokenizers[model_subpath.split("/")[0]]
    tokenizer_path = str(MODEL_ROOT.joinpath(Path(tokenizer_subpath)))
    model = pipeline("text-classification", 
            model=model_path,
            tokenizer =tokenizer_path)

    preds = [] # one dict per tweet
    for id, row in tqdm.tqdm(all_dfs[path_key].iterrows()):
        try:
            label = model(row["text"])[0]["label"] 
        except:
            label = "neutral"
            print("Prediction failed:", id)

        preds.append({"ID": id, "label": label})

    fieldnames = list(preds[0].keys())
    
    with open(tsv_path, 'w', newline='') as wf:
        writer = csv.DictWriter(wf, fieldnames=fieldnames,  delimiter="\t")
        writer.writeheader()
        writer.writerows(preds)

   
# %%
# Convert and zip
from zipfile import ZipFile
task_fs = {"A": Path("results/predictions_1601/TaskA/dev"),
            "B": Path("results/predictions_1601/TaskB"),
            "C": Path("results/predictions_1601/TaskC")
}
l_convert = {"LABEL_0":"negative", 
             "LABEL_1": "neutral",
             "LABEL_2": "positive"}
            
for task, task_f in task_fs.items():
    for file_p in [f for f in task_f.iterdir() if f.suffix == ".tsv"]:
        # Recreate filename according to submission rules
        language = file_p.stem.split("_")[0]
        assert language != "dev"
        new_f_name = f"pred_{language}.tsv" # 
        zip_name = f"results/predictions_1601/{task}_{language}.zip"
        with file_p.open() as rf:
            tsv = rf.read().strip()
            
            # Got wrong label scheme in task C
            for old, new in l_convert.items():
                tsv= tsv.replace("\t"+old, "\t"+new)
                if task == "C":
                    print(old, tsv.count(old), tsv.count("\t"+old), tsv.count("\t"+new))

            with ZipFile(zip_name, "w") as zf:
                zf.writestr(new_f_name, tsv)

            with ZipFile(zip_name) as zf:
                print(zip_name, zf.namelist())





# %%
# Scrap
# test_text = all_dfs['datasets/data_jan23/TaskA/dev/am_dev.tsv']["text"].to_list()
# results = []
# for t in tqdm.tqdm (test_text):
#     pred = model(t)
#     results.append(pred)
all_dfs[path_key].tail(3)

