# %%
# !pip install --upgrade pip
# !pip install sentencepiece
# !pip install datasets
# !pip install transformers

# %% [markdown]
# # Fine-tuning XLM-T
# 
# This notebook describes a simple case of finetuning. You can finetune either the `XLM-T` language model, or XLM-T sentiment, which has already been fine-tuned on sentiment analysis data, in 8 languages (this could be useful to do sentiment transfer learning on new languages).,
# 
# This notebook was modified from https://huggingface.co/transformers/custom_datasets.html

# %%
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import sys, os, json
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from datetime import datetime
from pathlib import Path
from shutil import rmtree
import datasets
from datasets import load_metric


# TODO: Lage individuelle savefolders for best model?
# Shared parameters

BATCH_SIZE = 32
NUM_LABELS = 3
SAVE_DIR = '/fp/projects01/ec30/egilron/tweet-afri/submit' #afro-xlmr-mini'
# MODEL = os.path.join(SAVE_DIR, "final")#"cardiffnlp/twitter-xlm-roberta-base" # use this to finetune the language model
# MODEL = "Davlan/afro-xlmr-mini" #"sentence-transformers/paraphrase-mpnet-base-v2" #"cardiffnlp/twitter-xlm-roberta-base-sentiment" # use this to finetune the sentiment classifier
MAX_TRAINING_EXAMPLES = -1 # set this to -1 if you want to use the whole training set


TIMESTAMP = datetime.now().strftime("%m%d%H%M")

LANG_GROUPS = {'Volta-Congo': ['ts', 'twi', 'kr', 'yo', 'ig', 'sw'],
 'Afro-Asiatic-Semitic': ['tg', 'am', 'dz', 'ma'],
 'Creole': ['pcm', 'pt'], 
 "multilingual": ['ts', 'twi', 'kr', 'yo', 'ig', 'sw', 'tg', 'am', 'dz', 'ma','pcm', 'pt' ]} 
LANG_GROUPS = { "multilingual": ['ts', 'twi', 'kr', 'yo', 'ig', 'sw', 'tg', 'am', 'dz', 'ma','pcm', 'pt', 'ha' ]} # Catch up what failed
LANG_GROUPS = { "multilingual": ['ha']} # I had forgotten to add ha which has no group for training
## Logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logs_folder = "logs"
ts = TIMESTAMP
filehandler = logging.FileHandler(os.path.join(logs_folder, f"{ts}.log"), "a")
logger.addHandler(filehandler)

log_level = "info"
transformers.utils.logging.set_verbosity_info()
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()



class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# %% Preprocessing
def ds_dict(train_path="", dev_path="", pred_path=""):
  label2id = {'negative': 0, 'neutral': 1, 'positive': 2}
  train_df = pd.read_csv(train_path, sep="\t")
  val_df = pd.read_csv(dev_path, sep="\t")
  dataset_dict = {}
  dataset_dict["train"] = {}
  dataset_dict["train"]["text"] = train_df["tweet"].to_list()
  dataset_dict["train"]["labels"] = [label2id[l] for l in train_df["label"]]
  dataset_dict["val"] = {}
  dataset_dict["val"]["text"] = val_df["tweet"].to_list()
  dataset_dict["val"]["labels"] = [label2id[l] for l in val_df["label"]]
  if pred_path == "":
    dataset_dict["test"] = dataset_dict["val"].copy()
  else:
    pred_df = pd.read_csv(pred_path, sep="\t")
    dataset_dict["test"]["text"] = pred_df["tweet"].to_list()


  assert len(dataset_dict["val"]["text"]) == len(dataset_dict["val"]["labels"])
  return dataset_dict


def df_dicts_concat(train_paths = [], dev_paths = []):
  """Concatenate a number of language data into one ds_dict"""
  assert len(train_paths) == len(dev_paths)
  dataset_dict = {"train":{"text":[], "labels":[]},
                  "val":{"text":[], "labels":[]}}
  for tp, dp in zip(train_paths, dev_paths):
    dsd = ds_dict(tp, dp)
    dataset_dict["train"]["text"] += dsd["train"]["text"]
    dataset_dict["train"]["labels"] += dsd["train"]["labels"] 
    dataset_dict["val"]["text"] += dsd["val"]["text"] 
    dataset_dict["val"]["labels"] += dsd["val"]["labels"]
    dataset_dict["test"] = dataset_dict["val"].copy()
  
  assert len(dataset_dict["train"]["text"]) == len(dataset_dict["train"]["labels"])
  assert len(dataset_dict["val"]["text"]) == len(dataset_dict["val"]["labels"])
  return dataset_dict

def trainme(dataset_dict,model="cardiffnlp/twitter-xlm-roberta-base-sentiment",  tokenizer = None, finetune = True,  training_args = None):
  if not tokenizer:
    tokenizer = model
  tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True,  model_max_length=512) # max_length=512,,truncation=True, padding=True
  train_encodings = tokenizer(dataset_dict['train']['text'], truncation=True, padding=True)
  val_encodings = tokenizer(dataset_dict['val']['text'], truncation=True, padding=True)
  test_encodings = tokenizer(dataset_dict['test']['text'], truncation=True, padding=True)
  train_dataset = MyDataset(train_encodings, dataset_dict['train']['labels'])
  val_dataset = MyDataset(val_encodings, dataset_dict['val']['labels'])
  test_dataset = MyDataset(test_encodings, dataset_dict['test']['labels'])
  model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=NUM_LABELS,   max_length=512)
  metric = load_metric( 'f1') # , 'precision', 'recall', process_id=1

  if not training_args:
    training_args = TrainingArguments(
      output_dir=SAVE_DIR,                   # output directory
      num_train_epochs=14,                  # total number of training epochs
      per_device_train_batch_size=BATCH_SIZE,   # batch size per device during training
      per_device_eval_batch_size=BATCH_SIZE,    # batch size for evaluation
      learning_rate=2e-5,
      warmup_steps=100,                         # number of warmup steps for learning rate scheduler
      weight_decay=0.01,                        # strength of weight decay
      logging_dir='./logs',                     # directory for storing logs #logging_steps=5000,                         # when to print log
      load_best_model_at_end=True,              # load or not best model at the end
      save_strategy="epoch", #"epoch",                      # please select one of ['no', 'steps', 'epoch']
      evaluation_strategy="epoch", #  #eval_steps = 10, #eval_accumulation_steps = 1,
      save_total_limit=1,
      disable_tqdm=True,
      log_level = log_level,
      logging_strategy = "epoch",
      metric_for_best_model = "eval_f1"

    )


  def compute_metrics(eval_pred):
      predictions, labels = eval_pred
      predictions = np.argmax(predictions, axis=1)
      return metric.compute(predictions=predictions, references=labels, average="macro")


  trainer = Trainer(
      model=model,                              # the instantiated 🤗 Transformers model to be trained
      args=training_args,                       # training arguments, defined above
      train_dataset=train_dataset,              # training dataset
      eval_dataset=val_dataset,                  # evaluation dataset
      compute_metrics=compute_metrics
  )
  if finetune:
    train_results = trainer.train()
  else:
    print("Deprecated, returning untrained model")



  return trainer




def train_individual(model= "cardiffnlp/twitter-xlm-roberta-base-sentiment", 
                     train_folder = Path("datasets/train_final/"),
                     eval_folder = Path("datasets/dev_final/"),
                     dev_template = "_dev_gold_label.tsv" , 
                     training_args = None

                     ):
 
  results_path = f"results/{TIMESTAMP}_tr-individual_test_individual.json"
  results = []
  zeroshots = []
  train_eval_paths = []
  tsv_data = [{"lang":dp.stem.split("_")[0], "train_tsv":dp, "eval_tsv":None }for dp in train_folder.iterdir() if dp.suffix == ".tsv"]
  for data in tsv_data:
    eval_candidate = Path(eval_folder, data["lang"]+dev_template)
    if eval_candidate.is_file():
      data["eval_tsv"]= eval_candidate
    else:
      print("Language", data["lang"], "does not have a dev data file at", str(eval_candidate))

  for lang_data in tsv_data:
    language = lang_data["lang"]
    print("Starting individual training:", language)
    dataset_dict = ds_dict(train_path=lang_data["train_tsv"], dev_path=lang_data["eval_tsv"])
    trainer = trainme( dataset_dict, finetune = True, training_args = training_args)
    lang_f1 = evaulate_one(trainer=trainer, dataset_dict=dataset_dict)
    results.append({"timestamp": TIMESTAMP,"train":language , "test":language, "f1":lang_f1 })

    with open (results_path, "w") as wf:
      json.dump(results, wf)
    logger.warning(f"{language} Train_individual:{results}")
  return results
    

    
def evaulate_one(trainer= None, 
        tokenizer =  AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment"),
        dataset_dict = None #WIth test data (Ususally equals dev data)
        ):
  """Evaluate a language on a model, based on the eval_path sent
  return the f1
  """
  train_encodings = tokenizer(dataset_dict['train']['text'], truncation=True, padding=True)
  val_encodings = tokenizer(dataset_dict['val']['text'], truncation=True, padding=True)
  test_encodings = tokenizer(dataset_dict['test']['text'], truncation=True, padding=True)
  train_dataset = MyDataset(train_encodings, dataset_dict['train']['labels'])
  val_dataset = MyDataset(val_encodings, dataset_dict['val']['labels'])
  test_dataset = MyDataset(test_encodings, dataset_dict['test']['labels'])

  eval_result = trainer.evaluate(eval_dataset = test_dataset)
      #       {'eval_loss': 1.4181029796600342,
      #  'eval_f1': 0.5824834753269498,
      #  'eval_runtime': 63.585,
      #  'eval_samples_per_second': 6.102,
      #  'eval_steps_per_second': 0.771} 
  return eval_result["eval_f1"]


def train_lg_groups(model= "cardiffnlp/twitter-xlm-roberta-base-sentiment", 
                     train_folder = Path("datasets/group_final"),
                     eval_folder = Path("datasets/dev_final/"),
                     dev_template = "_dev_gold_label.tsv" , 
                     training_args = None
                     ):
  """ The language group belonging has been found in Ethnologue / Wikipedia and we now train the language group together, and evaluate on each of the languages in it
In stead of using this, create a main that uses train_individual for both folders.
New 16. feb: We have now prepared the concatenated datasets before getting here.
  """
  results_path = f"results/{TIMESTAMP}_tr-group_test_individual.json"
  groups = LANG_GROUPS
  # groups = {  'Creole': ['pcm', 'pt'] } # 
  results = []

  for lg_group, eval_langs in groups.items():
    train_path = Path(train_folder, "{}_train.tsv".format(lg_group))
    dev_path = Path(train_folder, "{}_dev.tsv".format(lg_group))
    # Train with group train and dev data
    dataset_dict = ds_dict(train_path=train_path, dev_path=dev_path)
    trainer = trainme( dataset_dict, finetune = True)
    results.append({"timestamp": TIMESTAMP,"train":lg_group , 
          "test":lg_group, "f1": trainer.evaluate()["eval_f1"]})

    for eval_lang in  eval_langs: # Evaluate individual languages
      print(f"\nEvaluating {eval_lang} on a model trained on {lg_group}:")
      eval_path = Path(eval_folder, eval_lang+dev_template) 
      dataset_dict = ds_dict(train_path=eval_path, dev_path=eval_path)
      lang_f1 = evaulate_one(trainer=trainer, dataset_dict=dataset_dict)
      results.append({"timestamp": TIMESTAMP,"train":lg_group , "test":eval_lang, "f1":lang_f1 })

      with open (results_path, "w") as wf:
        json.dump(results, wf)
      logger.warning(f"{lg_group} Test:{eval_lang} F1:{lang_f1}")
  return results
 




def final_train_A():
  # model = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
  # train_langs = [l.stem.split("_")[0] for l in Path("afrisent-semeval-2023/SubtaskA/train").iterdir()]
  train_langs = {'dz':7, 'am':5, 'yo':6, 'twi':4, 'pcm':6, 'pt':7, 'ma':7, 'ha':4, 'ig':6, 'ts':5, 'kr':7}
  for lang in train_langs:
    if not lang in Path(SAVE_DIR).iterdir(): # Has been successfully completed
      print("Finetuning", lang)
      t_split = f"afrisent-semeval-2023/SubtaskA/train/{lang}_train.tsv"
      d_split = f"afrisent-semeval-2023/SubtaskA/dev_gold/{lang}_dev_gold_label.tsv"

      assert Path(t_split).is_file() and Path(d_split).is_file()
      dataset_dict = df_dicts_concat([t_split, d_split] , [d_split, d_split] ) # Need same length
      trainer = trainme( dataset_dict, finetune = True, epochs = train_langs[lang])
      trainer.save_model(Path(SAVE_DIR, lang)) # save best model

def final_train_B():
  t_split = "afrisent-semeval-2023/SubtaskB/multilingual_train.tsv"
  d_split = "afrisent-semeval-2023/SubtaskB/dev_gold/multilingual_dev_gold_label.tsv"
  lang = "multilingual"
  assert Path(t_split).is_file() and Path(d_split).is_file()
  dataset_dict = df_dicts_concat([t_split, d_split] , [d_split, d_split] ) # Need same length
  trainer = trainme( dataset_dict, finetune = True, epochs = 7)
  trainer.save_model(Path(SAVE_DIR, lang)) # save best model
    

if __name__ == "__main__":
    """16. feb: send the two folders from here"""
    if len(sys.argv) == 2:
        functions = {"train_lg_groups": train_lg_groups, 
        "train_individual": train_individual,
        "train_lg_groups": train_lg_groups}
        print(sys.argv[0], "running function",sys.argv[1] )
        print(functions[sys.argv[1]]()) # Only default values (set in function) by now
    else: 
        # "local run"    
        training_args = TrainingArguments(
            output_dir=SAVE_DIR,                   # output directory
            num_train_epochs=14,                  # total number of training epochs
            per_device_train_batch_size=BATCH_SIZE,   # batch size per device during training
            per_device_eval_batch_size=BATCH_SIZE,    # batch size for evaluation
            learning_rate=2e-5,
            warmup_steps=100,                         # number of warmup steps for learning rate scheduler
            weight_decay=0.01,                        # strength of weight decay
            logging_dir='./logs',                     # directory for storing logs #logging_steps=5000,                         # when to print log
            load_best_model_at_end=True,              # load or not best model at the end
            save_strategy="epoch",                      # please select one of ['no', 'steps', 'epoch']
            evaluation_strategy="epoch", #  #eval_steps = 10, #eval_accumulation_steps = 1,
            save_total_limit=1,
            disable_tqdm=True,
            log_level = log_level,
            logging_strategy = "epoch", 
            metric_for_best_model = "eval_f1"
        )
        results = train_individual(model= "cardiffnlp/twitter-xlm-roberta-base-sentiment", 
                            train_folder = Path("datasets/train_final/"),
                            eval_folder = Path("datasets/dev_final/"),
                            dev_template = "_dev_gold_label.tsv" ,
                            training_args = training_args
                            )
        print(results)