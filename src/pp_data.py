import pandas as pd
import json

LABEL_MAPPING = {'negative':0, 'neutral':1, 'positive':2}
COLUMN_MAPPING = {"tweet": "text",	"label": "label_text" } # Orig ID	tweet	label

def preprocess_td(filepath :str = "dummy" ):
    """ Preprocess training data
    get one filepath, return one df"""
    # print(filepath)
    df = pd.read_csv(filepath, sep='\t', header=0).dropna()# Moved preprcessing out of this du to hassle
    df = df.rename(columns = COLUMN_MAPPING).set_index("ID")
    # print(filepath, df.columns)
    # print( df.head(1))
    if "label_text" in df.columns:
        df["label"] = df["label_text"].map(LABEL_MAPPING)
    return df

def df_to_json(df:pd.DataFrame, save_path):
    """Save a dataframe to disk with consistent parameters"""
    jf = df.to_json(orient='records', force_ascii=False)
    with open(save_path, "w", encoding="utf-8") as wf:
        wf.write(jf )

def df_from_json(read_path:str) :
    """Read DataFrame from Json with consistent parameters"""
    return pd.read_json(read_path, orient="records", encoding="utf-8")    
