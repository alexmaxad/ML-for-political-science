
import pandas as pd
from GloVe.weights import *
import warnings

warnings.filterwarnings("ignore")
from Axes.projection_functions import *
from Axes.axes_definition import *
from Axes.models import *

## **DataFrames**

dfs = []
for i in range(14):
    dfs.append(
        open_to_project(
            "/Users/alexandrequeant/Desktop/Travail TSE/data/sentence_embeddings/sentence_embeddings_201" + str(i) + ".csv",
            eval("201" + str(i)),
        )
    )

### On retire les journaux comportant moins de 50 articles dans une année

dfs[0] = dfs[0][dfs[0]["source"] != "DM"]
dfs[0] = dfs[0][dfs[0]["source"] != "TE"]
dfs[1] = dfs[1][dfs[1]["source"] != "TE"]
dfs[2] = dfs[2][dfs[2]["source"] != "TE"]
dfs[3] = dfs[3][dfs[3]["source"] != "TE"]
dfs[4] = dfs[4][dfs[4]["source"] != "TE"]
dfs[5] = dfs[5][dfs[5]["source"] != "TE"]

# PART II **Projections** $\rightarrow$ *[DONNÉES À ENTRER]*

## Projections par année


def both_cosines(df, pos_1, neg_1, pos_2, neg_2, model_words, model_sentences):
    df["cos axe 1"] = df["text"].apply(
        cosine_with_axis,
        pos_1=pos_1,
        neg_1=neg_1,
        model_words=model_words,
        model_sentences=model_sentences,
    )
    df["cos axe 2"] = df["text"].apply(
        cosine_with_axis,
        pos_1=pos_2,
        neg_1=neg_2,
        model_words=model_words,
        model_sentences=model_sentences,
    )
    return df


for i in range(14):
    dfs[i] = both_cosines(dfs[i], pos_1, neg_1, pos_2, neg_2, models_w[i], models_s[i])

df = pd.concat([dfs[i] for i in range(len(dfs))])

# PART III **BigTech dataframes**


def tostring(list):
    return str(list)


df_BT = df.copy()
df_BT.reset_index(drop=True, inplace=True)
df_BT["theme"] = df_BT["keywords"].apply(theme)
df_BT["theme"] = df_BT["theme"].apply(tostring)

df_BT_amazon = df_BT[df_BT["theme"].str.contains("amazon")]
df_BT_facebook = df_BT[df_BT["theme"].str.contains("facebook")]
df_BT_apple = df_BT[df_BT["theme"].str.contains("apple")]
df_BT_google = df_BT[df_BT["theme"].str.contains("google")]
df_BT_microsoft = df_BT[df_BT["theme"].str.contains("microsoft")]

df_BT_amazon["class"] = "am"
df_BT_facebook["class"] = "fb"
df_BT_apple["class"] = "ap"
df_BT_google["class"] = "go"
df_BT_microsoft["class"] = "mi"

df_BT = pd.concat(
    [df_BT_amazon, df_BT_facebook, df_BT_apple, df_BT_google, df_BT_microsoft]
)

# Saving dataframes

df.to_csv("data/current_dataframes/df", index=False)
df_BT.to_csv("data/current_dataframes/df_BT", index=False)