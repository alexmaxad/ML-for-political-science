from GloVe.weights import *
import warnings
import os

warnings.filterwarnings("ignore")
from Axes.projection_functions import *
from Axes.axes_definition import *

os.chdir("../")
print(os.getcwd())


# PART I **word2vec models**

## Sentences

models_s = []
for i in range(14):
    models_s.append(
        txt_to_model_sentences(
            "data/sentence_embeddings/sentence_embeddings_201"
            + str(i)
            + ".txt"
        )
    )

## Words

models_w = []
for i in range(14):
    models_w.append(
        txt_to_model_words(
            "data/embeddings/embeddings_201"
            + str(i)
            + ".txt"
        )
    )

os.chdir(r"src")