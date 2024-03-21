
from GloVe.weights import *
import warnings

warnings.filterwarnings("ignore")
from Axes.projection_functions import *
from Axes.axes_definition import *

# PART I **word2vec models**

## Sentences

models_s = []
for i in range(14):
    models_s.append(
        txt_to_model_sentences(
            "/Users/alexandrequeant/Desktop/Travail TSE/data/sentence_embeddings/sentence_embeddings_201"
            + str(i)
            + ".txt"
        )
    )

## Words

models_w = []
for i in range(14):
    models_w.append(
        txt_to_model_words(
            "/Users/alexandrequeant/Desktop/Travail TSE/data/embeddings/embeddings_201"
            + str(i)
            + ".txt"
        )
    )