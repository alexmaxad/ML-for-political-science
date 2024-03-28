library(tidyverse)
library(RJSONIO)
datafile <- '/Users/alexandrequeant/Desktop/Stage Toulouse /Stage_TSE_2023/Stage_TSE_2023/data/FinalDataframes/FilteredFinalDataFrame_20113.csv'

data.source <- read.csv(datafile)

test <-  data.source %>%
          rowwise() %>%
          mutate(text=list(reticulate::py_eval(text)))


vocab <- unlist(test$text)

vocab <- c(vocab)

length(vocab)

xportJson <- toJSON(vocab)

write(xportJson, "/Users/alexandrequeant/Desktop/Stage Toulouse /Stage_TSE_2023/Stage_TSE_2023/data/words/Finalwords_20113.json")

