library(tidyverse)
library(RJSONIO)
library(reticulate)

# Define the base path for your files for easier reference
base_path <- "/Users/alexandrequeant/Desktop/Travail-TSE/data/without parliament"

# List of years or identifiers for your files
years <- c(2010, 2011, 2012, 2013, 2014, 2014, 2016, 2017, 2018, 2019, 20110, 20111, 20112, 20113) # Example years, adjust according to your actual files

# Loop through each year or identifier
for (year in years) {
  datafile <- paste0(base_path, "/FinalDataframes/FilteredFinalDataFrame_", year, "_WP.csv")
  
  # Reading the CSV file for the current year
  data.source <- read.csv(datafile)
  
  # Transforming the 'text' column by evaluating its content as Python code
  test <- data.source %>%
    rowwise() %>%
    mutate(text = list(reticulate::py_eval(text)))
  
  # Extracting the vocabulary
  vocab <- unlist(test$text)
  vocab <- c(vocab)
  
  # Converting the vocabulary to JSON
  xportJson <- toJSON(vocab)
  
  # Defining the path for the output JSON file
  output_path <- paste0(base_path, "/words/Finalwords_", year, "_WP.json")
  
  # Writing the JSON file
  write(xportJson, output_path)
}