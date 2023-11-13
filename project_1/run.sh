#!/bin/bash

# Render pre-processing RMarkdown file to PDF
Rscript -e "rmarkdown::render('src/preproc.Rmd', 'pdf_document', 'documentation_1_preprocessing', 'out')"

# Train the model using Python script and render to PDF using Jupyter
jupyter nbconvert --execute ./src/train_model.ipynb --output ../out/documentation_2_train_model.pdf --to 'pdf'
