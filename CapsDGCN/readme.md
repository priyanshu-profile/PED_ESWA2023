The folder contains pytorch implementation of the paper "A multi-task learning framework for politeness and emotion detection in dialogues for mental health counselling and legal aid"

This model simultaneously predicts politeness label and emotion labels of a utterance in a conversation using a unified multi-task framework CapsDGCN.

## Dependencies

- python>=3.6
- torch==1.9.1
- transformers==3.0.2

# Execute the model using the following commands

python main.py --dataset poem 
python main.py --dataset dailydialog

# Data Pre-processing

## Requirement

'StanfordCoreNLP' (version: 3.9.2) is required to obtain the dependency trees for the dataset. 
Reference: https://towardsdatascience.com/natural-language-processing-using-stanfords-corenlp-d9e64c1e1024

To obtain the dependency parse tree, run python create_dep_info.py --data_dir [path_to_data_file]


