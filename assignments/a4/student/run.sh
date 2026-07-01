#!/bin/bash
set -e

python3 src/run.py finetune vanilla wiki.txt \
        --writing_params_path vanilla.model.params \
        --finetune_corpus_path birth_places_train.tsv

# python3 src/run.py evaluate vanilla wiki.txt  \
#         --reading_params_path vanilla.model.params \
#         --eval_corpus_path birth_dev.tsv \
#         --outputs_path vanilla.nopretrain.dev.predictions

# python3 src/run.py evaluate vanilla wiki.txt  \
#         --reading_params_path vanilla.model.params \
#         --eval_corpus_path birth_test_inputs.tsv \
#         --outputs_path vanilla.nopretrain.test.predictions


# python3 src/dataset.py charcorruption