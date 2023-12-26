# ADL-2023: The Effectivess of Different Model Types in Detecting Sarcasm

**Seah Ying Xiang - T12902136** \
**Dao Lan Ha - B09701145** \
**Paul Ong - T12902130**

### This repository hosts the project files and scripts for training our models detect sarcasm in Reddit comments.

### Presentation link: https://youtu.be/roYIMcl7o5c

## Training the models

### Run.sh file

```
dataset="daniel2588/sarcasm"
python run_classification.py \
 --model_name_or_path xlnet-base-cased \
 --dataset_name ${dataset} \
 --shuffle_train_dataset \
 --shuffle_seed 12902136 \
 --metric_name accuracy \
 --text_column_name "comment" \
 --text_column_delimiter "\n" \
 --label_column_name label \
 --seed 42 \
 --do_train \
 --do_eval \
 --do_predict \
 --max_seq_length 512 \
 --per_device_train_batch_size 8 \
 --per_device_eval_batch_size 8 \
 --gradient_accumulation_steps 8 \
 --load_best_model_at_end \
 --metric_for_best_model accuracy \
 --learning_rate 2e-5 \
 --max_steps 10000 \
 --max_eval_samples 3000 \
 --overwrite_output_dir \
 --output_dir ./out/xlnet-base-cased/ \
 --save_steps 1000 \
 --eval_steps 1000 \
 --evaluation_strategy steps \
 --validation_split 0.2 \
 --test_split 0.1
```

### Adjustments for different models

For each different model the Run.sh model will have to be adjusted

```shell
--model_name_or_path roberta-base \
#or
--model_name_or_path bert-base-uncased \
#or
--model_name_or_path xlnet-base-cased \
```

### Running the Training Process

```shell
bash run.sh
```

## run_classification.py

based on the `run_classification.py` from huggingface transformers library:https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification

## Reproducing the Graphs

### Running graph.ipynb

Place the `trainer_states file` and the `graph.ipynb` in the same directory.

`runall` the cells in the graph.ipynb file to get all the graphs.

If there is a need to use the graph.ipynb file for new eval_data.

### trainer_states

The trainer*states file is a collection of the `trainer_state.json` files from the outputs of training each model and renamed `(modelname)*(combined/notcombined).json`.
