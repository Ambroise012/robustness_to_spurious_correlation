
# I- PLM : BERT / Task : MNLI - HANS / Shallow Model : BERT 
## 1. Download MNLI data
```bash
$ sh getdata.sh mnli && export MNLI_PATH=mnli/
```
## 2. First Fine Tuning on MNLI
```bash
$ python spurious_bert_mnli.py train_mnli_bert_base --output_dir mnli_bert_base/
```
## 3. Second Fine Tuning on Spurious Correlation
### Forgettable examples
```bash
# extract forgettables from bert model
$ python spurious_bert_mnli.py extract_hard_examples mnli_bow/ -train_path mnli/MNLI/train.tsv --task mnli 

$ nohup python spurious_bert_mnli.py finetune_hard_examples mnli_bert_base/checkpoint-last/ mnli_bert_base_fbert_forget_ex/ --hard_path mnli_bert_base/hard_examples.pkl &>output_log/out_fine_tune_bert_forget_ex.log
```

### Important examples
```bash
# extract important samples from bert model
$ python spurious_bert_mnli.py extract_important_examples mnli_bert_base/ --train_path mnli/MNLI/train.tsv --task mnli

$ nohup python spurious_bert_mnli.py finetune_hard_examples mnli_bert_base/checkpoint-last/ mnli_bert_base_fbert_important_samples/ --hard_path mnli_bert_base/important_examples.pkl &>output_log/out_fine_tune_bert_important_samples.log
```

### Important samples + Forgettable examples
```bash
# extract forgettables and important samples from bert model
$ python spurious_bert_mnli.py extract_important_forget_ex mnli_bert_base/ --train_path mnli/MNLI/train.tsv --task mnli

# fine-tune the model on bert forgettables combine with important samples
$ nohup python spurious_bert_mnli.py finetune_hard_examples mnli_bert_base/checkpoint-last/ mnli_bert_base_fbert_imp_forget/ --hard_path mnli_bert_base/important_and_forget_examples.pkl &>output_log/out_fine_tune_bert_spurious_important_forget.log
```

### LID
```bash
# extract important samples from bert model
$ python spurious_bert_mnli.py extract_important_examples_with_LID mnli_bert_base/ --train_path mnli/MNLI/train.tsv --task mnli

$ nohup python spurious_bert_mnli.py finetune_hard_examples mnli_bert_base/checkpoint-last/ mnli_bert_base_fbert_important_samples/ --hard_path mnli_bert_base/important_examples_lid.pkl &>output_log/out_fine_tune_bert_LID_10percent.log
```

### Important samples + LID
```bash
# extract
$ python spurious_bert_mnli.py extract_imp_lid mnli_bert_base/ --train_path mnli/MNLI/train.tsv --task mnli

# fine-tune the model on bert forgettables combine with important samples
$ nohup python spurious_bert_mnli.py finetune_hard_examples mnli_bert_base/checkpoint-last/ mnli_bert_base_fbert_imp_lid/ --hard_path mnli_bert_base/important_examples_imp_lid.pkl &>output_log/out_fine_tune_bert_imp_lid.log
```


# II- Task : QQP - PAWSc / PLM & Shallow Model : BERT 
## 1. Download QQP and PAWS data

Download train/dev/test.tsv (QQP) and shuffled_train/dev_and_test.tsv (PAWS) at : https://drive.google.com/drive/folders/1egfbW0OSeRSOSZwQedFzN-m1RLLYUQs4?usp=drive_link

## 2. First Fine Tuning on QQP
```bash
$ python spurious_bert_qqp.py train_qqp_bert_base --output_dir qqp_bert_base/
```
## 3. Second Fine Tuning on Spurious Correlation
### Forgettable examples
```bash
# extract forgettables from bert model
$ python spurious_bert_mnli.py extract_hard_examples mnli_bow/ -train_path mnli/MNLI/train.tsv --task mnli 

$ nohup python spurious_bert_mnli.py finetune_hard_examples mnli_bert_base/checkpoint-last/ mnli_bert_base_fbert_forget_ex/ --hard_path mnli_bert_base/hard_examples.pkl &>output_log/out_fine_tune_bert_forget_ex.log
```

### Important examples
```bash
# extract important samples from bert model
$ python spurious_bert_mnli.py extract_important_examples mnli_bert_base/ --train_path mnli/MNLI/train.tsv --task mnli

$ nohup python spurious_bert_mnli.py finetune_hard_examples mnli_bert_base/checkpoint-last/ mnli_bert_base_fbert_important_samples/ --hard_path mnli_bert_base/important_examples.pkl &>output_log/out_fine_tune_bert_important_samples.log
```

### Important samples + Forgettable examples
```bash
# extract forgettables and important samples from bert model
$ python spurious_bert_mnli.py extract_important_forget_ex mnli_bert_base/ --train_path mnli/MNLI/train.tsv --task mnli

# fine-tune the model on bert forgettables combine with important samples
$ nohup python spurious_bert_mnli.py finetune_hard_examples mnli_bert_base/checkpoint-last/ mnli_bert_base_fbert_imp_forget/ --hard_path mnli_bert_base/important_and_forget_examples.pkl &>output_log/out_fine_tune_bert_spurious_important_forget.log
```

### LID
```bash
# extract important samples from bert model
$ python spurious_bert_mnli.py extract_important_examples_LID mnli_bert_base/ --train_path mnli/MNLI/train.tsv --task mnli

$ nohup python spurious_bert_mnli.py finetune_hard_examples mnli_bert_base/checkpoint-last/ mnli_bert_base_fbert_important_samples/ --hard_path mnli_bert_base/important_examples_lid.pkl &>output_log/out_fine_tune_bert_LID_10percent.log
```

### Important samples + LID
```bash
# extract
$ python spurious_bert_mnli.py extract_imp_lid mnli_bert_base/ --train_path mnli/MNLI/train.tsv --task mnli

# fine-tune the model on bert forgettables combine with important samples
$ nohup python spurious_bert_mnli.py finetune_hard_examples mnli_bert_base/checkpoint-last/ mnli_bert_base_fbert_imp_lid/ --hard_path mnli_bert_base/important_examples_imp_lid.pkl &>output_log/out_fine_tune_bert_imp_lid.log
```


# II- Task : FEVER - FEVER symmetric / PLM & Shallow Model : BERT 
## 1. Download MNLI data
```bash
$ sh getdata.sh fever && export MNLI_PATH=fever/
```
Download fever.train.jsonl at : https://fever.ai/dataset/fever.html (training task) and fever.dev.jsonl (label)
## 2. First Fine Tuning on MNLI
```bash
$ python spurious_bert_fever.py train_fever_bert_base --output_dir fever_bert_base/
```
## 3. Second Fine Tuning on Spurious Correlation
### Forgettable examples
```bash
# extract forgettables from bert model
$ python spurious_bert_mnli.py extract_hard_examples mnli_bow/ -train_path mnli/MNLI/train.tsv --task mnli 

$ nohup python spurious_bert_mnli.py finetune_hard_examples mnli_bert_base/checkpoint-last/ mnli_bert_base_fbert_forget_ex/ --hard_path mnli_bert_base/hard_examples.pkl &>output_log/out_fine_tune_bert_forget_ex.log
```

### Important examples
```bash
# extract important samples from bert model
$ python spurious_bert_mnli.py extract_important_examples mnli_bert_base/ --train_path mnli/MNLI/train.tsv --task mnli

$ nohup python spurious_bert_mnli.py finetune_hard_examples mnli_bert_base/checkpoint-last/ mnli_bert_base_fbert_important_samples/ --hard_path mnli_bert_base/important_examples.pkl &>output_log/out_fine_tune_bert_important_samples.log
```

### Important samples + Forgettable examples
```bash
# extract forgettables and important samples from bert model
$ python spurious_bert_mnli.py extract_important_forget_ex mnli_bert_base/ --train_path mnli/MNLI/train.tsv --task mnli

# fine-tune the model on bert forgettables combine with important samples
$ nohup python spurious_bert_mnli.py finetune_hard_examples mnli_bert_base/checkpoint-last/ mnli_bert_base_fbert_imp_forget/ --hard_path mnli_bert_base/important_and_forget_examples.pkl &>output_log/out_fine_tune_bert_spurious_important_forget.log
```

### LID
```bash
# extract important samples from bert model
$ python spurious_bert_mnli.py extract_important_examples_LID mnli_bert_base/ --train_path mnli/MNLI/train.tsv --task mnli

$ nohup python spurious_bert_mnli.py finetune_hard_examples mnli_bert_base/checkpoint-last/ mnli_bert_base_fbert_important_samples/ --hard_path mnli_bert_base/important_examples_lid.pkl &>output_log/out_fine_tune_bert_LID_10percent.log
```

### Important samples + LID
```bash
# extract
$ python spurious_bert_mnli.py extract_imp_lid mnli_bert_base/ --train_path mnli/MNLI/train.tsv --task mnli

# fine-tune the model on bert forgettables combine with important samples
$ nohup python spurious_bert_mnli.py finetune_hard_examples mnli_bert_base/checkpoint-last/ mnli_bert_base_fbert_imp_lid/ --hard_path mnli_bert_base/important_examples_imp_lid.pkl &>output_log/out_fine_tune_bert_imp_lid.log
```

