
# I- PLM : BERT / Task : MNLI - HANS / Shallow Model : BERT 
## 1. Download MNLI data
```bash
sh getdata.sh mnli && export MNLI_PATH=mnli/
```
## 2. First Fine Tuning on MNLI
```bash
python spurious_bert_mnli.py train_mnli_bert_base --output_dir mnli_bert_base/
```
## 3. Second Fine Tuning on Spurious Correlation
### Forgettable examples
```bash
# extract forgettables from bert model
python spurious_bert_mnli.py extract_forget_examples mnli_bert_base/ -train_path mnli/train.tsv --task mnli 

nohup python spurious_bert_mnli.py finetune_hard_examples mnli_bert_base/checkpoint-last/ mnli_bert_base_fbert_forget_ex/ --hard_path mnli_bert_base/forget_examples.pkl &>output_log/out_mnli/out_fine_tune_bert_forget_ex.log
```

### Important examples
```bash
# extract important samples from bert model
python spurious_bert_mnli.py extract_important_examples mnli_bert_base/ --train_path mnli/train.tsv --task mnli

nohup python spurious_bert_mnli.py finetune_hard_examples mnli_bert_base/checkpoint-last/ mnli_bert_base_fbert_important_samples/ --hard_path mnli_bert_base/important_examples.pkl &>output_log/out_mnli/out_fine_tune_bert_important_samples.log
```

### Important samples + Forgettable examples
```bash
# extract forgettables and important samples from bert model
python spurious_bert_mnli.py extract_important_forget_ex mnli_bert_base/ --train_path mnli/train.tsv --task mnli

# fine-tune the model on bert forgettables combine with important samples
nohup python spurious_bert_mnli.py finetune_hard_examples mnli_bert_base/checkpoint-last/ mnli_bert_base_fbert_imp_forget/ --hard_path mnli_bert_base/important_and_forget_examples.pkl &>output_log/out_mnli/out_fine_tune_bert_spurious_important_forget.log
```

### LID
```bash
# extract important samples from bert model
python spurious_bert_mnli.py extract_important_examples_with_LID mnli_bert_base/ --train_path mnli/train.tsv --task mnli

nohup python spurious_bert_mnli.py finetune_hard_examples mnli_bert_base/checkpoint-last/ mnli_bert_base_fbert_important_samples/ --hard_path mnli_bert_base/important_examples_lid.pkl &>output_log/out_mnli/out_fine_tune_bert_LID_10percent.log
```

### Important samples + LID
```bash
# extract
python spurious_bert_mnli.py extract_imp_lid mnli_bert_base/ --train_path mnli/MNLI/train.tsv --task mnli

# fine-tune the model on bert forgettables combine with important samples
nohup python spurious_bert_mnli.py finetune_hard_examples mnli_bert_base/checkpoint-last/ mnli_bert_base_fbert_imp_lid/ --hard_path mnli_bert_base/important_examples_imp_lid.pkl &>output_log/out_mnli/out_fine_tune_bert_imp_lid.log
```

----

# II- Task : Fever - fever-symmetric / PLM & Shallow Model : BERT 
## 1. Download FEVER and Fever-Symmetric data

Download fever jsonl files (FEVER) and fever-symm at : https://drive.google.com/drive/folders/1JE1F0e2pxST-Jkx8wh-6VNukV4KwrwlR?usp=drive_link

## 2. First Fine Tuning on FEVER
```bash
nohup python spurious_bert_fever.py train_fever_bert_base --output_dir fever_bert_base/ &>output_log/out_fever/out_ft.log
```
## 3. Second Fine Tuning on Spurious Correlation
### Forgettable examples
```bash
# extract forgettables from bert model
python spurious_bert_fever.py extract_forget_examples fever_bert_base/ -train_path fever/fever.train.jsonl --task fever 

nohup python spurious_bert_fever.py finetune_hard_examples fever_bert_base/checkpoint-last/ fever_bert_forget/ --hard_path fever_bert_base/hard_examples.pkl &>output_log/out_fever/out_ft_forget.log
```

### Important examples
```bash
# extract important samples from bert model
python spurious_bert_fever.py extract_important_examples fever_bert_base/ --train_path fever/fever.train.jsonl --task fever

nohup python spurious_bert_fever.py finetune_hard_examples fever_bert_base/checkpoint-last/ fever_bert_important/ --hard_path fever_bert_base/important_examples.pkl &>output_log/out_fever/out_ft_important.log
```

### LID
```bash
# extract important samples from bert model
python spurious_bert_fever.py extract_important_examples_LID fever_bert_base/ --train_path fever/fever.train.jsonl --task fever

nohup python spurious_bert_fever.py finetune_hard_examples fever_bert_base/checkpoint-last/ fever_bert_lid/ --hard_path fever_bert_base/important_examples_lid.pkl &>output_log/out_fever/out_ft_lid.log
```
---

# III - PLM : RoBERTa / Task : MNLI - HANS / Shallow Model : BERT 
## 1. Download MNLI data
```bash
sh getdata.sh mnli && export MNLI_PATH=mnli/
```
## 2. First Fine Tuning on MNLI
```bash
nohup python spurious_roberta_mnli.py train_mnli_roberta_base --output_dir mnli_roberta_base/ &>output_log/roberta/out_mnli/out_ft.log
```
## 3. Second Fine Tuning on Spurious Correlation
### Forgettable examples
```bash
# extract forgettables from bert model
python spurious_roberta_mnli.py extract_forget_examples mnli_roberta_base/ -train_path mnli/train.tsv --task mnli 

nohup python spurious_roberta_mnli.py finetune_hard_examples mnli_roberta_base/checkpoint-last/ mnli_roberta_forget/ --hard_path mnli_roberta_base/forget_examples.pkl &>output_log/roberta/out_mnli/out_ft_forget.log
```

### Important examples
```bash
# extract important samples from bert model
python spurious_roberta_mnli.py extract_important_examples mnli_roberta_base/ --train_path mnli/train.tsv --task mnli

nohup python spurious_roberta_mnli.py finetune_hard_examples mnli_roberta_base/checkpoint-last/ mnli_roberta_important/ --hard_path mnli_roberta_base/important_examples.pkl &>output_log/roberta/out_mnli/out_ft_important.log
```

### LID
```bash
# extract important samples from bert model
python spurious_roberta_mnli.py extract_important_examples_with_LID mnli_roberta_base/ --train_path mnli/train.tsv --task mnli

nohup python spurious_roberta_mnli.py finetune_hard_examples mnli_roberta_base/checkpoint-last/ mnli_roberta_lid/ --hard_path mnli_roberta_base/important_examples_lid.pkl &>output_log/roberta/out_mnli/out_ft_lid.log
```


