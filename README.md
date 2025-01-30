
1. download MNLI data
```bash
$ sh getdata.sh mnli && export MNLI_PATH=mnli/MNLI/
```
2. First Fine Tune on MNLI :
## Fine tune BERT base model on MNLI
```bash
$ python exp_cli_spurious.py train_mnli_bert_base --output_dir mnli_bert_base/
```
## Forgettable examples
```bash
# extract forgettables from bert model
$ python exp_cli_spurious.py extract_hard_examples mnli_bow/ -train_path mnli/MNLI/train.tsv --task mnli 

$ nohup python exp_cli_spurious.py finetune_hard_examples mnli_bert_base/checkpoint-last/ mnli_bert_base_fbert_forget_ex/ --hard_path mnli_bert_base/hard_examples.pkl &>out_fine_tune_bert_forget_ex.log
```

## Important examples
```bash
# extract important samples from bert model
$ python exp_cli_spurious.py extract_important_examples mnli_bert_base/ --train_path mnli/MNLI/train.tsv --task mnli

$ nohup python exp_cli_spurious.py finetune_hard_examples mnli_bert_base/checkpoint-last/ mnli_bert_base_fbert_important_samples/ --hard_path mnli_bert_base/important_examples.pkl &>out_fine_tune_bert_important_samples.log
```

## Important samples + Forgettable examples
```bash
# extract forgettables and important samples from bert model
$ python exp_cli_spurious.py extract_important_forget_ex mnli_bert_base/ --train_path mnli/MNLI/train.tsv --task mnli

# fine-tune the model on bert forgettables combine with important samples
$ nohup python exp_cli_spurious.py finetune_hard_examples mnli_bert_base/checkpoint-last/ mnli_bert_base_fbert_imp_forget/ --hard_path mnli_bert_base/important_and_forget_examples.pkl &>out_fine_tune_bert_spurious_important_forget.log
```

## LID
```bash
# extract important samples from bert model
$ python exp_cli_spurious.py extract_important_examples_LID mnli_bert_base/ --train_path mnli/MNLI/train.tsv --task mnli

$ nohup python exp_cli_spurious.py finetune_hard_examples mnli_bert_base/checkpoint-last/ mnli_bert_base_fbert_important_samples/ --hard_path mnli_bert_base/important_examples_probs_threshold_10percent.pkl &>out_fine_tune_bert_LID_10percent.log
```

## Important samples + LID
```bash
# extract
$ python exp_cli_spurious.py extract_imp_lid mnli_bert_base/ --train_path mnli/MNLI/train.tsv --task mnli

# fine-tune the model on bert forgettables combine with important samples
$ nohup python exp_cli_spurious.py finetune_hard_examples mnli_bert_base/checkpoint-last/ mnli_bert_base_fbert_imp_lid/ --hard_path mnli_bert_base/important_examples_imp_lid.pkl &>out_fine_tune_bert_imp_lid.log
```
