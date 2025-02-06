#!/bin/bash

# Exécuter chaque commande une après l'autre et attendre qu'elle se termine avant de passer à la suivante

nohup python spurious_roberta_mnli.py finetune_hard_examples mnli_roberta_base/checkpoint-last/ mnli_roberta_forget/ --hard_path mnli_roberta_base/forget_examples.pkl &> output_log/roberta/out_mnli/out_ft_forget.log

wait  # Attendre que la première commande soit terminée

nohup python spurious_roberta_mnli.py finetune_hard_examples mnli_roberta_base/checkpoint-last/ mnli_roberta_important/ --hard_path mnli_roberta_base/important_examples.pkl &> output_log/roberta/out_mnli/out_ft_important.log

wait  # Attendre que la deuxième commande soit terminée

nohup python spurious_roberta_mnli.py finetune_hard_examples mnli_roberta_base/checkpoint-last/ mnli_roberta_lid/ --hard_path mnli_roberta_base/important_examples_lid.pkl &> output_log/roberta/out_mnli/out_ft_lid.log

# execute with : sh run_fine_tune_roberta.sh
