import os
import numpy
import subprocess
import glob
import logging
from pprint import pprint
import inspect
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MNLI_DATA_PATH = os.getenv("MNLI_PATH", "~/Documents/robustness_to_spurious_correlation/robustness_to_spurious_correlation/mnli/MNLI")
FEVER_DATA_PATH = os.getenv("FEVER_PATH", "~/mygit/jiant/data/FEVER/")
PAWSQQP_DATA_PATH = os.getenv("PAWS_QQP_PATH", "~/projet_3A/robustness_to_spurious_correlation/qqp")

class settings(type):
    def __new__(self, name, bases, classdict):
        classdict['fields'] = dict(
            (str(key), str(value)) for key, value in classdict.items()
            if key not in ('__module__', '__qualname__'))
        return type.__new__(self, name, bases, classdict)

class bert_defaults(metaclass=settings):
    per_gpu_eval_batch_size = 32
    per_gpu_train_batch_size = 32
    num_train_epochs = 10
    decay_learning_rate = 'True'
    do_lower_case = 'True'
    learning_rate = 5e-5
    model_name_or_path = 'bert-base-uncased'
    model_type = 'bert'

class bert_large_defaults(metaclass=settings):
    per_gpu_eval_batch_size = 32
    per_gpu_train_batch_size = 32
    num_train_epochs = 10
    decay_learning_rate = 'True'
    do_lower_case = 'True'
    learning_rate = 5e-5
    model_name_or_path = 'bert-large-uncased'
    model_type = 'bert'


class mnli_defaults(metaclass=settings):
    data_dir = f'{MNLI_DATA_PATH}'
    # fp16 = ''
    task_name = 'mnli'
    do_train = ''
    overwrite_output_dir = ''
    per_gpu_eval_batch_size = 32
    num_train_epochs = 4

class pawsqqp_defaults(metaclass=settings):
    data_dir = f'{PAWSQQP_DATA_PATH}'
    fp16 = ''
    task_name = 'qqp'
    do_train = ''
    overwrite_output_dir = ''
    eval_tasks = 'qqp qqp-wang qqp-wang-test paws-qqp paws-wiki paws-qqp-all-val'
    learning_rate = '5e-5'
    num_train_epochs = 3
    weight_decay = 0.0
    per_gpu_train_batch_size = 32
    per_gpu_eval_batch_size = 400


def execute(entry_point, kwargs):
    pprint(kwargs)
    args = ' '.join(f'--{str(k)} {str(v)}' for k, v in kwargs.items())
    print(f"python {entry_point} {args}")
    os.system(f"python {entry_point} {args}")


class Main():
    def extract_subset_from_glove(self, glove_path, dictionary, output_dir):
        """Extracts a subset of vectors from the full glove dictionary
           and stores them in output_dir/embeddings.pkl
        """
        from models_weak import extract_subset_from_glove
        extract_subset_from_glove(glove_path, dictionary, output_dir)
        
    ##########################
    ######   Spurious   ######
    ##########################

    def extract_important_examples(
            self,
            example_stats_path,
            labels_file=None,
            train_path=None,
            task='mnli'
        ):
        
        """Given a model examples stats, compute importance scores, select examples
           with the highest score differences, and store the example file in the specified directory.
        """
        
        import pickle
        import numpy as np
        from pathlib import Path
        import pandas as pd 

        output_path = example_stats_path + 'important_examples.pkl'
        examples_stats = pickle.load(open(example_stats_path + '/example_stats.pkl', 'rb'))
        n_epochs = examples_stats['accuracy'].shape[1]
        print("Loaded example stats,", examples_stats.keys())

        if labels_file:
            labels = open(labels_file, 'r').readlines()
            labels_dict = dict()
            for line in labels:
                id, label = line.strip().split()
                labels_dict[int(id)] = int(label)
        if train_path:
            if task == 'mnli':
                df = pd.read_csv(
                        train_path,
                        delimiter='\t',
                        on_bad_lines='skip', 
                        skiprows=0,
                        quoting=3,
                        keep_default_na=False,
                        encoding="utf-8",)
            elif task == 'fever':
                import json
                with open(train_path, 'r') as f:
                    data = [json.loads(s.strip()) for s in f.readlines()]
                df = pd.DataFrame(data)
            labels_dict = dict()
            for id, label in enumerate(df.gold_label):
                labels_dict[int(id)] = label

        def compute_importance_scores(examples_stats):
            """Calculates an importance score for each example."""
            loss = examples_stats['loss'][:, -1]  # Last loss value
            accuracy = examples_stats['accuracy'][:, -1]  # Last accuracy value
            importance_scores = loss * (1 - accuracy)  # Example score formula
            return importance_scores

        def select_examples_with_highest_score_diff(importance_scores, threshold=0.1):
            """Selects examples with the highest score differences."""
            score_diff = np.abs(np.diff(importance_scores))  # Absolute difference in scores
            selected_indices = np.where(score_diff > threshold)[0]  # Indices with significant score differences
            return selected_indices
        def balance_by_class(hard_ids, labels_dict):
            """Balances the examples by class."""
            by_label = dict()
            for id in hard_ids:
                label = labels_dict[id]
                arr = by_label.get(label, [])
                arr.append(id)
                by_label[label] = arr
            min_num = np.min([len(arr) for arr in by_label.values()])
            balanced_ids = []
            for arr in by_label.values():
                balanced_ids.extend(arr[:min_num])
            return np.array(balanced_ids)

        importance_scores = compute_importance_scores(examples_stats)

        # Select examples with the highest score differences
        selected_indices = select_examples_with_highest_score_diff(importance_scores)

        balanced_selected_indices = balance_by_class(selected_indices, labels_dict)

        # Save the results in a pickle file
        results = {
            'selected_examples': selected_indices,
            'forgettables_b': balanced_selected_indices
        }

        with open(output_path, "wb") as f:
            pickle.dump(results, f)

        print(f"Important examples saved to {output_path}")
    
    def extract_important_forget_ex(
            self,
            example_stats_path,
            labels_file=None,
            train_path=None,
            task='mnli'
        ):
        
        import pickle
        import numpy as np
        import pandas as pd 

        output_path = example_stats_path + 'important_and_forget_examples.pkl'
        examples_stats = pickle.load(open(example_stats_path + '/example_stats.pkl', 'rb'))
        n_epochs = examples_stats['accuracy'].shape[1]
        print("Loaded example stats,", examples_stats.keys())

        if labels_file:
            labels = open(labels_file, 'r').readlines()
            labels_dict = {int(id): int(label) for id, label in (line.strip().split() for line in labels)}
        elif train_path:
            if task == 'mnli':
                df = pd.read_csv(
                    train_path,
                    delimiter='\t',
                    on_bad_lines='skip', 
                    quoting=3,
                    keep_default_na=False,
                    encoding="utf-8",
                )
            labels_dict = {int(id): label for id, label in enumerate(df.gold_label)}

        def balance_by_class(ids, labels_dict):
            by_label = {}
            for id in ids:
                label = labels_dict[id]
                by_label.setdefault(label, []).append(id)
            min_count = min(len(arr) for arr in by_label.values())
            balanced_ids = [id for arr in by_label.values() for id in arr[:min_count]]
            return np.array(balanced_ids)
        
        ###############
        ## important ##
        ###############
        def compute_importance_scores(examples_stats):
            """Calculates an importance score for each example."""
            loss = examples_stats['loss'][:, -1]
            accuracy = examples_stats['accuracy'][:, -1]
            return loss * (1 - accuracy)

        def select_examples_with_highest_score_diff(importance_scores, threshold=0.3):
            """Selects examples with the highest score differences."""
            score_diff = np.abs(np.diff(importance_scores))
            return np.where(score_diff > threshold)[0]
        
        # Compute important examples
        importance_scores = compute_importance_scores(examples_stats)
        important_indices = select_examples_with_highest_score_diff(importance_scores)
        
        ##############
        ### forget ###
        ##############  
        def select_forgettables():
            from utils_forgetting import compute_forgetting
            forgetting, _, max_correct = compute_forgetting(examples_stats['accuracy'])
            never_learnt = np.where(forgetting == max_correct)[0]
            return forgetting, never_learnt

        # Compute forgettables
        forgettables, never_learnt = select_forgettables()

        ###############
        ### Combine ###
        ###############
        combined_indices = np.union1d(important_indices, forgettables)
        balanced_combined_indices = balance_by_class(combined_indices, labels_dict)

        results = {
            'selected_examples': combined_indices,
            'forgettables_b': balanced_combined_indices
        }

        with open(output_path, "wb") as f:
            pickle.dump(results, f)

        print(f"Important and forgettable examples saved to {output_path}")
    
    def extract_important_examples_with_LID(
            self,
            example_stats_path,
            labels_file=None,
            train_path=None,
            task='mnli'
        ):
        
        """Given a model examples stats, compute importance scores based on probs, select examples
        with the highest score differences, and store the example file in the specified directory.
        """
        import pickle
        import numpy as np
        from pathlib import Path
        import pandas as pd 

        output_path = example_stats_path + 'important_examples_lid.pkl'
        examples_stats = pickle.load(open(example_stats_path + '/example_stats.pkl', 'rb'))
        n_epochs = examples_stats['accuracy'].shape[1]
        print("Loaded example stats,", examples_stats.keys())

        if labels_file:
            labels = open(labels_file, 'r').readlines()
            labels_dict = dict()
            for line in labels:
                id, label = line.strip().split()
                labels_dict[int(id)] = int(label)
        if train_path:
            if task == 'mnli':
                df = pd.read_csv(
                        train_path,
                        delimiter='\t',
                        on_bad_lines='skip', 
                        skiprows=0,
                        quoting=3,
                        keep_default_na=False,
                        encoding="utf-8",)
            elif task == 'fever':
                import json
                with open(train_path, 'r') as f:
                    data = [json.loads(s.strip()) for s in f.readlines()]
                df = pd.DataFrame(data)
            labels_dict = dict()
            for id, label in enumerate(df.gold_label):
                labels_dict[int(id)] = label

        def compute_importance_scores(examples_stats):
            """Calculates an importance score for each example based on probs."""
            probs = examples_stats['probs'][:, -1]  # Last probability values
            accuracy = examples_stats['accuracy'][:, -1]  # Last accuracy value
            importance_scores = np.min(probs) * (1 - accuracy)  # Utiliser la probabilité minimale
            return importance_scores

        def select_examples_with_highest_score_diff(importance_scores, threshold):
            """Selects examples with the highest score differences."""
            score_diff = np.abs(np.diff(importance_scores))  # Absolute difference in scores
            selected_indices = np.where(score_diff > threshold)[0]  # Indices with significant score differences
            return selected_indices

        def balance_by_class(hard_ids, labels_dict):
            """Balances the examples by class."""
            by_label = dict()
            for id in hard_ids:
                label = labels_dict[id]
                arr = by_label.get(label, [])
                arr.append(id)
                by_label[label] = arr
            min_num = np.min([len(arr) for arr in by_label.values()])
            balanced_ids = []
            for arr in by_label.values():
                balanced_ids.extend(arr[:min_num])
            return np.array(balanced_ids)

        importance_scores = compute_importance_scores(examples_stats)

        # Select examples with the highest score differences
        # threshold = np.mean(importance_scores) - 2 * np.std(importance_scores) # select examples with scores significa less than means
        threshold = np.percentile(importance_scores, 10)  # select 10% examples 
        selected_indices = select_examples_with_highest_score_diff(importance_scores, threshold)

        balanced_selected_indices = balance_by_class(selected_indices, labels_dict)

        # Save the results in a pickle file
        results = {
            'selected_examples': selected_indices,
            'forgettables_b': balanced_selected_indices
        }

        with open(output_path, "wb") as f:
            pickle.dump(results, f)

        print(f"Important examples saved to {output_path}")

    def extract_imp_lid(
        self,
            example_stats_path,
            labels_file=None,
            train_path=None,
            task='mnli'
        ):
        
        """Given a model examples stats, compute importance scores based on probs, select examples
        with the highest score differences, and store the example file in the specified directory.
        """
        import pickle
        import numpy as np
        from pathlib import Path
        import pandas as pd 

        output_path = example_stats_path + 'important_examples_imp_lid.pkl'
        examples_stats = pickle.load(open(example_stats_path + '/example_stats.pkl', 'rb'))
        n_epochs = examples_stats['accuracy'].shape[1]
        print("Loaded example stats,", examples_stats.keys())

        if labels_file:
            labels = open(labels_file, 'r').readlines()
            labels_dict = dict()
            for line in labels:
                id, label = line.strip().split()
                labels_dict[int(id)] = int(label)
        if train_path:
            if task == 'mnli':
                df = pd.read_csv(
                        train_path,
                        delimiter='\t',
                        on_bad_lines='skip', 
                        skiprows=0,
                        quoting=3,
                        keep_default_na=False,
                        encoding="utf-8",)
            elif task == 'fever':
                import json
                with open(train_path, 'r') as f:
                    data = [json.loads(s.strip()) for s in f.readlines()]
                df = pd.DataFrame(data)
            labels_dict = dict()
            for id, label in enumerate(df.gold_label):
                labels_dict[int(id)] = label

        def balance_by_class(hard_ids, labels_dict):
            """Balances the examples by class."""
            by_label = dict()
            for id in hard_ids:
                label = labels_dict[id]
                arr = by_label.get(label, [])
                arr.append(id)
                by_label[label] = arr
            min_num = np.min([len(arr) for arr in by_label.values()])
            balanced_ids = []
            for arr in by_label.values():
                balanced_ids.extend(arr[:min_num])
            return np.array(balanced_ids)
        #############
        ###  LID  ###
        #############  

        def compute_importance_scores(examples_stats):
            """Calculates an importance score for each example based on probs."""
            probs = examples_stats['probs'][:, -1]  # Last probability values
            accuracy = examples_stats['accuracy'][:, -1]  # Last accuracy value
            importance_scores = np.min(probs) * (1 - accuracy)  # Utiliser la probabilité minimale
            return importance_scores

        def select_examples_with_highest_score_diff(importance_scores, threshold):
            """Selects examples with the highest score differences."""
            score_diff = np.abs(np.diff(importance_scores))  # Absolute difference in scores
            selected_indices = np.where(score_diff > threshold)[0]  # Indices with significant score differences
            return selected_indices

        importance_scores = compute_importance_scores(examples_stats)

        # threshold = np.mean(importance_scores) - 2 * np.std(importance_scores) # select examples with scores significa less than means
        threshold = np.percentile(importance_scores, 10)  # select 10% examples 
        selected_indices = select_examples_with_highest_score_diff(importance_scores, threshold)
        
        ###############
        ## important ##
        ###############
        def compute_importance_scores(examples_stats):
            """Calculates an importance score for each example."""
            loss = examples_stats['loss'][:, -1]
            accuracy = examples_stats['accuracy'][:, -1]
            return loss * (1 - accuracy)

        def select_examples_with_highest_score_diff(importance_scores, threshold=0.3):
            """Selects examples with the highest score differences."""
            score_diff = np.abs(np.diff(importance_scores))
            return np.where(score_diff > threshold)[0]
        
        # Compute important examples
        importance_scores = compute_importance_scores(examples_stats)
        important_indices = select_examples_with_highest_score_diff(importance_scores)

        ###############
        ### Combine ###
        ###############
        combined_indices = np.union1d(important_indices, selected_indices)
        balanced_combined_indices = balance_by_class(combined_indices, labels_dict)

        results = {
            'selected_examples': combined_indices,
            'forgettables_b': balanced_combined_indices
        }

        with open(output_path, "wb") as f:
            pickle.dump(results, f)

        print(f"Important and LID examples saved to {output_path}")
        

    def train_qqp_bert_base(self, output_dir, seed=0):
        args = bert_defaults.fields
        args.update(pawsqqp_defaults.fields) # modify mnli_default to pawsqqp_defaults
        args.update(dict(output_dir=output_dir, seed=seed))
        execute("exp_glue.py", args)

   
    def finetune_hard_examples(
        self,
        base_model_path,
        output_dir,
        base_model_type="bert_base",
        hard_path="",
        hard_type="forgettables_b",
        training_examples_ids=None,
        seed=0,
        task='qqp'
    ):
        """Finetune a base model, e.g. bert, on hard examples
           from a weaker model, e.g. bow.
        """
        from pathlib import Path

        if base_model_type == "bert_base":
            args = bert_defaults.fields
        else:
            assert False

        # if task == 'fever':
        #     args.update(fever_defaults.fields)
        if task == 'mnli':
            args.update(mnli_defaults.fields)
        elif task =='qqp':
            args.update(pawsqqp_defaults.fields)

        args.update(dict(num_train_epochs=4, learning_rate=5e-6, per_gpu_train_batch_size=32))
        args.update(dict(load_model=base_model_path))
        args.update(dict(output_dir=Path(output_dir) / base_model_type, seed=seed))
        args.update(dict(training_examples_ids=training_examples_ids))
        if hard_path:
            args.update(dict(hard_examples=hard_path, hard_type=hard_type))

        # fine-tune base model on hard examples
        execute("exp_glue.py", args)
        
    def test(
            self, base_model_path,
            base_model_type="bert_base",
            task='fever', dev=False
        ):
        from pathlib import Path
        args = bert_defaults.fields
        if task == 'mnli':
            args.update(mnli_defaults.fields)

        args.update(dict(eval_tasks=task))
        args.update(dict(load_model=base_model_path))
        args.update(dict(output_dir=base_model_path))
        args.update(dict(output_dir=base_model_path))
        if 'do_train' in args:
            args.pop('do_train')
        if not dev:
            args.update(dict(test=""))
        args.update(dict(per_gpu_eval_batch_size="32"))

        # train base model on hard examples
        execute("exp_glue.py", args)


if __name__ == '__main__':
    import fire
    fire.Fire(Main)
