import os
import shutil
from datasets import Dataset, load_metric
import numpy as np
import torch
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from dotenv import load_dotenv

load_dotenv(dotenv_path='../config/.env')


def clean_up_directories(*directories):
    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)


class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        input_ids = [torch.tensor(item["input_ids"]) for item in features]
        attention_mask = [torch.tensor(item["attention_mask"]) for item in features]
        labels = [torch.tensor(item["labels"]) for item in features]
        token_type_ids = [torch.tensor(item["token_type_ids"]) for item in features]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        token_type_ids = torch.nn.utils.rnn.pad_sequence(token_type_ids, batch_first=True,
                                                         padding_value=self.tokenizer.pad_token_type_id)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "token_type_ids": token_type_ids
        }


# read the CoNLL2003 data from the file exported by Label Studio
def read_conll2003_input_file(input_file_path):
    conll_input_dict = {"tokens": [], "ner_tags": []}
    tokens_inner_list = []
    ner_tags_inner_list = []
    with open(input_file_path, "r") as file:
        next(file)  # skip the first line, -DOCSTART- -X- O
        for line in file:
            parts = line.strip().split()
            if parts:
                if len(parts) == 4:  # CoNLL file has four columns
                    token = parts[0]
                    ner_tag = parts[3]

                    tokens_inner_list.append(token)
                    ner_tags_inner_list.append(ner_tag)
                else:
                    continue
            else:  # found a new line which means a separated page
                conll_input_dict["tokens"].append(tokens_inner_list)
                conll_input_dict["ner_tags"].append(ner_tags_inner_list)
                tokens_inner_list = []
                ner_tags_inner_list = []
    if tokens_inner_list:
        conll_input_dict["tokens"].append(tokens_inner_list)
    if ner_tags_inner_list:
        conll_input_dict["ner_tags"].append(ner_tags_inner_list)

    return conll_input_dict


def tokenize_and_align_labels(batch_of_sequences):
    tokenized_inputs = tokenizer(
        batch_of_sequences["tokens"], truncation=True, is_split_into_words=True, padding=True
    )
    labels = []
    for index, label in enumerate(batch_of_sequences["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=index)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_map.get(label[word_idx]))
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def train_and_evaluate_model(train_dataset, eval_dataset):
    bert = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(unique_label_list))
    training_args = TrainingArguments(  # hyperparameters
        output_dir=ner_trained_model_dir,
        evaluation_strategy="epoch",
        logging_dir=None,
        report_to=["none"],
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        save_strategy="epoch",
        warmup_ratio=0.1,
    )
    data_collator = CustomDataCollator(tokenizer=tokenizer)
    metric = load_metric("seqeval")

    def compute_metrics(eval_preds):
        pred_logits, labels = eval_preds
        pred_logits = np.argmax(pred_logits, axis=2)
        predictions = [[unique_label_list[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100]
                       for prediction, label in zip(pred_logits, labels)]
        true_labels = [[unique_label_list[l] for (eval_preds, l) in zip(prediction, label) if l != -100]
                       for prediction, label in zip(pred_logits, labels)]
        results = metric.compute(predictions=predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    trainer = Trainer(
        model=bert,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    testing_results = trainer.evaluate(test_dataset)
    trainer.save_model(ner_trained_model_dir)
    tokenizer.save_pretrained(ner_trained_tokenizer_dir)


def process_data(input_file):
    conll_input_dataset = Dataset.from_dict(read_conll2003_input_file(input_file))
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    unique_label_list = set()

    for sentence in conll_input_dataset:
        unique_label_list.update(sentence["ner_tags"])

    unique_label_list = sorted(unique_label_list)
    label_map = {label: i for i, label in enumerate(unique_label_list)}

    return conll_input_dataset, label_map, tokenizer, unique_label_list


if __name__ == "__main__":
    ner_trained_model_dir = os.getenv('NER_TRAINED_MODEL_DIR')
    ner_trained_tokenizer_dir = os.getenv('NER_TRAINED_TOKENIZER_DIR')
    input_train_file = os.getenv('INPUT_TRAIN_FILE')

    clean_up_directories(ner_trained_model_dir, ner_trained_tokenizer_dir)

    conll_input_dataset, label_map, tokenizer, unique_label_list = process_data(input_train_file)
    tokenized_dataset = conll_input_dataset.map(tokenize_and_align_labels, batched=True)

    # the tokenized_conll_input_dataset will be split into 80% train, 20% (10% test + 10% validation) datasets
    train_test_dataset_dict = tokenized_dataset.train_test_split(train_size=0.8, test_size=0.2)
    test_eval_dataset_dict = train_test_dataset_dict['test'].train_test_split(train_size=0.5, test_size=0.5)

    train_dataset = train_test_dataset_dict['train']
    test_dataset = test_eval_dataset_dict['test']
    eval_dataset = test_eval_dataset_dict['train']

    train_and_evaluate_model(train_dataset, eval_dataset)
