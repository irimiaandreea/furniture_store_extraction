from transformers import BertTokenizerFast, BertForTokenClassification
from transformers import pipeline
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path='../config/.env')

def load_model_and_tokenizer(model_dir, tokenizer_dir):
    bert = BertForTokenClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_dir)
    ner = pipeline("ner", model=bert, tokenizer=tokenizer)
    return ner


def perform_ner(input_file, output_file, ner):
    label_map = {'B-MISC': 0, 'B-PRODUCT': 1, 'I-MISC': 2, 'I-PRODUCT': 3, 'O': 4}
    label_map_inverse = {v: k for k, v in label_map.items()}

    with open(input_file, "r") as input_file:
        with open(output_file, "w") as output_file:
            for line in input_file:
                ner_results = ner(line.strip())

                for ner_result in ner_results:
                    entity_label = label_map_inverse[int(ner_result['entity'].split('_')[-1])]
                    if entity_label in ['I-PRODUCT', 'B-PRODUCT']:
                        output_file.write(f"{entity_label}\t{ner_result['word']}\n")


if __name__ == "__main__":
    ner_trained_model_dir = os.getenv('NER_TRAINED_MODEL_DIR')
    ner_trained_tokenizer_dir = os.getenv('NER_TRAINED_TOKENIZER_DIR')
    input_test_file = os.getenv('INPUT_TEST_FILE')
    output_test_file = os.getenv('OUTPUT_TEST_FILE')

    ner = load_model_and_tokenizer(ner_trained_model_dir, ner_trained_tokenizer_dir)
    perform_ner(input_test_file, output_test_file, ner)
