# furniture_store_extraction - challenge #2

Brief overview about the project workflow:

1. The **NER model** is already trained and saved inside the **output_ner_model** directory,
along with its configuration.
2. The NER's **tokenizer** is already computed and saved inside the **output_ner_tokenizer** directory,
along with its configuration.
3. The ***crawled_train_data.txt*** and ***crawled_test_data.txt*** files already contain prepared data.
4. To train the model, run ***train_ner.py*** file.
5. To test the trained model and tokenizer with new unseen data, run ***test_ner.py*** file.
6. The results after testing the model are stored as **('entity', 'word') tuples** , 
should be visible inside the ***product_names.txt*** file. 
7. To perform another set of unseen examples on trained NER model, 
run the ***crawl_data.py*** file with the desired input variables:
    - `input_file` represents the input .csv file that contains a list of URLs
    - `start_valid_url` represents the starting number of a valid crawled URL
    - `limit` represents the number of valid URLs crawled
    - `training_data` represents a boolean that separates the output data for training, from the output data for testing the model. 
The ***crawled_train_data.txt*** file is used for annotation and training phase,
while the ***crawled_test_data.txt*** file is used to perform new unseen pages on NER model. When creating new data for testing the model,
it is important to set the `training_data` boolean to ``False``. If `training_data` is set to ``True``, then the model needs to be trained again.
        