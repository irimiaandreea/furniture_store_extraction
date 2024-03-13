# furniture_store_extraction - challenge #2

Brief overview about the project workflow:

1. **Training the Model:** To train the NER model, execute the `train_ner.py` file. The training process takes approximately 6-7 minutes.
   - in terminal run the following command: `python train_ner.py`
2. **After Training:** Once the training is completed, two directories will be created:
   - the **output_ner_model** directory will contain the NER model along with its configuration
   - the **output_ner_tokenizer** directory will contain the NER's tokenizer along with its configuration
3. **Prepared Data:** The `crawled_train_data.txt` and `crawled_test_data.txt` files already contain prepared data for training and testing.
4. **Testing the Model:** To evaluate the trained model and tokenizer with new unseen data, execute the `test_ner.py` file.
   - in terminal run the following command: `python test_ner.py`
5. **After Testing:** The results of testing the model are stored as ***('entity', 'word') tuples*** and can be found in the `product_names.txt` file.
6. **Creating New Data:** (Optional) To generate a new set of unseen examples for testing the trained NER model, 
execute the `crawl_data.py` file with the desired input variables:
   - in terminal run the following command: `python crawl_data.py`
     - `input_file`: The input .csv file containing a list of URLs.
     - `start_valid_url`: Starting number of a valid crawled URL.
     - `limit`: Number of valid URLs crawled.
     - `training_data`: A boolean flag that separates the output data for training (`True`), from the output data for testing the model (`False`).
The ***crawled_train_data.txt*** file is used for annotation and training phase, 
while the ***crawled_test_data.txt*** file is used to perform new unseen pages on NER model. 
When creating new data for testing the model, ensure that the `training_data` is set to `False`. Setting it to `True` indicates that the model needs to be trained again.
7. For more information regarding the project and thought process, please visit this [Google Docs documentation](https://docs.google.com/document/d/1uQrA2weAnQ0KJe-A3IeMbnY54yZ-uBIgSt8anb8_YoY/edit?usp=sharing)