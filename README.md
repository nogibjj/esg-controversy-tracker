# Setup Instructions - Sentiment Analysis
1. Install all the libraries mentioned in the requirements.txt file
2. To train the generative model (Naive Bayes)
    a. cd sentiment_analysis/generative_model
    b. Run python naive_bayes.py (change the variable dataset_path on line 17 if required)
    c. The code will print the testing accuracy of the model and also create a synthetic dataset with the current datetime suffixed. 
    d. To skip the generation of the synthetic dataset, change the variable on line 18 generate_synthetic_data to False

3. To fine-tune the discriminative model (BERT)
    a. cd sentiment_analysis/discriminative_model
    b. Run python bert_fine_tuning.py (change the input_dataset_path on line 21/30 if required)
    c. The model gets saved under the folder ./sentiment-analysis-synthetic-data/. For the chosen dataset and training parameters, the last epoch is 90000, therefore the folder has to be used ./sentiment-analysis-synthetic-data/checkpoint-90000

4. To predict using the fine-tuned discriminative model (BERT)
    a. cd sentiment_analysis/discriminative_model
    b. Run python test_bert.py (change the input_dataset_path on line 21/30 if required)
    c. The script will print the accuracy metrics ()
