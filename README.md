# Assignment 1

Advanced NLP Assignment 1 - Submitted by Ashna Dua, 2021101072

## Neural Network Language Model (NNLM)

- **`nnlm.py`**: Main script for:
  - Training a new model.
  - Reloading and evaluating an existing model.

  **Usage:**
  - **Train from Scratch**: `python nnlm.py`
    - Preprocesses data, trains the model, and saves perplexity scores.
  - **Reload Model**: `python nnlm.py --reload`
    - Reloads the model and displays validation and test perplexities.

- **`nnlm.ipynb`**: Jupyter notebook detailing the entire workflow, including hyperparameter tuning.

- **`data_store/`**: Directory for storing preprocessed data.
- **`models/`**: Directory for saving trained model.
- **`Perplexities/`**: Directory for saving text files containing perplexities (LM1).
- **`hyperparameter_tuning_results_nnlm.csv`**: CSV file containing all hyperparam results.


## LSTM Model

- **`lstm_model.py`**: Main script for:
  - Training a new LSTM model.
  - Reloading and evaluating an existing model.

  **Usage:**
  - **Train from Scratch**: `python lstm_model.py`
    - Preprocesses data, trains the model, and saves perplexity scores.
  - **Reload Model**: `python lstm_model.py --reload`
    - Reloads the model and displays validation and test perplexities.

- **`lstm_model.ipynb`**: Jupyter notebook detailing the entire workflow, including hyperparameter tuning.

- **`data_store/`**: Directory for storing preprocessed data.
- **`models/`**: Directory for saving trained model.
- **`Perplexities/`**: Directory for saving text files containing perplexities (LM2).


## Transformer Model

- **`transformer_model.py`**: Main script for:
  - Training a new Transformer model.
  - Reloading and evaluating an existing model.

  **Usage:**
  - **Train from Scratch**: `python transformer_model.py`
    - Preprocesses data, trains the model, and saves perplexity scores.
  - **Reload Model**: `python transformer_model.py --reload`
    - Reloads the model and displays validation and test perplexities.

- **`transformer_model.ipynb`**: Jupyter notebook detailing the entire workflow, including hyperparameter tuning.

- **`data_store/`**: Directory for storing preprocessed data.
- **`models/`**: Directory for saving trained model.
- **`Perplexities/`**: Directory for saving text files containing perplexities (LM3).


Models & Data Store uploaded at: https://drive.google.com/drive/folders/1SWWrqpZuIjoFroMqnuuztUYDkbsly9Ll?usp=share_link
