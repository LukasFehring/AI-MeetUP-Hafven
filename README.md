
# Titanic Survival Prediction

This repository contains the data and scripts for predicting survival on the Titanic using auto-sklearn.

## Contents
- `titanic_train.csv` - The training dataset.
- `titanic_validation.csv` - The validation dataset for model tuning.
- `titanic_test.csv` - The test dataset for final model evaluation.
- `titanic_analysis.py` - Python script for training the model using auto-sklearn.

## Setup

### Prerequisites
- Python 3.x
- pip

### Installation
Install the required Python package `auto-sklearn` using pip:

```bash
pip install auto-sklearn
```

### Running the Script
To run the script and train the model, execute:

```bash
python titanic_analysis.py
```

The script will train a model using the training data and evaluate it on the validation set. The best model's performance will be printed out.

## Note
Ensure all data files and the script are in the same directory when you run the script.
