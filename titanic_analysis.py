
# Titanic Survival Prediction using auto-sklearn

import autosklearn.classification as auto_class
import autosklearn.metrics as auto_metrics
import pandas as pd

# Load your data
train_data = pd.read_csv('titanic_train.csv')
val_data = pd.read_csv('titanic_validation.csv')

# Separate features and target
X_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']
X_val = val_data.drop('Survived', axis=1)
y_val = val_data['Survived']

# Setup auto-sklearn classifier
automl = auto_class.AutoSklearnClassifier(
    time_left_for_this_task=120,  # Allocate time for search (in seconds)
    per_run_time_limit=30,        # Time limit for a single call to the machine learning model
    metric=auto_metrics.accuracy, # We optimize for classification accuracy
    ensemble_size=1               # Use a single model for simplicity
)

# Train the model
automl.fit(X_train.copy(), y_train.copy(), dataset_name='Titanic_train')

# Evaluate the performance on the validation set
validation_accuracy = automl.score(X_val.copy(), y_val.copy())
print(f"Validation Accuracy: {validation_accuracy}")

# You can also inspect the best model found
print(automl.show_models())
