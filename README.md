# Meal/No-Meal Classification with Machine Learning

 - This repository implements machine learning models to classify continuous glucose monitoring (CGM) data into meal and no-meal periods.

## Project Goals

 - Train a decision tree classifier (DTC) or a support vector machine (SVM) to differentiate between meal and no-meal periods based on CGM data.
Evaluate the model's performance using accuracy, precision, recall, and F1 score.

 - Write a Python script that accepts two CSV files: CGMData.csv and InsulinData.csv, runs the analysis procedure, and outputs the metrics discussed in the metrics section.

## Data

**The project assumes access to:**

- CGM data containing glucose readings over time.
- Insulin data with timestamps and corresponding carbohydrate (carb) intake values.
Methodology

**Meal/No-Meal Data Identification:**

- Carb intake values from the insulin data are used to identify potential meal periods.
- CGM data segments corresponding to potential meal periods (2 hours 30 minutes) and non-meal periods (2 hours after potential meals) are extracted.
 
**Data Preprocessing:**

- Missing values are handled by either removing data segments with excessive missing values or imputing zeros.
- Feature extraction is performed using Principal Component Analysis (PCA) to reduce dimensionality.
 
**Model Training and Evaluation:**

- Separate label matrices are created for meal and no-meal data.
- Feature and label matrices are combined for both meal and no-meal data.
- The combined data is split into training (80%) and testing (20%) sets using scikit-learn's train_test_split function.
- A DTC model is trained using scikit-learn's DecisionTreeClassifier. Alternatively, an SVM model can be implemented.
- The trained model is evaluated on the testing set using metrics like accuracy, precision, recall, and F1 score.


## Evaluation Metrics

 **The script will output the following evaluation metrics:**
 
 - Accuracy: Proportion of correctly classified data points.
 - Precision: Proportion of true positives among predicted positives.
 - Recall: Proportion of true positives identified by the model.
 - F1 Score: Harmonic mean of precision and recall.