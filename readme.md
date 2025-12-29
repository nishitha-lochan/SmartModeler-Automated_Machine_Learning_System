## **Project Name:** SmartModeler – Automated ML Workflow


### **Problem Statement:**

Data science projects require multiple stages: data preprocessing, model selection, training, evaluation, and prediction. For non-technical users or beginners, manually performing each step is time-consuming and error-prone.

**SmartModeler** solves this by providing an **automated machine learning workflow** that allows users to:

* Upload any CSV dataset.
* Automatically preprocess the data (handle missing values, scaling, encoding, PCA if needed).
* Automatically train multiple models for classification and regression.
* Evaluate models using multiple metrics (accuracy, precision, recall, F1, ROC-AUC, MSE, R², confusion matrix).
* Display predictions for new or test data.
* Download the predictions and optionally save the trained model.

### **Key Features:**

1. **Automatic Preprocessing**

   * Detects numerical and categorical columns.
   * Handles missing values.
   * Standardizes numerical features.
   * One-hot encodes categorical features.
   * Optional PCA for dimensionality reduction.

2. **Automatic Model Selection**

   * Trains multiple classification and regression models.
   * Evaluates each model on relevant metrics.
   * Selects the best model based on primary metric.

3. **Model Evaluation & Metrics**

   * Classification: Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix.
   * Regression: MSE, R² score.
   * Visualizes Confusion Matrix.

4. **Predictions**

   * Outputs predictions on new data.
   * Provides probabilities for classification tasks (if applicable).
   * Option to download predictions as CSV.

5. **Model Persistence**

   * Save the best-trained model for future use with a single click.

6. **Streamlit Interface**

   * User-friendly web app.
   * Interactive progress bar for each stage.
   * Tabbed interface for Dataset, Metrics, and Predictions.



### **Technologies Used:**

* Python 3.x
* scikit-learn
* Pandas, NumPy
* Seaborn, Matplotlib
* Streamlit (Web interface)
* Joblib (Model saving)


### **How to Use:**

1. Clone the repository:

   git clone <repository_url>
   
2. Install dependencies:

   pip install -r requirements.txt

3. Run the Streamlit app:

   streamlit run app.py
   
4. Upload your CSV dataset, select the target column, and run the workflow.
5. View metrics, predictions, and optionally download the CSV or save the trained model.



### **Sample Output:**

* Best model selected automatically (e.g., RandomForestClassifier or AdaBoostClassifier).
* Metrics displayed in tabular and JSON formats.
* Confusion matrix visualized for classification tasks.
* Predictions shown for test data with probability values (for classification).
* Downloadable CSV with predictions.
