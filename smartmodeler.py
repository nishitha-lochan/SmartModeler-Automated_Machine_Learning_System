# Import Libraries and cofigurations
# Core Libraries
import numpy as np
import pandas as pd

# Train-Test Split

from sklearn.model_selection import train_test_split

# Preprocessing

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Models - Classification

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier


# Models - Regression

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from xgboost import XGBRegressor

#Unsupervised
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA

# Evaluation Metrics

from sklearn.base import ClassifierMixin,RegressorMixin
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import RandomizedSearchCV


# Visualization 

import matplotlib.pyplot as plt
import seaborn as sns

# Utilities

import warnings
warnings.filterwarnings("ignore")

#---------------------------------------------------------------------------


#Data Loader Engine
def load_data(data,target_column):
    if isinstance(data,str):
        df=pd.read_csv(data)
    elif isinstance(data,pd.DataFrame):
        df=data.copy()
    else:
        raise ValueError("Input must be a file path or pandas DataFrame")

    #Validate Target Column
    if target_column not in df.columns:
        raise ValueError("The target not in the Data")
    
    #validate Data is not empty
    if df.empty:
        raise ValueError("The data is empty")
    
    #Separate X and y
    X=df.drop(columns=[target_column])
    y=df[target_column]
    

    #basic checks
    if X.shape[1]==0:
        raise ValueError("There is no feature column")
    if y.isnull().any():
        raise ValueError("Target column contains missing values values")

    return X,y
#---------------------------------------------------------------------------
#Preprocessing Engine
class PreprocessingPipeline:
    def __init__(self, use_pca=False, n_comp=None):
        """
        use_pca : bool
            Whether to apply PCA on numerical features
        n_comp : int
            Number of principal components (required if use_pca=True)
        """
        self.use_pca = use_pca
        self.n_comp = n_comp

        self.pipeline = None
        self.feature_names = None
        self.num_col = None
        self.cat_col = None

    def fit(self, X):
        # Auto-detect column types
        self.num_col = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.cat_col = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Numerical pipeline
        num_steps = [
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]
        if self.use_pca:
            num_steps.append(('pca', PCA(n_components=self.n_comp)))

        num_pipeline = Pipeline(steps=num_steps)

        # Categorical pipeline
        cat_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine pipelines
        transformers = []
        if self.num_col:
            transformers.append(('num', num_pipeline, self.num_col))
        if self.cat_col:
            transformers.append(('cat', cat_pipeline, self.cat_col))

        self.pipeline = ColumnTransformer(transformers=transformers)
        self.pipeline.fit(X)

        # Feature names (for interpretability / debugging)
        self.feature_names = []

        if self.num_col:
            if self.use_pca:
                self.feature_names += [f'PC{i+1}' for i in range(self.n_comp)]
            else:
                self.feature_names += self.num_col

        if self.cat_col:
            cat_features = (
                self.pipeline
                .named_transformers_['cat']
                .named_steps['encoder']
                .get_feature_names_out(self.cat_col)
            )
            self.feature_names += cat_features.tolist()

        return self

    def transform(self, X):
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        X_transformed = self.pipeline.transform(X)

        # Convert sparse → dense if required
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        return X_transformed

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

#-----------------------------------------------------------------------------------------
#Training engine
class TrainingEngine:
    def __init__(self, models=None, metrics=None, tune_hyperparams=True):
        self.tune_hyperparams = tune_hyperparams

        # Default models if not provided
        if models is None:
            self.models = {
                # Classification
                'LogisticRegression': LogisticRegression(max_iter=500),
                'DecisionTree': DecisionTreeClassifier(),
                'RandomForest': RandomForestClassifier(),
                'KNN': KNeighborsClassifier(),
                'SVM': SVC(probability=True),
                'NaiveBayes': GaussianNB(),
                'AdaBoost': AdaBoostClassifier(),
                'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                # Regression
                'LinearRegression': LinearRegression(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'RandomForestRegressor': RandomForestRegressor(),
                'KNNRegressor': KNeighborsRegressor(),
                'SVR': SVR(),
                'AdaBoostRegressor': AdaBoostRegressor(),
                'XGBoostRegressor': XGBRegressor()
            }
        else:
            self.models = models
        
        if metrics is None:
            self.metrics = [
                'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 
                'confusion_matrix', 'classification_report',
                'mean_squared_error', 'r2_score'
            ]
        else:
            self.metrics = metrics
        
        self.results = {}
        self.best_model = None

    def train_and_evaluate(self, X_train, X_test, y_train, y_test, primary_metric='accuracy'):
        for name, model in self.models.items():
            tuned_model = model

            # Automatic hyperparameter tuning for commonly tuned models
            if self.tune_hyperparams:
                if name in ['RandomForest', 'RandomForestRegressor']:
                    param_grid = {
                        'n_estimators':[100,200,300],
                        'max_depth':[None,5,10],
                        'min_samples_split':[2,5,10]
                    }
                    tuned_model = RandomizedSearchCV(model, param_distributions=param_grid, cv=3, scoring='accuracy' if 'Regressor' not in name else 'r2')
                elif name in ['XGBoost', 'XGBoostRegressor']:
                    param_grid = {
                        'n_estimators':[100,200],
                        'max_depth':[3,5,7],
                        'learning_rate':[0.01,0.1,0.2]
                    }
                    tuned_model = RandomizedSearchCV(model, param_distributions=param_grid, cv=3, scoring='accuracy' if 'Regressor' not in name else 'r2')
                elif name in ['DecisionTree', 'DecisionTreeRegressor']:
                    param_grid = {'max_depth':[None,5,10,15],'min_samples_split':[2,5,10]}
                    tuned_model = RandomizedSearchCV(model, param_distributions=param_grid, cv=3, scoring='accuracy' if 'Regressor' not in name else 'r2')
                elif name in ['KNN', 'KNNRegressor']:
                    param_grid = {'n_neighbors':[3,5,7,9]}
                    tuned_model = RandomizedSearchCV(model, param_distributions=param_grid, cv=3, scoring='accuracy' if 'Regressor' not in name else 'r2')
                elif name in ['SVM', 'SVR']:
                    param_grid = {'C':[0.1,1,10],'gamma':['scale','auto']}
                    tuned_model = RandomizedSearchCV(model, param_distributions=param_grid, cv=3, scoring='accuracy' if 'Regressor' not in name else 'r2')
                # Others like LogisticRegression, NaiveBayes, AdaBoost can use default parameters for simplicity

            # Fit model (with tuning if applicable)
            tuned_model.fit(X_train, y_train)
            if isinstance(tuned_model, RandomizedSearchCV):
                model = tuned_model.best_estimator_
            else:
                model = tuned_model

            y_pred = model.predict(X_test)
            result = {}

            # Classification metrics
            if isinstance(model, ClassifierMixin):
                if 'accuracy' in self.metrics:
                     result['accuracy'] = accuracy_score(y_test, y_pred)
                if 'precision' in self.metrics: 
                    result['precision'] = precision_score(y_test, y_pred)
                if 'recall' in self.metrics: 
                    result['recall'] = recall_score(y_test, y_pred)
                if 'f1' in self.metrics: 
                    result['f1'] = f1_score(y_test, y_pred)
                if 'roc_auc' in self.metrics:
                    try: y_proba = model.predict_proba(X_test)[:,1]; result['roc_auc'] = roc_auc_score(y_test, y_proba)
                    except: result['roc_auc'] = None
                if 'confusion_matrix' in self.metrics:
                     result['confusion_matrix'] = confusion_matrix(y_test, y_pred)
                if 'classification_report' in self.metrics: 
                    result['classification_report'] = classification_report(y_test, y_pred)

            # Regression metrics
            if isinstance(model, RegressorMixin):
                if 'mean_squared_error' in self.metrics: 
                    result['mse'] = mean_squared_error(y_test, y_pred)
                if 'r2_score' in self.metrics: 
                    result['r2'] = r2_score(y_test, y_pred)

            self.results[name] = result

        # Select best model based on primary_metric
        valid_models = {k: v for k,v in self.results.items() if primary_metric in v}
        if valid_models:
            best_name = max(valid_models, key=lambda k: valid_models[k][primary_metric])
            self.best_model = self.models[best_name]
        
        return self.results, self.best_model


#-------------------------------------------------------------------------------------------------------------
#Prediction Engine
class PredictionEngine:
    def __init__(self, preprocessor, model):
        """
        preprocessor: fitted PreprocessingPipeline object
        model: trained model (best_model from TrainingEngine)
        """
        self.preprocessor = preprocessor
        self.model = model

    def predict(self, X_new):
        """
        Transform new data and return predictions.
        """
        # Transform new data
        X_transformed = self.preprocessor.transform(X_new)
        
        # Predict
        predictions = self.model.predict(X_transformed)
        
        # For classification, also return probabilities if available
        probabilities = None
        try:
            probabilities = self.model.predict_proba(X_transformed)
        except:
            pass  # some models like SVR won't have predict_proba
        
        return predictions, probabilities
#------------------------------------------------------------------------------------------


