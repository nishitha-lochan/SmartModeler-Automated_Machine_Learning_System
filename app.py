import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import joblib
from io import StringIO

from sklearn.model_selection import train_test_split
from smartmodeler import PreprocessingPipeline, TrainingEngine, PredictionEngine

# --- Page configuration ---
st.set_page_config(page_title="SmartModeler", layout="wide")

# --- Custom CSS for background ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #E6F2FF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Title ---
st.title(" SmartModeler - AutoML Workflow")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    target_col = st.selectbox("Select the target column", df.columns)

    if st.button("Run Workflow"):
        stages = [
            ("Splitting Data", 0.2, "💙"),
            ("Preprocessing", 0.4, "💚"),
            ("Training Models", 0.7, "💛"),
            ("Making Predictions", 0.9, "🧡"),
            ("Workflow Completed", 1.0, "❤️")
        ]

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Stage 1: Split Data
        status_text.markdown(f"**{stages[0][2]} Stage 1:** Splitting data...")
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        progress_bar.progress(int(stages[0][1]*100))
        time.sleep(0.5)

        # Stage 2: Preprocessing
        status_text.markdown(f"**{stages[1][2]} Stage 2:** Preprocessing data...")
        preprocessor = PreprocessingPipeline(use_pca=False)
        X_train_prep = preprocessor.fit_transform(X_train)
        X_test_prep = preprocessor.transform(X_test)
        progress_bar.progress(int(stages[1][1]*100))
        time.sleep(0.5)

        # Stage 3: Training
        status_text.markdown(f"**{stages[2][2]} Stage 3:** Training models...")
        trainer = TrainingEngine()
        results, best_model = trainer.train_and_evaluate(X_train_prep, X_test_prep, y_train, y_test)
        progress_bar.progress(int(stages[2][1]*100))
        time.sleep(0.5)

        # Stage 4: Predictions
        status_text.markdown(f"**{stages[3][2]} Stage 4:** Making predictions...")
        predictor = PredictionEngine(preprocessor, best_model)
        y_pred, y_proba = predictor.predict(X_test)
        progress_bar.progress(int(stages[3][1]*100))
        time.sleep(0.5)

        # Stage 5: Done
        status_text.markdown(f"**{stages[4][2]} Workflow Completed!** ✅")
        progress_bar.progress(int(stages[4][1]*100))

        # Tabs for display: Metrics + Predictions only
        tab1, tab2 = st.tabs(["Metrics", "Predictions"])

        with tab1:
            st.subheader("🏆 Best Model")
            st.write(type(best_model).__name__)
            st.subheader("📈 Model Metrics")
            st.json(results[type(best_model).__name__])

            # Confusion Matrix
            if 'confusion_matrix' in results[type(best_model).__name__]:
                cm = results[type(best_model).__name__]['confusion_matrix']
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", ax=ax)
                st.pyplot(fig)

        with tab2:
            st.subheader("🔮 Predictions")
            prediction_df = X_test.copy()
            prediction_df["Prediction"] = y_pred
            if y_proba is not None:
                if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
                    prediction_df["Probability"] = y_proba[:,1]
            st.dataframe(prediction_df.head())

            # Download predictions
            csv_buffer = StringIO()
            prediction_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv_buffer.getvalue(),
                file_name="predictions.csv",
                mime="text/csv"
            )

        # Save trained model
        save_model = st.checkbox("💾 Save trained model?")
        if save_model:
            joblib.dump((preprocessor, best_model), "best_model.pkl")
            st.success("Model saved as best_model.pkl")
