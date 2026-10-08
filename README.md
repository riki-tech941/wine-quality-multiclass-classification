# Wine Quality Multiclass Classification

End-to-end machine learning pipeline for multiclass classification on chemical properties of wine, evaluating the impact of multicollinearity and feature selection on model generalization.

The project explores whether removing highly correlated features improves or degrades cross-validation and test performance across multiple tree-based and linear classifiers.

## Workflow & Methodology

- **Exploratory Data Analysis:** Target distribution inspection (imbalanced multi-class setting) and full Pearson correlation analysis.
- **Preprocessing:** Feature scaling and evaluation of two feature subsets: the complete feature space vs. an uncorrelated subset.
- **Model Training & Tuning:** Stratified K-Fold cross-validation paired with hyperparameter grid search for multiple architectures (Random Forest, Gradient Boosting, Logistic Regression, etc.).
- **Evaluation:** Model selection based on Macro/Weighted F1-score and Accuracy, evaluated via confusion matrices on an independent test set.

## Structure

- `src/`: Python script with data preprocessing, model training, cross-validation routines, and evaluation metrics.
- `data/`: Train and test datasets (`wine_train.xlsx`, `wine_test.xlsx`).
- `results/`: Output metrics (`.csv`) and evaluation figures (heatmaps, tuning sensitivity curves, confusion matrices).
- `report/`: Technical report summarizing experimental setup, ablation studies, and discussion of results.



## Authors & Credits

* **Riccardo Siervo** ([LinkedIn](https://www.linkedin.com/in/riccardo-siervo))
* Developed as a collaborative project by **Group 19** for the *Business Intelligence for Big Data* course @ Politecnico di Torino (A.Y. 2025/2026).
