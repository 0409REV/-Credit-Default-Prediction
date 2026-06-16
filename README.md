# -Credit-Default-Prediction

Built an end-to-end credit card default prediction system on a 30,000-record dataset, covering EDA, feature engineering, model training, and performance evaluation using Python (pandas, scikit-learn, seaborn, Plotly)
Engineered a modular sklearn Pipeline combining median/mode imputation, OneHot encoding for categorical risk features (payment status, education, marital status), and a custom OutlierRemover transformer with configurable σ-band clipping
Trained and benchmarked Decision Tree and Random Forest classifiers; optimized for recall (default detection rate) using GridSearchCV, RandomizedSearchCV, and HalvingGridSearchCV across 160+ hyperparameter combinations with Stratified K-Fold cross-validation
Conducted multivariate EDA to surface risk patterns by demographic segment — age, credit limit, education, and payment behavior — using correlation heatmaps, violin plots, KDE distributions, and default-rate stacked bar charts
Applied memory optimization techniques (downcasting dtypes, categorical encoding) reducing DataFrame size significantly, demonstrating data governance awareness in handling large financial datasets
Evaluated model performance using Precision, Recall, ROC-AUC, and confusion matrix analysis, with explicit prioritization of recall to minimize undetected defaults — directly aligned with fraud and credit risk decision-making logic
