"""Main training script for AI-Based Network Intrusion Detection.

Executes the end-to-end pipeline:
1. Loads NSL-KDD train and test datasets.
2. Applies 5-class attack mapping.
3. Preprocesses 41 network features (Min-Max scaling + One-Hot encoding) fitted ONLY on train.
4. Fits K-Means (K=5) on preprocessed training features.
5. Transforms features to 5-dimensional distance-to-centroids representation.
6. Trains proposed Gaussian Naïve Bayes on 5D distance features.
7. Trains baseline Gaussian Naïve Bayes on full preprocessed features.
8. Trains baseline Decision Tree for comparison.
9. Evaluates all models on test set and computes accuracy, DR, FPR, F1.
10. Saves model artifacts to models/ and reports/plots to outputs/.
"""

import sys
from pathlib import Path
import joblib
import pandas as pd

from src.data import load_datasets, CLASS_NAMES
from src.preprocessing import Preprocessor
from src.model import (
    fit_kmeans,
    transform_distance_features,
    get_cluster_distribution,
    train_nb_kmeans_model,
    train_baseline_nb_model,
    train_decision_tree_model
)
from src.evaluation import (
    compute_metrics,
    plot_confusion_matrix,
    save_evaluation_results
)


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / 'data'
    models_dir = base_dir / 'models'
    outputs_dir = base_dir / 'outputs'

    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" AI-BASED NETWORK INTRUSION DETECTION: K-MEANS + NAÏVE BAYES")
    print("=" * 70)

    # 1. Load Data
    print("\n[Step 1/6] Loading NSL-KDD dataset from data/...")
    try:
        train_df, test_df = load_datasets(data_dir)
    except FileNotFoundError as e:
        print("\n" + "!" * 70)
        print("ERROR: DATASET NOT FOUND")
        print(str(e))
        print("!" * 70)
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Failed to load dataset: {e}")
        sys.exit(1)

    print(f" -> Train records: {len(train_df):,}")
    print(f" -> Test records:  {len(test_df):,}")
    print("\nTraining Class Distribution:")
    for cls_name, count in train_df['label_5class'].value_counts().items():
        print(f"    - {cls_name:<10}: {count:>6} ({count/len(train_df)*100:5.2f}%)")

    # 2. Preprocess features
    print("\n[Step 2/6] Fitting Preprocessor on Training Data...")
    preprocessor = Preprocessor()
    X_train_preprocessed = preprocessor.fit_transform(train_df)
    X_test_preprocessed = preprocessor.transform(test_df)

    y_train = train_df['label_5class'].values
    y_test = test_df['label_5class'].values

    print(f" -> Raw features: 41 (numerical & categorical)")
    print(f" -> Preprocessed feature space dimension: {X_train_preprocessed.shape[1]}")

    # 3. K-Means Clustering (K=5)
    print("\n[Step 3/6] Fitting K-Means (K=5) on Training Data...")
    kmeans = fit_kmeans(X_train_preprocessed, n_clusters=5, random_state=42)
    cluster_counts = get_cluster_distribution(kmeans, X_train_preprocessed)
    cluster_counts = {cluster_id: cluster_counts.get(cluster_id, 0) for cluster_id in range(5)}
    print(" -> Centroids extracted: 5")
    print(" -> Train samples per nearest cluster:")
    for cluster_id, count in sorted(cluster_counts.items()):
        print(f"    Cluster {cluster_id}: {count:,} samples ({count/len(X_train_preprocessed)*100:.2f}%)")
    print("    *Note: Cluster numbers are arbitrary mathematical partitions.")
    pd.Series(cluster_counts, name='sample_count').rename_axis('cluster_id').to_csv(
        outputs_dir / 'cluster_sizes.csv'
    )

    # 4. Distance-Based Feature Transformation
    print("\n[Step 4/6] Generating 5D Centroid Distance Features (Paper Methodology)...")
    X_train_dist = transform_distance_features(kmeans, X_train_preprocessed)
    X_test_dist = transform_distance_features(kmeans, X_test_preprocessed)
    print(f" -> Transformed Train feature shape: {X_train_dist.shape} (5 distances to centroids)")
    print(f" -> Transformed Test feature shape:  {X_test_dist.shape}")

    # 5. Model Training
    print("\n[Step 5/6] Training Classifiers...")
    print(" -> 1. Proposed Model: Gaussian Naïve Bayes on 5D Distance Features")
    nb_kmeans_model = train_nb_kmeans_model(X_train_dist, y_train)

    print(" -> 2. Baseline Model: Gaussian Naïve Bayes on Full Preprocessed Features")
    nb_baseline_model = train_baseline_nb_model(X_train_preprocessed, y_train)

    print(" -> 3. Comparison Model: Decision Tree Classifier")
    dt_model = train_decision_tree_model(X_train_preprocessed, y_train, random_state=42)

    # 6. Evaluation
    print("\n[Step 6/6] Evaluating Models on Test Set (KDDTest+)...")
    results = {}

    models_to_eval = [
        ("K-Means + Naïve Bayes (Proposed)", nb_kmeans_model, X_test_dist),
        ("Naïve Bayes Baseline", nb_baseline_model, X_test_preprocessed),
        ("Decision Tree Baseline", dt_model, X_test_preprocessed),
    ]

    for model_name, model_obj, X_eval in models_to_eval:
        y_pred = model_obj.predict(X_eval)
        m = compute_metrics(y_test, y_pred, model_name=model_name, classes=CLASS_NAMES)
        results[model_name] = m

        print(f"\n--- {model_name} ---")
        print(f"  Accuracy:             {m['accuracy']*100:.2f}%")
        print(f"  Detection Rate (DR):  {m['detection_rate']*100:.2f}%")
        print(f"  False Positive (FPR): {m['false_positive_rate']*100:.2f}%")
        print(f"  F1-Score (Macro):     {m['f1_macro']:.4f}")
        print(f"  F1-Score (Weighted):  {m['f1_weighted']:.4f}")

        # Save confusion matrix plot
        slug = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')
        cm_path = outputs_dir / f"confusion_matrix_{slug}.png"
        plot_confusion_matrix(m['confusion_matrix'], CLASS_NAMES, title=f"Confusion Matrix: {model_name}", save_path=cm_path)

    # Save summary & reports
    save_evaluation_results(results, outputs_dir)

    # Save artifacts
    print("\nSaving model artifacts to models/...")
    joblib.dump(preprocessor, models_dir / 'preprocessor.joblib')
    joblib.dump(kmeans, models_dir / 'kmeans_model.joblib')
    joblib.dump(nb_kmeans_model, models_dir / 'nb_kmeans_model.joblib')
    joblib.dump(nb_baseline_model, models_dir / 'nb_baseline_model.joblib')
    joblib.dump(dt_model, models_dir / 'dt_model.joblib')

    print("\n" + "=" * 70)
    print(" PERFORMANCE SUMMARY TABLE")
    print("=" * 70)
    summary_df = pd.read_csv(outputs_dir / 'model_comparison.csv')
    print(summary_df.to_string(index=False))
    print("=" * 70)
    print(f"\nArtifacts successfully saved to:")
    print(f" - Models:  {models_dir.resolve()}")
    print(f" - Outputs: {outputs_dir.resolve()}")


if __name__ == '__main__':
    main()
