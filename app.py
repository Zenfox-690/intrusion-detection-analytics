"""Streamlit dashboard for the NSL-KDD intrusion detection mini project."""

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA

from src.data import ALL_COLUMNS, CLASS_NAMES, FEATURE_NAMES, load_datasets
from src.model import transform_distance_features


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
MODELS_DIR = ROOT / 'models'
OUTPUTS_DIR = ROOT / 'outputs'


@st.cache_data(show_spinner=False)
def get_datasets():
    return load_datasets(DATA_DIR)


@st.cache_resource(show_spinner=False)
def get_artifacts():
    artifact_paths = {
        'preprocessor': MODELS_DIR / 'preprocessor.joblib',
        'kmeans': MODELS_DIR / 'kmeans_model.joblib',
        'classifier': MODELS_DIR / 'nb_kmeans_model.joblib',
    }
    missing = [path.name for path in artifact_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifacts: {', '.join(missing)}. Run `python train.py` first."
        )
    return {name: joblib.load(path) for name, path in artifact_paths.items()}


def read_metrics():
    path = OUTPUTS_DIR / 'evaluation_metrics.json'
    if not path.is_file():
        return {}
    with path.open(encoding='utf-8') as metrics_file:
        return json.load(metrics_file)


def read_prediction_csv(uploaded_file):
    uploaded_file.seek(0)
    frame = pd.read_csv(uploaded_file)
    normalized_columns = {str(column).strip().lower(): column for column in frame.columns}
    if all(name in normalized_columns for name in FEATURE_NAMES):
        return frame.rename(columns={value: key for key, value in normalized_columns.items()})[
            FEATURE_NAMES
        ]

    uploaded_file.seek(0)
    frame = pd.read_csv(uploaded_file, header=None)
    if frame.shape[1] not in (41, 42, 43):
        raise ValueError(
            f"Expected 41 features, optionally followed by label and difficulty columns; "
            f"found {frame.shape[1]} columns."
        )
    frame.columns = ALL_COLUMNS[:frame.shape[1]]
    return frame[FEATURE_NAMES]


def load_dashboard_artifacts():
    try:
        return get_artifacts()
    except (FileNotFoundError, OSError) as error:
        st.warning(str(error))
        return None


st.set_page_config(page_title='NSL-KDD Intrusion Detection', layout='wide')
st.title('AI-Based Network Intrusion Detection')
st.caption('NSL-KDD analysis using K-Means centroid distances and Naïve Bayes')

section = st.sidebar.radio(
    'Section',
    ['Overview', 'Data Analysis', 'K-Means', 'Model Evaluation', 'Prediction'],
)
metrics = read_metrics()

if section == 'Overview':
    proposed = metrics.get('K-Means + Naïve Bayes (Proposed)', {})
    try:
        train_df, test_df = get_datasets()
        records = f'{len(train_df) + len(test_df):,}'
    except (FileNotFoundError, ValueError) as error:
        records = 'Unavailable'
        st.info(f'Dataset not loaded: {error}')

    columns = st.columns(4)
    columns[0].metric('Dataset records', records)
    columns[1].metric('Network features', len(FEATURE_NAMES))
    columns[2].metric('Traffic classes', len(CLASS_NAMES))
    columns[3].metric(
        'Proposed model accuracy',
        f"{proposed['accuracy'] * 100:.2f}%" if 'accuracy' in proposed else 'Not trained',
    )
    st.subheader('Analysis pipeline')
    st.write(
        'Train-only preprocessing → K-Means (K=5) → five Euclidean centroid distances '
        '→ Gaussian Naïve Bayes.'
    )
    st.caption('Cluster IDs are arbitrary partitions; they are not attack-class labels.')

elif section == 'Data Analysis':
    try:
        train_df, test_df = get_datasets()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.stop()

    train_counts = train_df['label_5class'].value_counts().reindex(CLASS_NAMES, fill_value=0)
    attack_counts = train_df['attack_type'].value_counts().head(20).rename_axis('attack_type')
    left, right = st.columns(2)
    left.plotly_chart(
        px.bar(train_counts.rename_axis('class').reset_index(name='records'), x='class', y='records',
               title='Training records by class', color='class'),
        use_container_width=True,
    )
    right.plotly_chart(
        px.bar(attack_counts.reset_index(name='records'), x='attack_type', y='records',
               title='Most frequent training attack labels'),
        use_container_width=True,
    )
    st.subheader('Dataset summary')
    st.write(f'Train: {len(train_df):,} rows · Test: {len(test_df):,} rows')
    st.write(f'Missing values in training data: {int(train_df.isna().sum().sum()):,}')
    st.write(f'Duplicate training rows: {int(train_df.duplicated().sum()):,}')
    st.dataframe(train_df[FEATURE_NAMES].describe().T, use_container_width=True)

elif section == 'K-Means':
    cluster_file = OUTPUTS_DIR / 'cluster_sizes.csv'
    if cluster_file.is_file():
        cluster_sizes = pd.read_csv(cluster_file)
        st.subheader('Training cluster sizes')
        st.plotly_chart(
            px.bar(cluster_sizes, x='cluster_id', y='sample_count',
                   title='Training records assigned to each cluster'),
            use_container_width=True,
        )
    else:
        st.info('Cluster counts are not available. Run `python train.py` first.')

    try:
        train_df, _ = get_datasets()
        artifacts = get_artifacts()
        sample = train_df.sample(n=min(5000, len(train_df)), random_state=42)
        transformed = artifacts['preprocessor'].transform(sample)
        clusters = artifacts['kmeans'].predict(transformed)
        coordinates = PCA(n_components=2, random_state=42).fit_transform(transformed)
        pca_frame = pd.DataFrame({
            'PC1': coordinates[:, 0],
            'PC2': coordinates[:, 1],
            'Cluster': clusters.astype(str),
            'Class': sample['label_5class'].to_numpy(),
        })
        st.subheader('PCA view of preprocessed training records')
        st.plotly_chart(
            px.scatter(pca_frame, x='PC1', y='PC2', color='Cluster', hover_data=['Class'],
                       title='PCA projection colored by nearest K-Means cluster', opacity=0.65),
            use_container_width=True,
        )
        st.caption('PCA is a visualization only; clustering uses the full preprocessed feature space.')
    except (FileNotFoundError, ValueError) as error:
        st.info(f'PCA visualization unavailable: {error}')

elif section == 'Model Evaluation':
    comparison_path = OUTPUTS_DIR / 'model_comparison.csv'
    if comparison_path.is_file():
        comparison = pd.read_csv(comparison_path)
        st.subheader('Model comparison')
        st.dataframe(comparison, use_container_width=True, hide_index=True)
    else:
        st.info('Evaluation results are not available. Run `python train.py` first.')

    proposed = metrics.get('K-Means + Naïve Bayes (Proposed)')
    if proposed:
        metric_columns = st.columns(5)
        for column, title, key in zip(
            metric_columns,
            ['Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1 (macro)', 'False-positive rate'],
            ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'false_positive_rate'],
        ):
            column.metric(title, f"{proposed[key] * 100:.2f}%")
        st.subheader('Proposed model confusion matrix')
        confusion = pd.DataFrame(
            proposed['confusion_matrix'], index=CLASS_NAMES, columns=CLASS_NAMES
        )
        st.plotly_chart(
            px.imshow(confusion, text_auto=True, aspect='auto', color_continuous_scale='Blues',
                      labels={'x': 'Predicted class', 'y': 'Actual class', 'color': 'Records'}),
            use_container_width=True,
        )
        report = pd.DataFrame(proposed['per_class_report']).T
        st.subheader('Per-class classification report')
        st.dataframe(report, use_container_width=True)
        st.caption('Detection rate is attack recall; false-positive rate is normal traffic classified as an attack.')

elif section == 'Prediction':
    artifacts = load_dashboard_artifacts()
    st.write('Upload a CSV containing the 41 NSL-KDD features. Headerless NSL-KDD records are also accepted.')
    uploaded_file = st.file_uploader('Feature CSV', type=['csv'])
    if uploaded_file is not None and artifacts is not None:
        try:
            features = read_prediction_csv(uploaded_file)
            transformed = artifacts['preprocessor'].transform(features)
            distances = transform_distance_features(artifacts['kmeans'], transformed)
            predictions = artifacts['classifier'].predict(distances)
            results = pd.DataFrame({'predicted_class': predictions})
            st.dataframe(results, use_container_width=True, hide_index=True)
            st.download_button(
                'Download predictions',
                results.to_csv(index=False).encode('utf-8'),
                file_name='nsl_kdd_predictions.csv',
                mime='text/csv',
            )
        except (ValueError, KeyError, TypeError) as error:
            st.error(f'Could not process this file: {error}')