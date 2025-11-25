import streamlit as st
import pandas as pd
import io
import csv
import sys
from pathlib import Path
import shap
import matplotlib.pyplot as plt
from collections import OrderedDict

# Ensure project root is on path so src imports work when running streamlit from repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Try multiple import strategies for compatibility with different environments
try:
    # First try: absolute import from src package (works when src is in PYTHONPATH)
    from src.visualization.eda_plots import histogram, correlation_heatmap, scatter
    from src.models.predict import predict_from_dict, load_model
except ModuleNotFoundError:
    # Second try: add src to path and import directly
    SRC_DIR = Path(__file__).resolve().parent.parent
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from visualization.eda_plots import histogram, correlation_heatmap, scatter
    from models.predict import predict_from_dict, load_model


TARGET = 'target'

# Try to infer features dynamically from processed CSV if available
def infer_features(processed_path='data/processed/heart_disease_processed.csv'):
    p = Path(processed_path)
    if not p.exists():
        # fallback to reasonable defaults
        return ['age','sex','cp','trestbps','chol','fbs','restecg','thalch','exang','oldpeak','slope','ca','thal']
    dfp = pd.read_csv(p, nrows=10)
    # drop id, dataset and target if present
    drop = [c for c in ['id','dataset', TARGET] if c in dfp.columns]
    features = [c for c in dfp.columns if c not in drop]
    return features

# Use a reduced, user-friendly feature map for the prediction form.
# Keys are the model feature names; values are user-facing labels shown in the UI.
FEATURE_MAP = OrderedDict([
    ('age', 'Age'),
    ('sex', 'Sex'),
    ('cp', 'Chest Pain Type'),
    ('trestbps', 'Resting BP (mm Hg)'),
    ('chol', 'Serum Cholesterol (mg/dl)'),
    ('thalch', 'Max Heart Rate Achieved'),
    ('exang', 'Exercise-Induced Angina'),
    ('oldpeak', 'ST Depression (oldpeak)')
])

# Fallback features (keeps backward compatibility if processed CSV missing)
FALLBACK_FEATURES = list(FEATURE_MAP.keys())

st.set_page_config(page_title='Heart Disease Dashboard', layout='wide')

st.title('Heart Disease Analytics Dashboard')

# Sidebar: dataset upload and model status
# Top bar: uploader and model status (acts like a navbar)
model_status = 'Not loaded'
model = None

try:
    model = load_model()  # default path
    model_status = 'Loaded'
except FileNotFoundError:
    # Model doesn't exist - show option to train it
    model_status = 'Not found - needs training'
    st.warning('Model not found. The model needs to be trained before making predictions.')
    
    if st.button('Train Model Now'):
        with st.spinner('Training model... This may take a minute.'):
            try:
                # Import and run training
                from src.models.train_model import train
                
                # Check if raw data exists
                raw_data_path = PROJECT_ROOT / 'data' / 'raw' / 'heart_disease_uci.csv'
                if not raw_data_path.exists():
                    st.error(f'Training data not found at {raw_data_path}. Please upload the dataset.')
                else:
                    # Train the model
                    output_dir = PROJECT_ROOT / 'models'
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    metrics = train(
                        input_csv=str(raw_data_path),
                        output_dir=str(output_dir),
                        test_size=0.2,
                        random_state=42,
                        n_estimators=150
                    )
                    
                    st.success(f'Model trained successfully! Accuracy: {metrics.get("accuracy", "N/A"):.3f}')
                    st.info('Please refresh the page to load the trained model.')
                    
            except Exception as e:
                st.error(f'Training failed: {e}')
                st.exception(e)
except Exception as e:
    model = None
    model_status = f'Error: {str(e)[:100]}'

# Initialize session state for predictions and SHAP
if 'df_with_predictions' not in st.session_state:
    st.session_state.df_with_predictions = None
if 'pred_input_df' not in st.session_state:
    st.session_state.pred_input_df = None
if 'X_transformed_for_shap' not in st.session_state:
    st.session_state.X_transformed_for_shap = None
if 'shap_explainer' not in st.session_state:
    st.session_state.shap_explainer = None
if 'shap_values' not in st.session_state:
    st.session_state.shap_values = None

# Tabs: Upload, EDA, Model Insights, Predict
uploaded = None
df = None
upload_tab, eda_tab, insights_tab, predict_tab = st.tabs(['Upload', 'EDA', 'Model Insights', 'Predict'])

with upload_tab:
    uploaded = st.file_uploader('Upload Patient Data (CSV)', type=['csv'])
    st.write('Model status:', model_status)
    if uploaded:
        # Quick debug info to help with malformed/empty uploads
        try:
            name = uploaded.name
        except Exception:
            name = '<unknown>'
        try:
            size = uploaded.size
        except Exception:
            # Some UploadedFile implementations provide `buffer` or behave like io.BytesIO
            try:
                cur = uploaded.tell()
                uploaded.seek(0, io.SEEK_END)
                size = uploaded.tell()
                uploaded.seek(cur)
            except Exception:
                size = None

        st.write(f'Uploaded file: **{name}**')
        if size is not None:
            st.write(f'File size: **{size}** bytes')
        else:
            st.write('File size: unknown')

        if st.checkbox('Show raw file preview (first 1KB)'):
            try:
                # Use getvalue when available (Streamlit's UploadedFile supports this)
                raw = None
                try:
                    raw = uploaded.getvalue()
                except Exception:
                    try:
                        uploaded.seek(0)
                        raw = uploaded.read(1024)
                    except Exception:
                        raw = b''
                if isinstance(raw, (bytes, bytearray)):
                    text = raw[:1024].decode('utf-8', errors='replace')
                else:
                    text = str(raw)[:1024]
                st.text_area('File preview (first 1KB)', value=text, height=200)
                # reset pointer
                try:
                    uploaded.seek(0)
                except Exception:
                    pass
            except Exception as e:
                st.write('Could not show preview:', e)

        # If file size is zero, show clear message
        if size == 0:
            st.error('File is empty (0 bytes). Please verify the file and upload again.')

        
        def _read_uploaded_file(upl):
            # Robust reading: try to sniff delimiter/header, then fallbacks.
            upl.seek(0)
            # Read a small sample for sniffing
            try:
                sample = upl.read(4096)
            except Exception:
                sample = b''
            # Reset pointer
            try:
                upl.seek(0)
            except Exception:
                pass

            if not sample or (isinstance(sample, (bytes, bytearray)) and len(sample.strip()) == 0):
                # empty upload
                raise pd.errors.EmptyDataError('Uploaded file appears empty')

            # Ensure we have a text sample for csv.Sniffer
            try:
                text = sample.decode('utf-8', errors='replace') if isinstance(sample, (bytes, bytearray)) else str(sample)
            except Exception:
                text = str(sample)

            # Try to sniff delimiter
            delimiter = None
            try:
                dialect = csv.Sniffer().sniff(text)
                delimiter = dialect.delimiter
            except Exception:
                delimiter = None

            # Keep the first exception to show if all attempts fail
            first_exc = None

            # 1) Try with detected delimiter
            if delimiter:
                try:
                    upl.seek(0)
                    return pd.read_csv(upl, sep=delimiter)
                except Exception as e:
                    first_exc = first_exc or e

            # 2) Try default comma
            try:
                upl.seek(0)
                return pd.read_csv(upl)
            except Exception as e2:
                first_exc = first_exc or e2

            # 3) Try semicolon
            try:
                upl.seek(0)
                return pd.read_csv(upl, sep=';')
            except Exception as e3:
                first_exc = first_exc or e3

            # 4) Try Excel
            try:
                upl.seek(0)
                return pd.read_excel(upl)
            except Exception as e4:
                first_exc = first_exc or e4

            # 5) Try headerless CSV and attempt to assign expected feature names if counts match
            try:
                upl.seek(0)
                df_no_header = pd.read_csv(upl, header=None)
                expected = infer_features()
                if df_no_header.shape[1] == len(expected):
                    df_no_header.columns = expected
                return df_no_header
            except Exception:
                # Give up — raise the first meaningful exception
                raise first_exc if first_exc is not None else pd.errors.EmptyDataError('Could not parse uploaded file')

        try:
            df = _read_uploaded_file(uploaded)
            st.success('Dataset uploaded')
            st.write(f'Rows: {len(df)} | Columns: {len(df.columns)}')
            st.subheader('Preview')
            st.dataframe(df.head(10))

            # --- File insights ---
            st.subheader('File insights')

            # Duplicates
            dup_count = int(df.duplicated().sum())
            st.write(f'**Duplicate rows:** {dup_count}')

            # Numeric summary
            num_df = df.select_dtypes(include=['number'])
            if not num_df.empty:
                st.markdown('**Numeric summary (first 10 columns shown)**')
                desc = num_df.describe().T
                # show common stats
                stats = desc[['count', 'mean', 'std', 'min', '50%', 'max']].rename(columns={'50%': 'median'})
                st.dataframe(stats.head(50))

            # Categorical columns: show top value counts
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if cat_cols:
                st.markdown('**Categorical columns (top values)**')
                for c in cat_cols:
                    vc = df[c].value_counts(dropna=False).head(10)
                    st.write(f'- **{c}** — {df[c].nunique()} unique values')
                    try:
                        st.bar_chart(vc)
                    except Exception:
                        st.table(vc)

            # Target presence hint
            if TARGET not in df.columns and 'num' not in df.columns:
                st.info('No target column found in dataset. Predictions can be generated using the Predict tab.')

            # If a model is available, run batch predictions and show group-level insights
            if model is not None:
                try:
                    st.markdown('---')
                    st.header('Batch Predictions')
                    req_feats = infer_features()
                    # prepare sample_df for fill values
                    sample_df_local = None
                    try:
                        sample_df_local = pd.read_csv('data/processed/heart_disease_processed.csv', nrows=200)
                    except Exception:
                        sample_df_local = None

                    prepared = []
                    for _, row in df.iterrows():
                        row_inputs = {}
                        for feat in req_feats:
                            if feat in row and pd.notna(row[feat]):
                                row_inputs[feat] = row[feat]
                            else:
                                if sample_df_local is not None and feat in sample_df_local.columns:
                                    col = sample_df_local[feat].dropna()
                                    if len(col) == 0:
                                        row_inputs[feat] = 0.0
                                    elif pd.api.types.is_numeric_dtype(col):
                                        row_inputs[feat] = float(col.median())
                                    else:
                                        try:
                                            row_inputs[feat] = col.mode().iloc[0]
                                        except Exception:
                                            row_inputs[feat] = col.iloc[0]
                                else:
                                    row_inputs[feat] = 0.0
                        prepared.append(row_inputs)

                    pred_input_df = pd.DataFrame(prepared)
                    try:
                        preds = model.predict(pred_input_df)
                    except Exception as e:
                        st.error(f'Batch prediction failed: {e}')
                        preds = None

                    if preds is not None:
                        df_pred = df.reset_index(drop=True).copy()
                        df_pred['predicted'] = preds
                        df_pred['diagnosis'] = df_pred['predicted'].map({1: 'Heart disease detected', 0: 'No heart disease detected'})
                        
                        # Store in session state for use in other tabs
                        st.session_state.df_with_predictions = df_pred
                        # Store the prepared input (what was actually used for predictions)
                        st.session_state.pred_input_df = pred_input_df
                        
                        # Compute SHAP values for interpretability
                        try:
                            # Transform data using the preprocessor
                            X_transformed = model.named_steps['preprocessor'].transform(pred_input_df)
                            # Store transformed data for visualization
                            st.session_state.X_transformed_for_shap = X_transformed
                            
                            # Use TreeExplainer for tree-based models (RandomForest)
                            if st.session_state.shap_explainer is None:
                                st.session_state.shap_explainer = shap.TreeExplainer(model.named_steps['classifier'])
                            
                            # Compute SHAP values - for binary classification, this returns a list [class_0, class_1]
                            # Each element has shape (n_samples, n_features)
                            shap_values_raw = st.session_state.shap_explainer.shap_values(X_transformed)
                            
                            # For binary classification, extract SHAP values for positive class (class 1)
                            if isinstance(shap_values_raw, list) and len(shap_values_raw) == 2:
                                st.session_state.shap_values = shap_values_raw[1]  # Positive class
                            else:
                                st.session_state.shap_values = shap_values_raw
                                
                        except Exception as e:
                            st.warning(f'Could not compute SHAP values: {e}')
                            st.session_state.shap_values = None
                            st.session_state.X_transformed_for_shap = None

                        pos = df_pred[df_pred['predicted'] == 1]
                        neg = df_pred[df_pred['predicted'] == 0]
                        st.write(f'Disease detected: **{len(pos)}** patients')
                        st.write(f'No disease: **{len(neg)}** patients')

                        # Export predictions
                        try:
                            csv_bytes = df_pred.to_csv(index=False).encode('utf-8')
                            st.download_button('Export Results', data=csv_bytes, file_name='heart_disease_predictions.csv', mime='text/csv')
                        except Exception:
                            pass

                        # Show examples
                        with st.expander('Sample Results: Disease Detected'):
                            st.dataframe(pos.head(10))
                        with st.expander('Sample Results: No Disease'):
                            st.dataframe(neg.head(10))

                        # Group-level numeric insights: compare group means to population medians
                        def group_numeric_insights(group_df, population_df, features, top_n=5):
                            insights = []
                            for feat in features:
                                try:
                                    if feat not in group_df.columns:
                                        continue
                                    grp_vals = pd.to_numeric(group_df[feat], errors='coerce').dropna()
                                    if grp_vals.empty:
                                        continue
                                    if population_df is not None and feat in population_df.columns:
                                        pop_median = float(pd.to_numeric(population_df[feat], errors='coerce').median())
                                    else:
                                        pop_median = float(pd.to_numeric(df[feat], errors='coerce').median())
                                    grp_mean = float(grp_vals.mean())
                                    diff = grp_mean - pop_median
                                    insights.append((feat, abs(diff), diff, grp_mean, pop_median))
                                except Exception:
                                    continue
                            insights = sorted(insights, key=lambda x: x[1], reverse=True)
                            return insights[:top_n]

                        numeric_feats = [f for f in req_feats if (sample_df_local is None or f in sample_df_local.columns) or pd.api.types.is_numeric_dtype(df.get(f))]

                        st.subheader('Group insights — Predicted positives')
                        if not pos.empty:
                            pos_ins = group_numeric_insights(pos, sample_df_local, numeric_feats, top_n=6)
                            if pos_ins:
                                for feat, score, diff, grp_mean, pop_med in pos_ins:
                                    direction = 'higher' if diff > 0 else 'lower'
                                    st.write(f'- **{feat}**: group mean {grp_mean:.2f} is {abs(diff):.2f} {direction} than population median {pop_med:.2f}')
                            else:
                                st.write('No clear numeric deviations found for predicted positives.')
                        else:
                            st.write('No predicted positives in this dataset.')

                        st.subheader('Group insights — Predicted negatives')
                        if not neg.empty:
                            neg_ins = group_numeric_insights(neg, sample_df_local, numeric_feats, top_n=6)
                            if neg_ins:
                                for feat, score, diff, grp_mean, pop_med in neg_ins:
                                    direction = 'higher' if diff > 0 else 'lower'
                                    st.write(f'- **{feat}**: group mean {grp_mean:.2f} is {abs(diff):.2f} {direction} than population median {pop_med:.2f}')
                            else:
                                st.write('No clear numeric deviations found for predicted negatives.')
                        else:
                            st.write('No predicted negatives in this dataset.')
                except Exception as e:
                    st.warning(f'Automatic batch prediction failed: {e}')
        except Exception as e:
            st.warning('Uploaded file could not be read as CSV. Try saving as a UTF-8 encoded CSV with a header row, or upload an Excel (.xlsx) file.')
            st.info('Debug: pandas error message follows (useful for fixing file format)')
            st.exception(e)

# EDA
with eda_tab:
    st.header('Exploratory Data Analysis')
    
    # Use df with predictions from session state if available
    df_for_eda = st.session_state.df_with_predictions if st.session_state.df_with_predictions is not None else df
    
    # Check if we have predictions available
    has_predictions = df_for_eda is not None and 'predicted' in df_for_eda.columns
    
    if has_predictions:
        chart_type = st.selectbox('Chart type', options=[
            'Histogram', 
            'Scatter', 
            'Correlation heatmap',
            'Compare Groups - Histogram',
            'Compare Groups - Box Plot',
            'Compare Groups - Violin Plot'
        ])
    else:
        chart_type = st.selectbox('Chart type', options=['Histogram', 'Scatter', 'Correlation heatmap'])

    col1, col2 = st.columns([2,1])
    with col1:
        if df_for_eda is not None:
            if chart_type == 'Histogram':
                num_cols = df_for_eda.select_dtypes(include=['number']).columns.tolist()
                if not num_cols:
                    st.write('No numeric columns available for histogram in the uploaded CSV.')
                else:
                    col = st.selectbox('Choose numeric column for histogram', options=num_cols)
                    fig = histogram(df_for_eda, col)
                    st.plotly_chart(fig, width='stretch')
            elif chart_type == 'Scatter':
                num_cols = df_for_eda.select_dtypes(include=['number']).columns.tolist()
                if len(num_cols) < 2:
                    st.write('Need at least two numeric columns for a scatter plot.')
                else:
                    x_col = st.selectbox('X axis', options=num_cols, index=0)
                    y_col = st.selectbox('Y axis', options=num_cols, index=1)
                    fig = scatter(df_for_eda, x_col, y_col)
                    st.plotly_chart(fig, width='stretch')
            elif chart_type == 'Correlation heatmap':
                numeric_cols = df_for_eda.select_dtypes(include=['number']).columns.tolist()
                if len(numeric_cols) >= 2:
                    fig = correlation_heatmap(df_for_eda, numeric_cols)
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.write('Need at least two numeric columns for a correlation heatmap.')
            elif chart_type == 'Compare Groups - Histogram':
                import plotly.express as px
                num_cols = [c for c in df_for_eda.select_dtypes(include=['number']).columns.tolist() if c != 'predicted']
                if not num_cols:
                    st.write('No numeric columns available.')
                else:
                    col = st.selectbox('Choose numeric column to compare', options=num_cols)
                    fig = px.histogram(df_for_eda, x=col, color='predicted', barmode='overlay',
                                     labels={'predicted': 'Prediction'},
                                     color_discrete_map={0: 'lightblue', 1: 'salmon'},
                                     title=f'{col} Distribution by Prediction Group')
                    fig.update_layout(legend_title_text='Group', 
                                    legend=dict(orientation='h', y=1.1))
                    st.plotly_chart(fig, width='stretch')
            elif chart_type == 'Compare Groups - Box Plot':
                import plotly.express as px
                num_cols = [c for c in df_for_eda.select_dtypes(include=['number']).columns.tolist() if c != 'predicted']
                if not num_cols:
                    st.write('No numeric columns available.')
                else:
                    col = st.selectbox('Choose numeric column to compare', options=num_cols)
                    df_box = df_for_eda.copy()
                    df_box['Group'] = df_box['predicted'].map({0: 'Predicted Negative', 1: 'Predicted Positive'})
                    fig = px.box(df_box, x='Group', y=col, color='Group',
                               color_discrete_map={'Predicted Negative': 'lightblue', 'Predicted Positive': 'salmon'},
                               title=f'{col} Distribution by Prediction Group')
                    st.plotly_chart(fig, width='stretch')
            elif chart_type == 'Compare Groups - Violin Plot':
                import plotly.express as px
                num_cols = [c for c in df_for_eda.select_dtypes(include=['number']).columns.tolist() if c != 'predicted']
                if not num_cols:
                    st.write('No numeric columns available.')
                else:
                    col = st.selectbox('Choose numeric column to compare', options=num_cols)
                    df_violin = df_for_eda.copy()
                    df_violin['Group'] = df_violin['predicted'].map({0: 'Predicted Negative', 1: 'Predicted Positive'})
                    fig = px.violin(df_violin, x='Group', y=col, color='Group', box=True,
                                  color_discrete_map={'Predicted Negative': 'lightblue', 'Predicted Positive': 'salmon'},
                                  title=f'{col} Distribution by Prediction Group')
                    st.plotly_chart(fig, width='stretch')
        else:
            st.write('Upload a CSV to see EDA charts.')

    with col2:
        if df_for_eda is not None:
            st.write(f"Rows: {len(df_for_eda)} | Columns: {len(df_for_eda.columns)}")
            if has_predictions:
                pos_count = int((df_for_eda['predicted'] == 1).sum())
                neg_count = int((df_for_eda['predicted'] == 0).sum())
                st.write(f"Predicted Positive: {pos_count}")
                st.write(f"Predicted Negative: {neg_count}")
        else:
            st.write('Upload a CSV to see dataset info.')

# Model Insights tab - SHAP explanations
with insights_tab:
    st.header('Model Insights & Explainability')
    
    # Add button to clear cached SHAP values
    if st.button('Reset Analysis Cache'):
        st.session_state.shap_explainer = None
        st.session_state.shap_values = None
        st.session_state.X_transformed_for_shap = None
        st.session_state.df_with_predictions = None
        st.session_state.pred_input_df = None
        st.success('Analysis cache reset. Please re-upload your data in the Upload tab.')
        st.stop()
    
    if model is None:
        st.warning('Model not loaded. Train and save a model first.')
    elif st.session_state.shap_values is None:
        st.info('Upload a CSV in the Upload tab to generate SHAP explanations.')
    else:
        st.subheader('SHAP Feature Importance')
        st.write('SHAP (SHapley Additive exPlanations) values show how each feature contributes to predictions.')
        
        try:
            # Get feature names from the preprocessor
            preprocessor = model.named_steps['preprocessor']
            try:
                feature_names = list(preprocessor.get_feature_names_out())
            except Exception:
                feature_names = [f'Feature {i}' for i in range(st.session_state.shap_values[0].shape[1])]
            
            # SHAP values are now directly the array for positive class (already extracted during computation)
            shap_vals = st.session_state.shap_values
            
            # Get transformed features (use the exact same data that was used for SHAP computation)
            if 'X_transformed_for_shap' in st.session_state and st.session_state.X_transformed_for_shap is not None:
                X_transformed = st.session_state.X_transformed_for_shap
            elif 'pred_input_df' in st.session_state and st.session_state.pred_input_df is not None:
                X_transformed = model.named_steps['preprocessor'].transform(st.session_state.pred_input_df)
            else:
                X_transformed = model.named_steps['preprocessor'].transform(
                    st.session_state.df_with_predictions.drop(columns=['predicted', 'diagnosis'], errors='ignore')
                )
            
            # Individual patient explanations
            st.subheader('Patient-Level Analysis')
            st.write('Analyze feature contributions for individual predictions.')
            
            patient_idx = st.selectbox('Select Patient', options=list(range(len(st.session_state.df_with_predictions))))
            
            if patient_idx is not None:
                patient_row = st.session_state.df_with_predictions.iloc[patient_idx]
                prediction = int(patient_row['predicted'])
                diagnosis = patient_row['diagnosis']
                
                st.write(f'**Patient ID: {patient_idx}** | **Result:** {diagnosis}')
                
                # Show patient data
                with st.expander('View Patient Data'):
                    patient_data = patient_row.drop(['predicted', 'diagnosis'], errors='ignore')
                    st.dataframe(patient_data.to_frame().T)
                
                # SHAP waterfall plot for this patient
                st.write('**Feature Contribution Analysis**')
                # Use the stored transformed data
                if 'X_transformed_for_shap' in st.session_state and st.session_state.X_transformed_for_shap is not None:
                    X_transformed = st.session_state.X_transformed_for_shap
                elif 'pred_input_df' in st.session_state and st.session_state.pred_input_df is not None:
                    X_transformed = model.named_steps['preprocessor'].transform(st.session_state.pred_input_df)
                else:
                    X_transformed = model.named_steps['preprocessor'].transform(
                        st.session_state.df_with_predictions.drop(columns=['predicted', 'diagnosis'], errors='ignore')
                    )
                
                # Get SHAP values for this patient (already extracted for positive class during computation)
                patient_shap = shap_vals[patient_idx]
                
                # Debug
                st.write(f"🔍 Patient SHAP shape: {patient_shap.shape}")
                
                # If still 2D (has 2 classes), extract positive class
                if len(patient_shap.shape) > 1 and patient_shap.shape[-1] == 2:
                    patient_shap = patient_shap[:, 1]  # Take positive class
                    st.write(f"🔍 After extraction, patient SHAP shape: {patient_shap.shape}")
                
                # Get base value for positive class (ensure it's a scalar)
                base_val = st.session_state.shap_explainer.expected_value
                if isinstance(base_val, list):
                    base_val = base_val[1]
                # Convert to scalar if it's an array
                if hasattr(base_val, 'shape') and len(base_val.shape) > 0:
                    base_val = float(base_val[0]) if base_val.shape[0] > 0 else float(base_val)
                else:
                    base_val = float(base_val)
                
                # Create explanation object for waterfall plot
                explanation = shap.Explanation(
                    values=patient_shap,
                    base_values=base_val,
                    data=X_transformed[patient_idx],
                    feature_names=feature_names
                )
                
                fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 8))
                shap.waterfall_plot(explanation, show=False)
                st.pyplot(fig_waterfall)
                plt.close()
                
                # Force plot
                st.write('**Prediction Decomposition**')
                try:
                    force_plot = shap.force_plot(
                        base_val,
                        patient_shap,
                        X_transformed[patient_idx],
                        feature_names=feature_names,
                        matplotlib=True,
                        show=False
                    )
                    st.pyplot(force_plot)
                    plt.close()
                except Exception:
                    st.info('Force plot visualization not available.')
                
        except Exception as e:
            st.error(f'Error generating SHAP visualizations: {e}')
            st.exception(e)

with predict_tab:
    st.header('Predict Heart Disease')
    st.write('Enter patient values')
    inputs = {}
    # Load a small sample of processed data to get value lists for categorical fields
    sample_df = None
    try:
        sample_df = pd.read_csv('data/processed/heart_disease_processed.csv', nrows=200)
    except Exception:
        sample_df = None

    cols = st.columns(3)
    # Render a reduced, friendly form using FEATURE_MAP (keeps internal keys intact)
    for i, (feat, label) in enumerate(FEATURE_MAP.items()):
        c = cols[i % 3]
        # If we have a sample processed CSV and the feature is categorical, show selectbox
        if sample_df is not None and feat in sample_df.columns and not pd.api.types.is_numeric_dtype(sample_df[feat]):
            options = sample_df[feat].dropna().unique().tolist()
            # prepend empty option so users must choose deliberately
            options = [''] + list(options)
            try:
                val = c.selectbox(label, options=options, index=0)
            except Exception:
                val = c.text_input(label, value='')
        else:
            # Numeric input: leave empty (text input) so no random defaults are shown
            val = c.text_input(label, value='')
        inputs[feat] = val

    submitted = st.button('Predict')

    if submitted:
        if model is None:
            st.error('Model not available. Train and save the model first (see README).')
        else:
            if model is None:
                st.error('Model not available. Train and save the model first (see README).')
            else:
                # Build a complete feature dict matching the processed CSV / model expectations.
                required_features = infer_features()

                # Convert reduced inputs into typed values (try numeric) and then fill any missing features
                typed_inputs = {}
                for k, v in inputs.items():
                    if isinstance(v, (int, float)):
                        typed_inputs[k] = v
                    else:
                        try:
                            typed_inputs[k] = float(v)
                        except Exception:
                            typed_inputs[k] = v

                full_inputs = {}
                for feat in required_features:
                    if feat in typed_inputs and typed_inputs[feat] != '':
                        full_inputs[feat] = typed_inputs[feat]
                    else:
                        # Fill from sample_df if available
                        if sample_df is not None and feat in sample_df.columns:
                            col = sample_df[feat].dropna()
                            if len(col) == 0:
                                # empty column in sample, fallback to 0/empty
                                full_inputs[feat] = 0.0
                            elif pd.api.types.is_numeric_dtype(col):
                                try:
                                    full_inputs[feat] = float(col.median())
                                except Exception:
                                    full_inputs[feat] = float(col.iloc[0])
                            else:
                                # categorical: use mode
                                try:
                                    full_inputs[feat] = col.mode().iloc[0]
                                except Exception:
                                    full_inputs[feat] = col.iloc[0]
                        else:
                            # Last-resort defaults
                            full_inputs[feat] = 0.0

                # Convert any string numeric inputs to numeric where possible
                parsed_inputs = {}
                for k, v in full_inputs.items():
                    if isinstance(v, str):
                        try:
                            parsed_inputs[k] = float(v)
                        except Exception:
                            parsed_inputs[k] = v
                    else:
                        parsed_inputs[k] = v

                result = predict_from_dict(parsed_inputs)
                # Show cautious, human-friendly diagnosis text instead of raw numeric prediction and confidence
                if result.get('prediction') == 1:
                    st.error('There is a high chance that you have Heart disease.')
                else:
                    st.success('There is a low chance that you have Heart disease.')

                # Provide human-readable insights using processed data (no model metrics shown)
                def analyze_insights(sample: pd.DataFrame, inputs_map: dict):
                    # Returns only contributing and protective lists. If sample is empty or missing,
                    # return empty lists (no notes displayed).
                    insights = {'contributing': [], 'protective': []}
                    if sample is None or sample.empty:
                        return insights

                    # Determine which column is the target
                    target_col = None
                    for tc in ['target', 'num']:
                        if tc in sample.columns:
                            target_col = tc
                            break
                    if target_col is None:
                        return insights

                    overall_rate = float(sample[target_col].dropna().astype(float).mean())

                    for feat, val in inputs_map.items():
                        if feat not in sample.columns:
                            continue
                        col = sample[[feat, target_col]].dropna()
                        if col.empty:
                            continue
                        # Numeric feature: use correlation and median to judge direction
                        if pd.api.types.is_numeric_dtype(col[feat]):
                            try:
                                corr = float(col[feat].astype(float).corr(col[target_col].astype(float)))
                            except Exception:
                                corr = 0.0
                            median = float(col[feat].median())
                            try:
                                user_val = float(val)
                            except Exception:
                                # couldn't parse user value: skip
                                continue
                            deviation = user_val - median
                            score = abs(corr) * abs(deviation)
                            if corr > 0 and deviation > 0:
                                insights['contributing'].append((feat, score, f'value {user_val} > median {median} (corr {corr:.2f})'))
                            elif corr < 0 and deviation < 0:
                                insights['contributing'].append((feat, score, f'value {user_val} < median {median} (corr {corr:.2f})'))
                            else:
                                insights['protective'].append((feat, score, f'value {user_val} vs median {median} (corr {corr:.2f})'))
                        else:
                            # Categorical: compute disease rate per category
                            try:
                                rates = col.groupby(feat)[target_col].mean()
                                cat = val
                                if pd.isna(cat) or cat == '':
                                    continue
                                cat_rate = float(rates.get(cat, overall_rate))
                                score = abs(cat_rate - overall_rate)
                                if cat_rate > overall_rate:
                                    insights['contributing'].append((feat, score, f'category "{cat}" has disease rate {cat_rate:.2f} > overall {overall_rate:.2f}'))
                                else:
                                    insights['protective'].append((feat, score, f'category "{cat}" has disease rate {cat_rate:.2f} <= overall {overall_rate:.2f}'))
                            except Exception:
                                continue

                    # Sort by score desc and return top items
                    insights['contributing'] = sorted(insights['contributing'], key=lambda x: x[1], reverse=True)
                    insights['protective'] = sorted(insights['protective'], key=lambda x: x[1], reverse=True)
                    return insights

                # Use the exact inputs that were passed to prediction for insights
                insights = analyze_insights(sample_df if sample_df is not None else pd.DataFrame(), parsed_inputs)
                st.header('Personalized Insights')
                if result['prediction'] == 1:
                    st.subheader('Factors indicating higher disease risk')
                    if insights['contributing']:
                        for feat, score, msg in insights['contributing'][:6]:
                            st.write(f"- **{feat}**: {msg}")
                    else:
                        st.write('No clear contributing factors found from your provided inputs.')
                else:
                    st.subheader('Protective factors / things you are doing well')
                    if insights['protective']:
                        for feat, score, msg in insights['protective'][:6]:
                            st.write(f"- **{feat}**: {msg}")
                    else:
                        st.write('No clear protective factors found from your provided inputs.')

                # Batch prediction on uploaded CSV: predict disease for each row and show examples
                if uploaded is not None and model is not None:
                    if st.button('Predict on uploaded CSV'):
                        try:
                            uploaded_df = pd.read_csv(uploaded)
                        except Exception:
                            # If uploaded is already a DataFrame or fails, fall back
                            uploaded_df = df if df is not None else pd.DataFrame()

                        if uploaded_df is None or uploaded_df.empty:
                            st.warning('Uploaded CSV could not be read or is empty.')
                        else:
                            req_feats = infer_features()
                            # prepare sample_df for fill values
                            sample_df_local = None
                            try:
                                sample_df_local = pd.read_csv('data/processed/heart_disease_processed.csv', nrows=200)
                            except Exception:
                                sample_df_local = None

                            prepared = []
                            for _, row in uploaded_df.iterrows():
                                row_inputs = {}
                                for feat in req_feats:
                                    if feat in row and pd.notna(row[feat]):
                                        row_inputs[feat] = row[feat]
                                    else:
                                        if sample_df_local is not None and feat in sample_df_local.columns:
                                            col = sample_df_local[feat].dropna()
                                            if len(col) == 0:
                                                row_inputs[feat] = 0.0
                                            elif pd.api.types.is_numeric_dtype(col):
                                                row_inputs[feat] = float(col.median())
                                            else:
                                                try:
                                                    row_inputs[feat] = col.mode().iloc[0]
                                                except Exception:
                                                    row_inputs[feat] = col.iloc[0]
                                        else:
                                            row_inputs[feat] = 0.0
                                prepared.append(row_inputs)

                            pred_input_df = pd.DataFrame(prepared)
                            try:
                                preds = model.predict(pred_input_df)
                            except Exception as e:
                                st.error(f'Batch prediction failed: {e}')
                                preds = None

                            if preds is not None:
                                uploaded_df = uploaded_df.reset_index(drop=True)
                                uploaded_df['predicted'] = preds

                                # Add a human-readable diagnosis message per row
                                uploaded_df['diagnosis'] = uploaded_df['predicted'].map({
                                    1: 'There is a high chance that this patient has Heart disease.',
                                    0: 'There is a low chance that this patient has Heart disease.'
                                })

                                pos = uploaded_df[uploaded_df['predicted'] == 1]
                                neg = uploaded_df[uploaded_df['predicted'] == 0]
                                st.markdown('**Batch prediction results**')
                                st.write(f'Predicted positive (disease): {len(pos)}')
                                st.write(f'Predicted negative (no disease): {len(neg)}')

                                # Allow downloading the full predictions as CSV
                                try:
                                    csv_bytes = uploaded_df.to_csv(index=False).encode('utf-8')
                                    st.download_button('Download predictions CSV', data=csv_bytes, file_name='predictions.csv', mime='text/csv')
                                except Exception:
                                    pass

                                st.subheader('Examples — Predicted positive')
                                st.dataframe(pos.head(10))
                                st.subheader('Examples — Predicted negative')
                                st.dataframe(neg.head(10))

                                # For the top predicted positives, show a per-patient insight summary
                                if not pos.empty and sample_df_local is not None:
                                    st.subheader('Per-patient insights (predicted positive)')
                                    for idx, row in pos.head(3).iterrows():
                                        with st.expander(f'Patient index {idx} — diagnosis: {row.get("diagnosis", "")}', expanded=False):
                                            # Build a dict of the required features for this row
                                            inputs_map = {feat: (row.get(feat) if feat in row.index else None) for feat in req_feats}
                                            row_insights = analyze_insights(sample_df_local, inputs_map)
                                            st.write('**Diagnosis**: ', row.get('diagnosis', ''))
                                            if row_insights['contributing']:
                                                st.write('Contributing factors:')
                                                for feat, score, msg in row_insights['contributing'][:6]:
                                                    st.write(f'- **{feat}**: {msg}')
                                            else:
                                                st.write('No clear contributing factors found from available data.')


st.markdown('---')
