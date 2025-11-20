import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Ensure project root is on path so src imports work when running streamlit from repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.visualization.eda_plots import histogram, correlation_heatmap, scatter
from src.models.predict import predict_from_dict, load_model
from collections import OrderedDict


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
try:
    model = load_model()  # default path
    model_status = 'Loaded'
except Exception as e:
    model = None
    model_status = f'Not available: {e}'

# Tabs: Upload, EDA, Predict
uploaded = None
df = None
upload_tab, eda_tab, predict_tab = st.tabs(['Upload', 'EDA', 'Predict'])

with upload_tab:
    uploaded = st.file_uploader('Upload CSV (optional)', type=['csv'])
    st.write('Model status:', model_status)
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
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
                st.info('No target column (`target` or `num`) found in uploaded CSV — you can still run predictions from the Predict tab or use the batch-predict action.')
        except Exception:
            st.warning('Uploaded file could not be read as CSV.')

# EDA
with eda_tab:
    st.header('Exploratory Data Analysis')
    # Let the user choose which chart to view
    chart_type = st.selectbox('Chart type', options=['Histogram', 'Scatter', 'Correlation heatmap'])

    col1, col2 = st.columns([2,1])
    with col1:
        if df is not None:
            if chart_type == 'Histogram':
                num_cols = df.select_dtypes(include=['number']).columns.tolist()
                if not num_cols:
                    st.write('No numeric columns available for histogram in the uploaded CSV.')
                else:
                    col = st.selectbox('Choose numeric column for histogram', options=num_cols)
                    fig = histogram(df, col)
                    st.plotly_chart(fig, use_container_width=True)
            elif chart_type == 'Scatter':
                num_cols = df.select_dtypes(include=['number']).columns.tolist()
                if len(num_cols) < 2:
                    st.write('Need at least two numeric columns for a scatter plot.')
                else:
                    x_col = st.selectbox('X axis', options=num_cols, index=0)
                    y_col = st.selectbox('Y axis', options=num_cols, index=1)
                    fig = scatter(df, x_col, y_col)
                    st.plotly_chart(fig, use_container_width=True)
            else:  # Correlation heatmap
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if len(numeric_cols) >= 2:
                    fig = correlation_heatmap(df, numeric_cols)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write('Need at least two numeric columns for a correlation heatmap.')
        else:
            st.write('Upload a CSV to see EDA charts.')

    with col2:
        if df is not None:
            st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")
        else:
            st.write('Upload a CSV to see dataset info.')

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
                                pos = uploaded_df[uploaded_df['predicted'] == 1]
                                neg = uploaded_df[uploaded_df['predicted'] == 0]
                                st.markdown('**Batch prediction results**')
                                st.write(f'Predicted positive (disease): {len(pos)}')
                                st.write(f'Predicted negative (no disease): {len(neg)}')
                                st.subheader('Examples — Predicted positive')
                                st.dataframe(pos.head(10))
                                st.subheader('Examples — Predicted negative')
                                st.dataframe(neg.head(10))

st.markdown('---')
