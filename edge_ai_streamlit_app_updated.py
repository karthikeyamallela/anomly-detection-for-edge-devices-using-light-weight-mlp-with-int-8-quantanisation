import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import joblib

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

st.set_page_config(page_title="Edge AI Real-Time Anomaly Detector", page_icon="⚡", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "lightweight_model_int8.tflite")
KERAS_MODEL_PATHS = {
    "Lightweight MLP": os.path.join(BASE_DIR, "models", "lightweight_model.keras"),
    "Baseline MLP": os.path.join(BASE_DIR, "models", "baseline_model.keras"),
    "KronNet-inspired": os.path.join(BASE_DIR, "models", "kronnet_model.keras"),
}
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
IMPUTER_PATH = os.path.join(BASE_DIR, "models", "imputer.pkl")
X_TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "X_test.npy")
Y_TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "y_test.npy")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")

if "history" not in st.session_state:
    st.session_state.history = []
if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = None
if "selected_actual" not in st.session_state:
    st.session_state.selected_actual = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None

@st.cache_resource
def load_resources():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)
    return interpreter, input_details, output_details, scaler, imputer, X_test, y_test

@st.cache_resource
def get_keras_model(model_name):
    path = KERAS_MODEL_PATHS[model_name]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return tf.keras.models.load_model(path)

def predict_int8(sample, interpreter, input_details, output_details):
    """Run one already-preprocessed sample through the INT8 TFLite model."""
    input_detail = input_details[0]
    input_dtype = input_detail["dtype"]
    x = np.asarray(sample, dtype=np.float32).reshape(1, -1)

    # Quantize input only when the TFLite model expects integer input.
    if input_dtype in (np.int8, np.uint8):
        input_scale, input_zero_point = input_detail["quantization"]
        if input_scale > 0:
            x = np.round(x / input_scale + input_zero_point)
        limits = np.iinfo(input_dtype)
        x = np.clip(x, limits.min, limits.max).astype(input_dtype)
    else:
        x = x.astype(input_dtype)

    interpreter.set_tensor(input_detail["index"], x)
    start = time.perf_counter()
    interpreter.invoke()
    elapsed_ms = (time.perf_counter() - start) * 1000

    output_detail = output_details[0]
    raw_output = interpreter.get_tensor(output_detail["index"])
    raw_value = np.asarray(raw_output).reshape(-1)[0]  # extract one scalar

    output_dtype = output_detail["dtype"]
    if output_dtype in (np.int8, np.uint8):
        output_scale, output_zero_point = output_detail["quantization"]
        if output_scale > 0:
            probability = (float(raw_value) - float(output_zero_point)) * float(output_scale)
        else:
            probability = float(raw_value)
    else:
        probability = float(raw_value)

    probability = float(np.clip(probability, 0.0, 1.0))
    return int(probability >= 0.5), probability, elapsed_ms

def predict_selected_model(model_name, sample, interpreter, input_details, output_details, scaler, imputer):
    x = np.asarray(sample, dtype=np.float32).reshape(1, -1)
    if model_name == "Lightweight MLP (INT8 TFLite)":
        return predict_int8(sample, interpreter, input_details, output_details)
    model = get_keras_model(model_name)
    start = time.perf_counter()
    probability = float(model.predict(x, verbose=0).reshape(-1)[0])
    elapsed_ms = (time.perf_counter() - start) * 1000
    return int(probability >= 0.5), probability, elapsed_ms

def predict_all_models(sample, interpreter, input_details, output_details, scaler, imputer, actual_label=None):
    model_names = ["Lightweight MLP", "Lightweight MLP (INT8 TFLite)", "Baseline MLP", "KronNet-inspired"]
    results = []
    for name in model_names:
        pred, prob, elapsed = predict_selected_model(
            name, sample, interpreter, input_details, output_details, scaler, imputer
        )
        results.append({
            "Model": name,
            "Prediction": label_text(pred),
            "Anomaly Probability (%)": round(prob * 100, 4),
            "Confidence (%)": round((prob if pred == 1 else 1 - prob) * 100, 4),
            "Inference Time (ms)": round(elapsed, 4),
            "Actual Label": label_text(actual_label) if actual_label is not None else "Not provided",
            "Correct": ("Yes" if pred == int(actual_label) else "No") if actual_label is not None else "Not evaluated",
        })
    return pd.DataFrame(results)

def label_text(label):
    return "ANOMALY" if int(label) == 1 else "NORMAL"

def add_history(prediction, probability, inference_ms, actual=None):
    entry = {
        "Time": time.strftime("%H:%M:%S"),
        "Prediction": label_text(prediction),
        "Confidence": probability if prediction == 1 else 1 - probability,
        "Inference (ms)": inference_ms,
    }
    if actual is not None:
        entry["Actual"] = label_text(actual)
        entry["Correct"] = int(prediction) == int(actual)
    st.session_state.history.append(entry)
    st.session_state.history = st.session_state.history[-100:]

def figure_path(filename):
    path = os.path.join(FIGURES_DIR, filename)
    return path if os.path.exists(path) else None

try:
    interpreter, input_details, output_details, scaler, imputer, X_test, y_test = load_resources()
except Exception as e:
    st.error(f"Unable to load the Edge AI model/resources.\n\n{e}")
    st.stop()

INPUT_FEATURES = int(input_details[0]["shape"][-1])

FEATURE_NAMES_PATH = os.path.join(BASE_DIR, "data", "processed", "feature_names.json")
if os.path.exists(FEATURE_NAMES_PATH):
    import json
    with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
        FEATURE_NAMES = json.load(f)
elif hasattr(scaler, "feature_names_in_"):
    # Use the exact column names and order recorded by the fitted training scaler.
    FEATURE_NAMES = list(scaler.feature_names_in_)
else:
    FEATURE_NAMES = [f"Feature_{i+1}" for i in range(INPUT_FEATURES)]

if len(FEATURE_NAMES) != INPUT_FEATURES:
    raise ValueError(f"Expected {INPUT_FEATURES} feature names, found {len(FEATURE_NAMES)}.")

def preprocess_raw_features(raw_frame, scaler, imputer):
    if raw_frame.shape[1] != INPUT_FEATURES:
        raise ValueError(f"Expected {INPUT_FEATURES} features, received {raw_frame.shape[1]}.")
    numeric_frame = raw_frame.apply(pd.to_numeric, errors="coerce")
    return np.asarray(scaler.transform(imputer.transform(numeric_frame)), dtype=np.float32)


MODEL_SIZE_KB = os.path.getsize(MODEL_PATH) / 1024
PARAMETERS = 4385

with st.sidebar:
    st.markdown("## ⚡ Edge AI")
    st.markdown("### Select Detection Model")
    available_models = ["All Models Prediction", "Lightweight MLP (INT8 TFLite)"] + [
        name for name, path in KERAS_MODEL_PATHS.items() if os.path.exists(path)
    ]
    selected_model = st.selectbox(
        "Model used for predictions",
        available_models,
        index=0,
        key="active_model"
    )
    st.markdown("### System Status")
    st.success("🟢 Model loaded")
    st.metric("Model", "INT8 TFLite")
    st.metric("Model size", f"{MODEL_SIZE_KB:.2f} KB")
    st.metric("Parameters", f"{PARAMETERS:,}")
    st.metric("Input features", INPUT_FEATURES)
    st.divider()
    st.markdown("### Edge Resource Monitor")
    if PSUTIL_AVAILABLE:
        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory().percent
        st.progress(min(cpu / 100, 1.0), text=f"CPU usage: {cpu:.1f}%")
        st.progress(min(ram / 100, 1.0), text=f"RAM usage: {ram:.1f}%")
    else:
        st.info("Install psutil to display CPU/RAM usage.")
    st.divider()
    if st.button("🗑️ Clear Detection History", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_result = None
        st.rerun()

st.title("⚡ Edge AI Real-Time Anomaly Detector")
st.caption("Lightweight Neural Network for Real-Time Anomaly Detection on Resource-Constrained Edge Devices")
st.divider()

st.subheader("Edge AI Model Information")
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Model", "INT8 TFLite")
with c2: st.metric("Parameters", f"{PARAMETERS:,}")
with c3: st.metric("Model Size", f"{MODEL_SIZE_KB:.2f} KB")
with c4: st.metric("Input Features", INPUT_FEATURES)

st.subheader("🧠 Lightweight Neural Network Architecture")
st.code("""115 Input Features
        ↓
Dense Layer – 32 Neurons + ReLU
        ↓
Dense Layer – 16 Neurons + ReLU
        ↓
Dense Layer – 8 Neurons + ReLU
        ↓
Output Layer – 1 Neuron + Sigmoid
        ↓
NORMAL / ANOMALY
        ↓
INT8 TensorFlow Lite
        ↓
Edge Inference""", language="text")


st.divider()
st.header("🧑‍💻 Manual Network Traffic Analysis")
st.caption("Enter your own 115 raw feature values or upload a CSV. The saved training imputer and scaler are applied before prediction.")

with st.expander("Input instructions"):
    st.markdown(
        "- Enter raw numerical values, not values already scaled.\n"
        "- Keep the exact feature definitions, units and order used during training.\n"
        "- CSV must contain 115 feature columns. An optional label column can contain 0 (Normal) or 1 (Anomaly).\n"
        "- If data/processed/feature_names.json exists, CSV columns are matched by name. Otherwise, CSV column order is used.\n"
        "- Arbitrary values outside the training-data distribution may produce unreliable predictions."
    )

def display_user_prediction(processed_sample, model_choice):
    if model_choice == "All Models Prediction":
        df_result = predict_all_models(processed_sample, interpreter, input_details, output_details, scaler, imputer)
        st.dataframe(df_result, use_container_width=True, hide_index=True)
    else:
        pred, prob, elapsed = predict_selected_model(model_choice, processed_sample, interpreter, input_details, output_details, scaler, imputer)
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", label_text(pred))
        c2.metric("Anomaly Probability", f"{prob*100:.4f}%")
        c3.metric("Confidence", f"{(prob if pred == 1 else 1-prob)*100:.4f}%")
        st.caption(f"Inference time: {elapsed:.4f} ms")
        st.error("🔴 ANOMALY DETECTED") if pred == 1 else st.success("🟢 NORMAL")
    st.info("No actual label was supplied, so correctness cannot be evaluated.")

manual_tab, csv_tab = st.tabs(["Enter 115 features manually", "Upload CSV"])

with manual_tab:
    st.write("Enter all 115 raw numerical feature values in the same order used during training.")
    manual_model = st.selectbox("Choose model", ["All Models Prediction", "Lightweight MLP (INT8 TFLite)", "Lightweight MLP", "Baseline MLP", "KronNet-inspired"], key="manual_input_model")
    with st.form("manual_115_feature_form"):
        entered = []
        for start_idx in range(0, INPUT_FEATURES, 3):
            cols = st.columns(3)
            for offset, col in enumerate(cols):
                idx = start_idx + offset
                if idx < INPUT_FEATURES:
                    with col:
                        val = st.number_input(FEATURE_NAMES[idx], value=0.0, format="%.8f", key=f"manual_feature_{idx}")
                        entered.append((idx, val))
        submitted = st.form_submit_button("🔍 Predict from my 115 inputs", type="primary", use_container_width=True)
    if submitted:
        try:
            values = [v for _, v in sorted(entered, key=lambda pair: pair[0])]
            raw_df = pd.DataFrame([values], columns=FEATURE_NAMES)
            processed = preprocess_raw_features(raw_df, scaler, imputer)
            display_user_prediction(processed[0], manual_model)
        except Exception as exc:
            st.error(f"Manual prediction failed: {exc}")

with csv_tab:
    st.write("Upload a CSV in the same layout as your example: one row per observation and 115 feature columns.")
    st.caption("An optional leading index column (for example, Unnamed: 0) is ignored.")

    csv_model = st.selectbox(
        "Choose model",
        ["All Models Prediction", "Lightweight MLP (INT8 TFLite)", "Lightweight MLP", "Baseline MLP", "KronNet-inspired"],
        key="csv_input_model"
    )

    # Download a template with the exact names learned by the fitted scaler.
    template_df = pd.DataFrame([{name: 0.0 for name in FEATURE_NAMES}])
    st.download_button(
        "⬇️ Download 115-feature CSV template",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="edge_ai_115_features_template.csv",
        mime="text/csv"
    )

    uploaded_csv = st.file_uploader("Upload feature CSV", type=["csv"], key="manual_feature_csv")
    if uploaded_csv is not None:
        try:
            uploaded_frame = pd.read_csv(uploaded_csv)

            # Remove common exported row-index columns, but do not drop real features.
            index_like = [
                c for c in uploaded_frame.columns
                if str(c).strip().lower().startswith("unnamed:")
                or str(c).strip().lower() in {"index", "row_id"}
            ]
            uploaded_frame = uploaded_frame.drop(columns=index_like)

            label_col = next(
                (c for c in ["label", "Label", "target", "Target", "class", "Class"]
                 if c in uploaded_frame.columns),
                None
            )
            feature_frame = uploaded_frame.drop(columns=[label_col]) if label_col else uploaded_frame.copy()

            if feature_frame.shape[1] != INPUT_FEATURES:
                st.error(
                    f"Expected {INPUT_FEATURES} feature columns after removing an optional index/label column; "
                    f"found {feature_frame.shape[1]}."
                )
                st.stop()

            # Accept either real feature names or the Feature_1 ... Feature_115 layout
            # shown in the user's uploaded CSV. The latter is mapped by column position.
            if all(name in feature_frame.columns for name in FEATURE_NAMES):
                feature_frame = feature_frame[FEATURE_NAMES]
            elif list(feature_frame.columns) == [f"Feature_{i}" for i in range(1, INPUT_FEATURES + 1)]:
                feature_frame.columns = FEATURE_NAMES
                st.info("Detected Feature_1 ... Feature_115. Columns are mapped by position to the exact training feature order.")
            else:
                st.error(
                    "CSV columns do not match the training feature names or the expected Feature_1 ... Feature_115 layout. "
                    "Download the template above or use the exact training feature order."
                )
                st.stop()

            # The attached example has values around -0.7 to 1.1, suggesting it may
            # already be scaled. Let the user explicitly choose to avoid scaling twice.
            values_look_scaled = bool(
                feature_frame.apply(pd.to_numeric, errors="coerce").notna().all().all()
            )
            already_scaled = st.checkbox(
                "My CSV values are already preprocessed/scaled (skip imputer and scaler)",
                value=False,
                help="Enable this only if the CSV values were exported from the processed X_test/X_train data after the same imputer and scaler."
            )

            st.write(f"Rows detected: **{len(feature_frame)}** | Feature columns: **{feature_frame.shape[1]}**")
            st.dataframe(feature_frame.head(5), use_container_width=True)

            if st.button("🔍 Predict uploaded CSV", type="primary", key="run_manual_csv"):
                numeric_frame = feature_frame.apply(pd.to_numeric, errors="coerce")
                if already_scaled:
                    if numeric_frame.isna().any().any():
                        st.error("Scaled input CSV contains blank or non-numeric values. Please correct them.")
                        st.stop()
                    processed_rows = numeric_frame.to_numpy(dtype=np.float32)
                else:
                    processed_rows = preprocess_raw_features(numeric_frame, scaler, imputer)

                output_rows = []
                progress = st.progress(0)
                for row_idx, sample in enumerate(processed_rows):
                    actual = None
                    if label_col:
                        try:
                            actual = int(uploaded_frame.iloc[row_idx][label_col])
                        except (TypeError, ValueError):
                            actual = None

                    if csv_model == "All Models Prediction":
                        model_results = predict_all_models(
                            sample, interpreter, input_details, output_details,
                            scaler, imputer, actual_label=actual
                        )
                        for _, result in model_results.iterrows():
                            item = result.to_dict()
                            item["CSV Row"] = row_idx + 1
                            output_rows.append(item)
                    else:
                        pred, prob, elapsed = predict_selected_model(
                            csv_model, sample, interpreter, input_details,
                            output_details, scaler, imputer
                        )
                        item = {
                            "CSV Row": row_idx + 1,
                            "Prediction": label_text(pred),
                            "Anomaly Probability (%)": round(prob * 100, 4),
                            "Confidence (%)": round((prob if pred == 1 else 1-prob) * 100, 4),
                            "Inference Time (ms)": round(elapsed, 4)
                        }
                        if actual in (0, 1):
                            item["Actual Label"] = label_text(actual)
                            item["Correct"] = "Yes" if pred == actual else "No"
                        output_rows.append(item)

                    progress.progress((row_idx + 1) / len(processed_rows))

                progress.empty()
                result_df = pd.DataFrame(output_rows)
                st.subheader("CSV prediction results")
                st.dataframe(result_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Download prediction results",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name="manual_csv_anomaly_predictions.csv",
                    mime="text/csv"
                )
        except Exception as exc:
            st.error(f"CSV processing failed: {exc}")

st.divider()
st.subheader("🔎 Real-Time Detection")
st.write("Generate observations from the held-out N-BaIoT test set and run the deployed INT8 Edge AI model.")
b1, b2 = st.columns(2)
with b1:
    if st.button("🎲 Generate Test Sample", use_container_width=True):
        index = np.random.randint(0, len(X_test))
        st.session_state.selected_sample = X_test[index]
        st.session_state.selected_actual = int(y_test[index])
        st.session_state.last_result = None
        st.session_state.all_model_results = None
with b2:
    if st.button("🧹 Clear Current Sample", use_container_width=True):
        st.session_state.selected_sample = None
        st.session_state.selected_actual = None
        st.session_state.last_result = None
        st.session_state.all_model_results = None

if st.session_state.selected_sample is not None:
    actual = st.session_state.selected_actual
    st.info(f"Selected test observation. Actual label: **{label_text(actual)}**")
    with st.expander(f"View {INPUT_FEATURES} Input Features"):
        feature_df = pd.DataFrame([st.session_state.selected_sample], columns=[f"Feature_{i+1}" for i in range(INPUT_FEATURES)])
        st.dataframe(feature_df, use_container_width=True)
    if st.button("🔍 RUN PREDICTION", type="primary", use_container_width=True):
        if selected_model == "All Models Prediction":
            st.session_state.all_model_results = predict_all_models(
                st.session_state.selected_sample, interpreter, input_details,
                output_details, scaler, imputer, actual_label=st.session_state.selected_actual
            )
            st.session_state.last_result = None
        else:
            prediction, probability, inference_ms = predict_selected_model(selected_model, st.session_state.selected_sample, interpreter, input_details, output_details, scaler, imputer)
            confidence = probability if prediction == 1 else 1 - probability
            correct = prediction == actual
            st.session_state.last_result = {"prediction": prediction, "probability": probability, "confidence": confidence, "inference_ms": inference_ms, "actual": actual, "correct": correct}
            st.session_state.all_model_results = None
            add_history(prediction, probability, inference_ms, actual)

if st.session_state.get("all_model_results") is not None:
    st.divider()
    st.subheader("🔬 Same Input — All Four Models")
    st.caption("Every model below received the exact same selected test observation.")
    st.dataframe(st.session_state.all_model_results, use_container_width=True, hide_index=True)

if st.session_state.last_result is not None:
    result = st.session_state.last_result
    st.divider()
    st.subheader("AI Prediction")
    if result["prediction"] == 1: st.error("🔴 ANOMALY DETECTED")
    else: st.success("🟢 NORMAL")
    m1, m2, m3 = st.columns(3)
    with m1: st.metric("Confidence", f"{result['confidence'] * 100:.2f}%")
    with m2: st.metric("Inference Time", f"{result['inference_ms']:.4f} ms")
    with m3: st.metric("Actual Label", label_text(result["actual"]))
    if result["correct"]: st.success("✅ Correct prediction")
    else: st.warning("⚠️ Incorrect prediction")

st.divider()
st.subheader("📡 Live Edge Monitoring")
st.write("Run a real-time demonstration using randomly selected observations from the N-BaIoT test set.")
monitor_count = st.slider("Samples per monitoring run", 10, 100, 30, 10)
if st.button("▶ Start Live Monitoring", type="secondary", use_container_width=True):
    progress = st.progress(0, text="Starting Edge AI monitoring...")
    status_box = st.empty()
    correct_count = 0
    anomaly_count = 0
    total_time = 0.0
    for i in range(monitor_count):
        index = np.random.randint(0, len(X_test))
        sample = X_test[index]
        actual = int(y_test[index])
        prediction, probability, inference_ms = predict_selected_model(selected_model, sample, interpreter, input_details, output_details, scaler, imputer)
        confidence = probability if prediction == 1 else 1 - probability
        correct_count += int(prediction == actual)
        anomaly_count += int(prediction == 1)
        total_time += inference_ms
        add_history(prediction, probability, inference_ms, actual)
        if prediction == 1:
            status_box.error(f"⚠️ ANOMALY DETECTED | Confidence: {confidence * 100:.2f}% | Inference: {inference_ms:.4f} ms")
        else:
            status_box.success(f"🟢 NORMAL | Confidence: {confidence * 100:.2f}% | Inference: {inference_ms:.4f} ms")
        progress.progress((i + 1) / monitor_count, text=f"Processing sample {i + 1}/{monitor_count}")
        time.sleep(0.03)
    stream_accuracy = correct_count / monitor_count * 100
    avg_inference = total_time / monitor_count
    progress.empty()
    st.success(f"Monitoring run complete: {correct_count}/{monitor_count} correct ({stream_accuracy:.2f}%).")
    a, b, c, d = st.columns(4)
    with a: st.metric("Samples", monitor_count)
    with b: st.metric("Correct", correct_count)
    with c: st.metric("Anomalies", anomaly_count)
    with d: st.metric("Avg Inference", f"{avg_inference:.4f} ms")

if st.session_state.history:
    st.divider()
    st.subheader("📜 Detection History")
    history_df = pd.DataFrame(st.session_state.history)
    display_df = history_df.copy()
    display_df["Confidence"] = (display_df["Confidence"] * 100).map(lambda x: f"{x:.2f}%")
    display_df["Inference (ms)"] = display_df["Inference (ms)"].map(lambda x: f"{x:.4f}")
    st.dataframe(display_df.iloc[::-1], use_container_width=True, height=300)
    st.download_button("⬇️ Download Detection History", data=history_df.to_csv(index=False).encode("utf-8"), file_name="edge_ai_detection_history.csv", mime="text/csv")

if len(st.session_state.history) >= 2:
    st.divider()
    st.subheader("📈 Detection Probability History")
    chart_df = pd.DataFrame(st.session_state.history)
    chart_df["Sample"] = range(1, len(chart_df) + 1)
    chart_df["Anomaly Probability"] = chart_df["Confidence"].where(chart_df["Prediction"] == "ANOMALY", 1 - chart_df["Confidence"])
    st.line_chart(chart_df[["Sample", "Anomaly Probability"]].set_index("Sample"))

st.divider()
st.subheader("⚡ Edge AI Performance")
p1, p2, p3 = st.columns(3)
with p1: st.metric("INT8 Model Size", "8.69 KB")
with p2: st.metric("Parameter Reduction", "82.95%")
with p3: st.metric("INT8 Test Accuracy", "99.99%")

comparison_path = os.path.join(BASE_DIR, "results", "model_comparison.csv")
if os.path.exists(comparison_path):
    comparison = pd.read_csv(comparison_path)
    st.subheader("📋 Experiment-based Model Comparison")
    st.caption("Values are loaded from results/model_comparison.csv. Re-run src/compare_models.py to refresh.")
    st.dataframe(comparison, use_container_width=True, hide_index=True)
else:
    st.info("Run `python src\\compare_models.py` to generate the comparison table.")

st.divider()
st.header("📊 Model Performance & Research Results")
st.write("These figures are generated from the actual experiments performed during this project.")

# Dynamic comparison charts: include all four models from the experiment CSV.
comparison_path = os.path.join(BASE_DIR, "results", "model_comparison.csv")
if os.path.exists(comparison_path):
    comparison = pd.read_csv(comparison_path)

    # Normalize possible model-name variations for display
    if "Model" in comparison.columns:
        comparison["Model"] = comparison["Model"].replace({
            "Lightweight INT8": "Lightweight MLP (INT8 TFLite)",
            "Lightweight MLP INT8": "Lightweight MLP (INT8 TFLite)",
            "KronNet": "KronNet-inspired",
        })

    st.subheader("1. Model Parameter Comparison")
    parameter_data = pd.DataFrame({
        "Model": ["Baseline MLP", "Lightweight MLP", "Lightweight MLP (INT8 TFLite)", "KronNet-inspired"],
        "Parameters": [25729, 4385, 4385, 9737],
    }).set_index("Model")
    st.bar_chart(parameter_data)
    st.caption("INT8 TFLite uses the same 4,385-parameter Lightweight MLP architecture; quantization reduces deployed model size.")

    metric_aliases = {
        "Accuracy": ["Accuracy", "accuracy"],
        "F1 Score": ["F1", "F1-score", "f1", "f1_score"],
        "Precision": ["Precision", "precision"],
        "Recall": ["Recall", "recall"],
    }

    for chart_title, aliases in metric_aliases.items():
        col = next((name for name in aliases if name in comparison.columns), None)
        if col:
            st.subheader(f"Model {chart_title} Comparison")
            chart_df = comparison[["Model", col]].copy().set_index("Model")
            chart_df[col] = pd.to_numeric(chart_df[col], errors="coerce")
            st.bar_chart(chart_df)

    size_col = next((name for name in ["Model Size (KB)", "Keras Size (KB)", "Size (KB)", "keras_size_kb"] if name in comparison.columns), None)
    if size_col:
        st.subheader("Model File Size Comparison")
        size_df = comparison[["Model", size_col]].copy().set_index("Model")
        size_df[size_col] = pd.to_numeric(size_df[size_col], errors="coerce")
        st.bar_chart(size_df)

# Keep the original training history charts, which are specific to the trained lightweight model.
figure_sections = [
    ("Training and Validation Accuracy", "training_validation_accuracy.png", "Training history for the Lightweight MLP."),
    ("Training and Validation Loss", "training_validation_loss.png", "Training history for the Lightweight MLP."),
]
for title, filename, caption in figure_sections:
    path = figure_path(filename)
    if path:
        st.subheader(title)
        st.image(path, use_container_width=True)
        st.caption(caption)

confusion_path = os.path.join(BASE_DIR, "results", "lightweight_confusion_matrix.png")
if os.path.exists(confusion_path):
    st.subheader("7. Lightweight Model Confusion Matrix")
    st.image(confusion_path, use_container_width=True)
    st.caption("Confusion matrix from the lightweight model evaluation.")

st.divider()
st.subheader("🔬 Complete Project Pipeline")
st.code("""N-BaIoT Dataset
        ↓
Data Cleaning
        ↓
115 Input Features
        ↓
Imputation + Scaling
        ↓
Lightweight MLP
        ↓
4,385 Parameters
        ↓
TensorFlow Lite Conversion
        ↓
INT8 Quantization
        ↓
8.69 KB Model
        ↓
Real-Time Edge Inference
        ↓
NORMAL / ANOMALY
        ↓
Dashboard Monitoring""", language="text")

st.divider()
st.caption("Real-Time Anomaly Detection on Resource-Constrained Edge Devices | N-BaIoT | Lightweight MLP | TensorFlow Lite INT8")