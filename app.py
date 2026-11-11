import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# 2. BACKGROUND AND INTERFACE DESIGN
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        135deg,
        #fce4ec 0%,
        #e3f2fd 50%,
        #f8f9ff 100%
    );
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    color: #17365D !important;
}

.stNumberInput input {
    background-color: white !important;
    color: #222222 !important;
    border-radius: 8px;
}

.stButton > button {
    background-color: #1769aa;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.65rem 1.2rem;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover {
    background-color: #0d477a;
    color: white;
}

div[data-testid="stMetric"] {
    background-color: rgba(255, 255, 255, 0.92);
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #d5dce5;
}

div[data-testid="stExpander"] {
    background-color: rgba(255, 255, 255, 0.75);
    border-radius: 10px;
}

section[data-testid="stSidebar"] {
    background-color: #edf3fc;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 3. LOAD THE TRAINED MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/best_model.joblib")

try:
    bundle = load_model()

    model = bundle["model"]
    model_name = bundle["model_name"]
    threshold = float(bundle["threshold"])
    feature_names = list(bundle["feature_names"])

except Exception as e:
    st.error("Unable to load the trained model.")
    st.write(
        "Check that models/best_model.joblib exists "
        "inside your project folder."
    )
    st.code(str(e))
    st.stop()

# --------------------------------------------------
# 4. LOAD DEFAULT INPUT VALUES
# --------------------------------------------------
dataset = load_breast_cancer()

dataset_df = pd.DataFrame(
    dataset.data,
    columns=dataset.feature_names
)

default_values = dataset_df.median()

# --------------------------------------------------
# 5. APPLICATION HEADER
# --------------------------------------------------
st.title("🎗️ Breast Cancer Detection System")

st.markdown("""
### Machine Learning Based Prediction

Enter the numerical tumor measurements below. The trained
model will estimate whether the input is classified as
**malignant** or **benign**.
""")

st.warning(
    "Educational project only. This model is not clinically "
    "validated and must not be used for medical diagnosis "
    "or treatment decisions."
)

# --------------------------------------------------
# 6. SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.header("About This Project")

    st.write(
        "This application demonstrates machine learning "
        "using the Wisconsin Diagnostic Breast Cancer dataset."
    )

    st.divider()

    st.subheader("Model Information")
    st.write(f"**Selected model:** {model_name}")
    st.write(f"**Malignant threshold:** {threshold:.2f}")

    st.divider()

    st.subheader("How to Use")
    st.write("1. Enter the 30 feature values.")
    st.write("2. Click Predict Result.")
    st.write("3. Review the output and probabilities.")

    st.caption("MCA Final Semester ML Project")

# --------------------------------------------------
# 7. INPUT FORM
# --------------------------------------------------
st.subheader("🧪 Enter Tumor Measurements")

st.write(
    "The inputs start with the dataset's median feature values. "
    "These are demonstration defaults, not patient measurements."
)

with st.form("prediction_form"):

    input_values = {}

    col1, col2, col3 = st.columns(3)
    columns = [col1, col2, col3]

    for i, feature in enumerate(feature_names):
        with columns[i % 3]:
            input_values[feature] = st.number_input(
                label=feature.replace("_", " ").title(),
                min_value=0.0,
                value=float(default_values[feature]),
                step=0.1,
                format="%.4f",
                help=f"Numerical value for {feature}"
            )

    st.divider()

    submitted = st.form_submit_button(
        "🔍 Predict Result",
        use_container_width=True
    )

# --------------------------------------------------
# 8. PREDICTION
# --------------------------------------------------
if submitted:
    try:
        input_df = pd.DataFrame(
            [[input_values[f] for f in feature_names]],
            columns=feature_names
        )

        # Get model probabilities
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)

        # Dataset labels: 0 = malignant, 1 = benign
        malignant_index = classes.index(0)
        benign_index = classes.index(1)

        malignant_probability = float(
            probabilities[malignant_index]
        )

        benign_probability = float(
            probabilities[benign_index]
        )

        # Apply the threshold selected during validation
        if malignant_probability >= threshold:
            prediction = "Malignant"
        else:
            prediction = "Benign"

        st.divider()
        st.subheader("📊 Prediction Results")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "Malignant Probability",
                f"{malignant_probability * 100:.2f}%"
            )

        with result_col2:
            st.metric(
                "Benign Probability",
                f"{benign_probability * 100:.2f}%"
            )

        # Predicted class
        st.subheader("Model Output")

        if prediction == "Malignant":
            st.error("Model prediction: Malignant class")
        else:
            st.success("Model prediction: Benign class")

        st.caption(
            f"Decision threshold: {threshold:.2f}. "
            "This is a model prediction, not a diagnosis."
        )

        # --------------------------------------------------
        # 9. PROBABILITY COMPARISON WITHOUT A CHART
        # --------------------------------------------------
        st.divider()
        st.subheader("Probability Comparison")

        st.write("**Malignant probability**")
        st.progress(malignant_probability)
        st.write(f"{malignant_probability * 100:.2f}%")

        st.write("**Benign probability**")
        st.progress(benign_probability)
        st.write(f"{benign_probability * 100:.2f}%")

        # --------------------------------------------------
        # 10. VIEW INPUT VALUES
        # --------------------------------------------------
        with st.expander("View Entered Measurements"):
            st.dataframe(
                input_df.T.rename(
                    columns={0: "Input Value"}
                ),
                use_container_width=True
            )

    except Exception as e:
        st.error("Prediction failed.")
        st.write("Error details:")
        st.code(str(e))

# --------------------------------------------------
# 11. FOOTER
# --------------------------------------------------
st.divider()

st.caption(
    "Breast Cancer Detection | Machine Learning Project | "
    "For educational and research purposes only."
)