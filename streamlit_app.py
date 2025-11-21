import streamlit as st
import requests
import base64

# ========================================
# CONFIGURATION
# ========================================

# Change this to your FastAPI URL
# For local testing:
BACKEND_URL = "http://127.0.0.1:8000/predict"

# For render deployment later (example):
# BACKEND_URL = "https://your-backend.onrender.com/predict"


# ========================================
# STREAMLIT UI
# ========================================

st.set_page_config(page_title="MRI Dementia Classifier", layout="wide")

st.title("🧠 MRI Dementia Classification")
st.write("Upload a ZIP containing DICOM MRI slices of a single patient.")

uploaded_file = st.file_uploader("Upload Patient ZIP File", type=["zip"])

if uploaded_file is not None:
    if st.button("Predict Diagnosis"):
        with st.spinner("Processing MRI… Please wait..."):

            # Send POST request to FastAPI
            files = {"file": (uploaded_file.name, uploaded_file, "application/zip")}

            try:
                response = requests.post(BACKEND_URL, files=files)

                if response.status_code != 200:
                    st.error(f"Backend Error: {response.text}")
                else:
                    result = response.json()

                    pred = result["prediction"]
                    conf = float(result["confidence"])
                    metadata = result["patient_metadata"]
                    middle_slice_b64 = result["middle_slice_base64"]

                    # ================================
                    # DISPLAY RESULTS
                    # ================================
                    st.success(f"### 🩺 Diagnosis: **{pred}**")
                    st.info(f"Model Confidence: **{conf:.2f}**")

                    # Metadata table
                    st.subheader("📋 Patient Metadata")
                    st.json(metadata)

                    # Middle slice preview
                    st.subheader("🖼️ Middle MRI Slice")

                    st.image(middle_slice_b64, caption="Middle Slice", use_column_width=True)

            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")
