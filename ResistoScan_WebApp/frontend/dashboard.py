import streamlit as st
import requests

st.set_page_config(layout="wide")
st.title("🧬 ResistoScan AI Dashboard")

# Upload file
uploaded_file = st.file_uploader("Upload ARG Dataset", type=["csv"])

if uploaded_file is not None:
    st.write("File uploaded successfully")

    if st.button("Analyze"):
        files = {"file": uploaded_file.getvalue()}

        response = requests.post(
            "http://localhost:5000/predict",
            files={"file": uploaded_file}
        )

        if response.status_code == 200:
            result = response.json()

            st.subheader("🔍 Results")
            st.write(f"Environment: **{result['environment']}**")
            st.write(f"ITI Score: **{result['iti_score']:.2f}**")

            # Risk level
            score = result['iti_score']

            if score > 25000:
                st.error("CRITICAL RISK 🔴")
            elif score > 20000:
                st.warning("VERY HIGH RISK 🟠")
            elif score > 15000:
                st.warning("HIGH RISK 🟡")
            else:
                st.success("MODERATE RISK 🟢")

        else:
            st.error("Backend error")
