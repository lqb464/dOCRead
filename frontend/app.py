import streamlit as st
import requests
import os

st.set_page_config(page_title="DOCRead", layout="wide")
API_BASE_URL = os.environ.get("INTERNAL_API_URL", "http://localhost:8000/api")

st.title("DOCRead")
st.write("Streamlit interface for DOCRead")

uploaded_file = st.file_uploader("Upload a document")
if uploaded_file is not None:
    if st.button("Process Document"):
        st.info("Document processing logic will connect to FastAPI...")
