import streamlit as st
import json
from utils import *

# Initialize session state variables if they don't exist
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'pmid_id' not in st.session_state:
    st.session_state.pmid_id = None
if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False
if 'labels_submitted' not in st.session_state:
    st.session_state.labels_submitted = False

@st.cache_data
def load_json_file(uploaded_file):
    return json.load(uploaded_file)

@st.cache_resource
def load_model_cached():
    return load_model()

@st.cache_data
def labelling_similarity_cached(data):
    return labelling_similarity(data)

# Function to clear the caches
def clear_caches():
    load_json_file.clear()
    labelling_similarity_cached.clear()

# Side bar
with st.sidebar.form("my_form"):
    st.write("Add PMID Article:")
    uploaded_file = st.file_uploader("Upload your JSON file", help='Upload a pmid_id.json file')
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.session_state.form_submitted = True
        st.session_state.labels_submitted = False  # Reset label submission state
        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            st.session_state.pmid_id = None
            st.write("Uploaded file name: {}".format(uploaded_file.name))
            st.session_state.data = load_json_file(uploaded_file)


# Tabs
tab0, tab1, tab2, tab3 = st.tabs(["Instruction", "Research Article", "Prediction", "Labelling"])

# Tab 0
with tab0:
    st.markdown("#### These are the instructions to label the claims in the research articles:")




# Tab 1
with tab1:
    if st.session_state.form_submitted:
        display_research_paper(st.session_state.data)

# Tab 2
with tab2:
    model = load_model_cached()
    if st.session_state.form_submitted:
        st.title("Predicted claims in Research Article")
        run_prediction = st.checkbox("Run Prediction")
        if run_prediction:
            predict_similarity(st.session_state.data, model)
        else:
            st.write("Prediction is turned off. Toggle the checkbox to run predictions.")

# Tab 3
with tab3:
    st.title("Labelling claims in Research Article")
    if st.session_state.form_submitted:
        df_sentences = labelling_similarity_cached(st.session_state.data)
        user_inputs = []  # Collect user inputs here
        if not st.session_state.labels_submitted:
            with st.form(key='input_form'):
                for i, sentence in enumerate(df_sentences.Sentence.values):
                    user_input = st.number_input(f"{sentence}", key=f"input_{i}", step=1, min_value=1, max_value=10, value=1)
                    user_inputs.append(user_input)  # Collect the user input
                submit_labels = st.form_submit_button(label='Submit-Label')
            if submit_labels:
                st.session_state.user_inputs = user_inputs
                st.session_state.labels_submitted = True
                df_sentences['User Input'] = user_inputs
                st.dataframe(df_sentences)
                output_file_name = str(uploaded_file.name).split('.')[0]+'.csv'
                download_dataframe_to_csv(df_sentences, output_file_name)
                clear_caches()
        else:
            st.write("Labels submitted successfully!")
    else:
        st.write("Waiting to Upload the file and hit submit...")
