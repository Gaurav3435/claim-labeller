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

@st.cache_resource
def load_model2_cached():
    return load_model2()

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
    #load model and tokenizers
    model = load_model_cached() 
    tokenizer, model2  = load_model2_cached()  
    
    #check if the paper is uploaded
    if st.session_state.form_submitted:

        #create list of required sections from the paper
        sections = ['abstractText','INTRO','METHODS','RESULTS','DISCUSS']

        st.markdown('Approach 1: Confidence score between 0 to 100')
        st.markdown('Approach 2: Claim: 1 and Non Claim 0')

        #show title
        title = st.session_state.data[0]['title'] 
        st.markdown('### Title:')
        st.markdown("#### {}".format(title))

        # Create an empty DataFrame with the specified columns
        df = pd.DataFrame({'Section':[],'Sentence': [],'Approach 1': [],'Approach 2':[]})

        #iterate through abstract section in the paper
        st.markdown('### {}:'.format(sections[0]))
        df_sentences1 = predict_similarity(st.session_state.data, model, model2, tokenizer, sections[0])
        df_sentences =  df_sentences1.style.background_gradient(subset=['Approach 1', 'Approach 2'], cmap='Greys')
        st.dataframe(df_sentences, use_container_width=True)
        df =  pd.concat([df, df_sentences1], ignore_index=True)
        #iterate through INTRO section in the paper
        st.markdown('### {}:'.format(sections[1]))
        df_sentences1 = predict_similarity(st.session_state.data, model, model2, tokenizer, sections[1])
        df_sentences =  df_sentences1.style.background_gradient(subset=['Approach 1', 'Approach 2'], cmap='Greys')
        st.dataframe(df_sentences, use_container_width=True)
        df =  pd.concat([df, df_sentences1], ignore_index=True)
        #iterate through METHODS section in the paper
        st.markdown('### {}:'.format(sections[2]))
        df_sentences1 = predict_similarity(st.session_state.data, model, model2, tokenizer, sections[2])
        df_sentences =  df_sentences1.style.background_gradient(subset=['Approach 1', 'Approach 2'], cmap='Greys')
        st.dataframe(df_sentences, use_container_width=True)
        df =  pd.concat([df, df_sentences1], ignore_index=True)
        #iterate through RESULTS section in the paper
        st.markdown('### {}:'.format(sections[3]))
        df_sentences1 = predict_similarity(st.session_state.data, model, model2, tokenizer, sections[3])
        df_sentences =  df_sentences1.style.background_gradient(subset=['Approach 1', 'Approach 2'], cmap='Greys')
        st.dataframe(df_sentences, use_container_width=True)
        df =  pd.concat([df, df_sentences1], ignore_index=True)
        #iterate through DISCUSS section in the paper
        st.markdown('### {}:'.format(sections[4]))
        df_sentences1 = predict_similarity(st.session_state.data, model, model2, tokenizer, sections[4])
        df_sentences =  df_sentences1.style.background_gradient(subset=['Approach 1', 'Approach 2'], cmap='Greys')
        st.dataframe(df_sentences, use_container_width=True)
        df =  pd.concat([df, df_sentences1], ignore_index=True)
       
        # Convert DataFrame to CSV
        csv = df.to_csv(index=False)
        # Create a download button
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='{}.csv'.format(str(uploaded_file.name).split('.')[0]),
            mime='text/csv'
        )

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
