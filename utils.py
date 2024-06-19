import streamlit as st 
import stanza
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Tokenize pipeline for biomedical text
tok = stanza.Pipeline('en', package='genia', processors='tokenize', verbose=False)

#function for segmenting text
def text_segment(text):
  #tokenize
  tokdoc = (tok(text))
  #store sentences
  sentences = []
  # Extract sentences from the processed document
  for sentence in tokdoc.sentences:
      sentences.append(sentence.text)
  # return the list of sentences
  return sentences

@st.cache_resource
def load_model():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model 

def similiarity_score(title, sentences, model):
    scores = []
    embedding_1= model.encode(title, convert_to_tensor=True)
    for i in range(len(sentences)):
        embedding_2 = model.encode(sentences[i], convert_to_tensor=True)
        s = util.pytorch_cos_sim(embedding_1, embedding_2),sentences[i]
        scores.append(float(s[0][0]))
    return scores


# Function to display the research paper
def display_research_paper(paper):
    paper = paper[0]

    st.markdown("#### {}".format(paper.get('title', 'No Title')))

    st.markdown("### Stanza Verb")
    st.write(paper.get('stanza_verb', 'No Stanza Verb'))

    st.markdown("### Genia Verb")
    st.write(paper.get('genia_verb', 'No Genia Verb'))

    st.markdown("### Abstract")
    st.write(paper.get('abstractText', 'No Abstract'))

    st.markdown("### Introduction")
    st.write(paper.get('INTRO', 'No Introduction'))

    st.markdown("### Methods")
    st.write(paper.get('METHODS', 'No Methods'))

    st.markdown("### Results")
    st.write(paper.get('RESULTS', 'No Results'))

    st.markdown("### Discussion")
    st.write(paper.get('DISCUSS', 'No Discussion'))

    st.markdown("### Figure Captions")
    st.write(paper.get('FIG_CAPTIONS', 'No Figure Captions'))

    st.markdown("### Table Captions")
    st.write(paper.get('TABLE_CAPTIONS', 'No Table Captions'))

    st.markdown("---")  # Separator for multiple papers

    st.markdown("### Journal")
    st.write(paper.get('journal', 'No Journal'))

    st.markdown("### Authors")
    authors = paper.get('authors', [])
    st.write(authors)

    st.markdown("### Year")
    st.write(paper.get('year', 'No Year'))

    st.markdown("### DOI")
    st.write(paper.get('doi', 'No DOI'))

    st.markdown("### PMID")
    st.write(paper.get('pmid', 'No PMID'))

    st.markdown("### Mesh Terms")
    st.write(paper.get('mesh', 'No Mesh Terms'))

    st.markdown("### Supplementary Mesh Terms")
    st.write(paper.get('supplMesh', 'No Supplementary Mesh Terms'))

    st.markdown("### Chemicals")
    st.write(paper.get('chemicals', 'No Chemicals'))

 
# Function to display the research paper
def predict_similarity(paper, model):
    paper = paper[0]

    title = paper['title'] 
    st.markdown('### Title:')
    st.markdown("#### {}".format(title))

    st.markdown('### Abstract:')
    text = text_segment(paper['abstractText'])
    scores  = similiarity_score(title,  text, model)
    df = {'Sentence': text,'Score': scores}
    df = pd.DataFrame(df)
    df = df.style.background_gradient(subset=['Score'], cmap='Greys')
    st.dataframe(df, use_container_width=True)

    st.markdown('### Introduction:')
    text = text_segment(paper['INTRO'])
    scores  = similiarity_score(title,  text, model)
    df = {'Sentence': text,'Score': scores}
    df = pd.DataFrame(df)
    df = df.style.background_gradient(subset=['Score'], cmap='Greys')
    st.dataframe(df, use_container_width=True)

    st.markdown('### Methods:')
    text = text_segment(paper['METHODS'])
    scores  = similiarity_score(title,  text, model)
    df = {'Sentence': text,'Score': scores}
    df = pd.DataFrame(df)
    df = df.style.background_gradient(subset=['Score'], cmap='Greys')
    st.dataframe(df, use_container_width=True)

    st.markdown('### Results:')
    text = text_segment(paper['RESULTS'])
    scores  = similiarity_score(title,  text, model)
    df = {'Sentence': text,'Score': scores}
    df = pd.DataFrame(df)
    df = df.style.background_gradient(subset=['Score'], cmap='Greys')
    st.dataframe(df, use_container_width=True)

    st.markdown('### Discussion:')
    text = text_segment(paper['DISCUSS'])
    scores  = similiarity_score(title,  text, model)
    df = {'Sentence': text,'Score': scores}
    df = pd.DataFrame(df)
    df = df.style.background_gradient(subset=['Score'], cmap='Greys')
    st.dataframe(df, use_container_width=True)




def labelling_similarity(paper):
    paper = paper[0]

    title = paper['title'] 
    st.markdown('### Title:')
    st.markdown("#### {}".format(title))

    st.markdown('### Sentences:')
    sentences = text_segment(paper['abstractText'])
    df = {'section':['ABSTRACT']*len(sentences), 'Sentence': sentences, 'User Input': [None] * len(sentences)}
    data1 = pd.DataFrame(df)
    sentences = text_segment(paper['INTRO'])
    df = {'section':['INTRO']*len(sentences), 'Sentence': sentences, 'User Input': [None] * len(sentences)}
    data2 = pd.DataFrame(df)
    sentences = text_segment(paper['METHODS'])
    df = {'section':['METHODS']*len(sentences), 'Sentence': sentences, 'User Input': [None] * len(sentences)}
    data3 = pd.DataFrame(df)
    sentences = text_segment(paper['RESULTS'])
    df = {'section':['RESULTS']*len(sentences), 'Sentence': sentences, 'User Input': [None] * len(sentences)}
    data4 = pd.DataFrame(df)
    sentences = text_segment(paper['DISCUSS'])
    df = {'section':['DISCUSS']*len(sentences), 'Sentence': sentences, 'User Input': [None] * len(sentences)}
    data5 = pd.DataFrame(df)

    data =  pd.concat([data1, data2, data3, data4, data5], ignore_index=True)

    return data

# Function to download dataframe as CSV
def download_dataframe_to_csv(df, filename):
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name=filename,
        mime='text/csv',
    )
