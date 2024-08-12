import streamlit as st
import pandas as pd

# Sample DataFrame
data = {
    'Sentence': [
        'This is the first sentence.',
        'Here is the second sentence.',
        'And this is the third sentence.'
    ],
    'Prediction': ['Correct', 'Incorrect', 'Correct']
}

df = pd.DataFrame(data)

# Streamlit app
st.title('Sentence Validation')

# Create a form to handle validation inputs
with st.form(key='validation_form'):
    # Display DataFrame with checkboxes for validation
    validated_sentences = []
    validation_results = []

    for index, row in df.iterrows():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**Sentence {index + 1}:** {row['Sentence']}")
        with col2:
            is_correct = st.checkbox(f"Correct?", key=f"checkbox_{index}")
            validation_results.append(is_correct)

    # Add a submit button to process the form
    submit_button = st.form_submit_button(label='Submit')

# Update the DataFrame with user validation results if the form is submitted
if submit_button:
    df['Validated'] = validation_results

    # Button to download the validated DataFrame as CSV
    st.write("### DataFrame with Validation Results")
    st.dataframe(df)

    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name='validated_sentences.csv',
        mime='text/csv'
    )
