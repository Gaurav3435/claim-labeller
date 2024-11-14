# Argumentative Claim Detection in Biomedical Research Articles

## Description

This project is a Streamlit application designed to identify argumentative claims in biomedical research articles. The approach combines two techniques for claim detection:

1. **Title-Claim Similarity**: This method is based on the idea that the title of a paper often contains the most important claim or summary of the research. By measuring the **comparative similarity** between the title and claim sentences within the paper, we can identify potential claims. This approach assumes that the title and claims should be contextually aligned.

2. **Fine-Tuning Pre-Trained Language Models**: This method fine-tunes a pre-trained language model (e.g., BERT) for claim detection. We use a dataset of English literature sentences labeled as "claim" or "non-claim" to fine-tune the model. Our goal is to investigate the accuracy of this claim detection method when applied to biomedical research.

Additionally, the platform provides a tab for **annotators** to manually label sentences as claims or non-claims. These annotations are used to verify the accuracy of the automated claim detection methods. The Streamlit web application serves as a demonstration for these functionalities.

## Features

- **Claim Detection**: Automatically detects claims in biomedical research articles using two approaches.
- **Manual Annotation**: Annotators can label each sentence to verify and improve the claim detection accuracy.
- **Real-time Results**: Watch the prediction results in real-time via the Streamlit application.

## Installation Instructions

To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Gaurav3435/claim-labeller.git
   cd claim-labeller
   ```

2. **Install the requirements**:
   It is recommended to set up a virtual environment for this project. You can install the necessary dependencies using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**:
   After installing the dependencies, you can start the Streamlit app by running:
   ```bash
   streamlit run app.py
   ```
   This will open the application in your default browser.

## Usage

1. **Upload the Sample Data**: 
   - The application expects a **JSON file** containing the details of the research articles. The JSON file includes the section of the article (e.g., title, abstract).
   - Upload this file using the provided upload button in the app interface.

2. **View the Predictions**: 
   - Once the data is uploaded, the model will process the content and display predictions for each sentence, identifying whether it is a claim or not.
   - You can compare the results from both the title-claim similarity approach and the fine-tuned language model (2nd approach is currently using pre trained bert model not the actual fine tuned weights for prediction, which can lead to inaccurate results).

3. **Annotator's Tab**:
   - Annotators can manually label each sentence as a "claim" or "non-claim" using the dedicated interface.
   - These annotations will help improve the detection accuracy by providing additional ground truth data.

## Contributing

We welcome contributions to improve the project. We welcome new approaches for detecting claims in biomedical literature. If you'd like to contribute, please follow these guidelines:

1. Fork the repository and create a new branch for your changes.
2. Make your changes and ensure that the code is properly tested.
3. Submit a pull request with a clear description of your changes and the rationale behind them.

Please ensure that your contributions adhere to the existing coding standards and conventions used in the project.

## License

## Contact

If you have any questions or suggestions, feel free to reach out to us via [gpatil@uwo.ca].

---
