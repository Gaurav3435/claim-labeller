import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('saved_weights')
model = BertForSequenceClassification.from_pretrained('saved_weights', num_labels=2)  # Change num_labels as needed

# Function to predict the class of input text
def predict_class(text):
    # Tokenize input text
    inputs = tokenizer(text,return_tensors="pt", truncation=True, padding=True)
    print(inputs['input_ids'])
    print(inputs['attention_mask'])
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
    # Get predicted class
    print(outputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return predicted_class

# Example usage
text = "Hi, My name is Gaurav"
predicted_class = predict_class(text)
print(f"The predicted class is: {predicted_class}")
