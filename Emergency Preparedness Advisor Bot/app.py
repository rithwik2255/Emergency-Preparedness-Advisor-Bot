import gradio as gr
from utils import load_pdfs, load_pdfs_to_csv, extract_sentences_from_text  # Ensure these functions exist in utils.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')



# Define the folder path for PDFs and the path for the output CSV file
folder_path = 'C:\\Users\\rithw\\OneDrive\\Desktop\\Emergency Preparedness Advisor Bot\\datasets'  # Replace with your folder path
csv_output_path = 'C:\\Users\\rithw\\OneDrive\\Desktop\\Emergency Preparedness Advisor Bot\\pdf_texts.csv'  # Replace with your output path

# Extract PDF content and save to CSV
load_pdfs_to_csv(folder_path, csv_output_path)

# Load your PDFs into a list
pdf_texts = load_pdfs(folder_path)

# Check if pdf_texts is loaded successfully
if not pdf_texts:
    raise ValueError("No PDF texts loaded. Please check the load_pdfs function.")

# Print the loaded PDF texts for verification
for i, text in enumerate(pdf_texts):
    print(f"PDF {i}: {text[:500]}...")  # Print first 500 characters of each PDF

# Prepare the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(pdf_texts)

def generate_response(query):
    """Generate a response based on the user's query."""
    query_vec = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    highest_index = np.argmax(cosine_similarities)

    if cosine_similarities[highest_index] > 0.1:  # Adjust threshold as needed
        # Extract relevant text from the most relevant PDF
        relevant_text = pdf_texts[highest_index]
        
        # Split the relevant text into sentences
        sentences = extract_sentences_from_text(relevant_text)
        
        # Find the most relevant sentence based on similarity to the query
        sentence_vector = vectorizer.transform(sentences)
        similarities = cosine_similarity(query_vec, sentence_vector).flatten()
        
        # Get indices of top N most similar sentences
        top_n_indices = similarities.argsort()[-5:][::-1]  # Get top 5 sentences
        
        # Combine the top sentences into a response
        response_sentences = [sentences[i] for i in top_n_indices if similarities[i] > 0.1]
        
        # Optional: Add some context or adjacent sentences to enrich the response
        context_sentences = [sentences[i - 1] for i in top_n_indices if i > 0]  # Get preceding sentences
        response_sentences.extend(context_sentences)
        
        return ' '.join(response_sentences) if response_sentences else "I'm sorry, I don't have information on that."
    
    else:
        return "I'm sorry, I don't have information on that."

# Create a Gradio interface
iface = gr.Interface(fn=generate_response, 
                     inputs="text", 
                     outputs="text", 
                     title="Emergency Preparedness Chatbot",
                     description="Ask me any questions about emergency preparedness.")

# Launch the Gradio interface
iface.launch(server_port=7860)










