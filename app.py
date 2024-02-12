import streamlit as st
from transformers import pipeline

# Initialize the Hugging Face sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    """Analyzes the sentiment of the provided text and returns the result."""
    result = sentiment_pipeline(text)[0]
    return result

def main():
    st.title("Sentiment Analysis with Streamlit")
    st.write("This app uses a pre-trained model to analyze sentiment of the text you enter.")

    # Text input
    user_input = st.text_area("Enter text for sentiment analysis", "I love coding with Python!")

    if st.button("Analyze"):
        # Perform sentiment analysis
        result = analyze_sentiment(user_input)
        st.write(f"Sentiment: {result['label']}")
        st.write(f"Confidence: {result['score']:.2f}")

if __name__ == "__main__":
    main()
