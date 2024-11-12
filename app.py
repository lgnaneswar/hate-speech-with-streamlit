# Import necessary modules
import streamlit as st
import pyttsx3
import threading
import tweepy
from main import clean, clf, cv, extract_text

# Twitter API Bearer Token
bearer_token = 'AAAAAAAAAAAAAAAAAAAAABGuwwEAAAAAvAn3ByVMr4UIYH9PGBW0OFWFbQk%3DGtO5yRzIjZq8LfYDjggLGbfYCVZbejffleOfii8NDFcKqaWJZD'
client = tweepy.Client(bearer_token=bearer_token)

# Set Streamlit page config
st.set_page_config(page_title="Hate Speech Detection", page_icon="🤫", layout="wide")

# CSS for a visually pleasing, dynamic multi-colored background and styling
st.markdown("""
    <style>
        /* Multi-colored animated gradient background */
        body {
            background: linear-gradient(45deg, #ff7e5f, #feb47b, #6a82fb, #fc5c7d);
            background-size: 300% 300%;
            animation: gradientBG 10s ease infinite;
            color: #ffffff;
            font-family: Arial, sans-serif;
        }

        /* Gradient background animation */
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Title styling */
        .header-title {
            text-align: center;
            font-size: 3em;
            color: #ffffff;
            margin-bottom: 0.5em;
            text-shadow: 2px 2px #333333;
        }

        /* Sidebar styling */
        .css-1lcbmhc {
            background-color: rgba(50, 50, 50, 0.8);
            border-radius: 10px;
            padding: 1em;
        }

        /* Button styling */
        .stButton button {
            background-color: #ff7e5f;
            color: white;
            padding: 0.5em 1.5em;
            border-radius: 5px;
            font-size: 1em;
            font-weight: bold;
            border: none;
            transition: 0.3s;
        }

        /* Button hover effect */
        .stButton button:hover {
            background-color: #feb47b;
            color: #ffffff;
        }
        
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.markdown("<div class='header-title'>🤫 Hate Speech Detection App</div>", unsafe_allow_html=True)
st.write("### A tool to detect hate speech, offensive language, or neutral content.")

# Sidebar with input options
st.sidebar.title("🛠️ Options")
input_choice = st.sidebar.radio("Select Input Type:", ["Text Input", "Twitter URL", "Upload Image"])
st.sidebar.write("Choose an input type, enter your text, or upload an image to analyze hate speech.")

# Function to speak text asynchronously
def speak_async(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Function to run speech asynchronously in a thread
def speak_async_thread(text):
    threading.Thread(target=speak_async, args=(text,)).start()

def find_verdict(user_input):
    cleaned_input = clean(user_input)
    input_data = cv.transform([cleaned_input]).toarray()
    verdict = clf.predict(input_data)[0]
    st.success(f"**Verdict**: {verdict}")
    speak_async_thread(f"The Verdict is {verdict}")

# For text input
if input_choice == 'Text Input':
    user_input = st.text_area("Enter your text:", help="Type text you want to analyze.")
    if st.button("Detect Hate Speech"):
        if user_input:
            find_verdict(user_input)
        else:
            st.warning("Please enter text before clicking the button.")

# For Twitter URL
elif input_choice == 'Twitter URL':
    twitter_url = st.text_input("Enter Twitter URL:", help="Paste the URL of the tweet here.")
    if st.button("Fetch Tweet and Detect"):
        if twitter_url:
            try:
                tweet_id = twitter_url.split("/")[-1]
                tweet = client.get_tweet(tweet_id, tweet_fields=["text"])
                tweet_content = tweet.data["text"]
                find_verdict(tweet_content)
            except tweepy.TweepyException as e:
                st.error(f"Error fetching tweet: {str(e)}")
        else:
            st.warning("Please enter URL before clicking the button.")

# For image input
elif input_choice == 'Upload Image':
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        extracted_text = extract_text(uploaded_file)
        st.write("**Extracted Text:**")
        st.write(extracted_text)
        find_verdict(extracted_text)
