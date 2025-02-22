import streamlit as st
import requests

# Define the FastAPI backend endpoint URL
FASTAPI_URL = "http://localhost:8000/query/"

# Set up the Streamlit page configuration
st.set_page_config(page_title="Titanic Analyzer", layout="wide")
st.title("🚢 Titanic Dataset Chatbot")
st.markdown("Ask questions about Titanic passengers! (e.g., 'Show age distribution', 'What percentage survived?')")

# Provide some example questions in an expandable section
with st.expander("Example Questions"):
    st.markdown("""
    - What's the average age of survivors?
    - Show gender distribution
    - Create histogram of ticket prices
    - What percentage of 1st class passengers survived?
    """)

# Get user input via chat interface
user_query = st.chat_input("Ask your question...")

if user_query:
    with st.spinner("Analyzing data..."):
        try:
            # Send the user's query to the FastAPI backend as a GET request
            params = {"query": user_query}
            response = requests.get(FASTAPI_URL, params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors
            result = response.json()     # Expecting a JSON response like {"response": ...}
            
            # Retrieve the content from the response
            content = result.get("response", "")
            
            # Check if the content appears to be an image (e.g., ends with ".png")
            if isinstance(content, str) and content.endswith(".png"):
                st.image(content, use_column_width=True)
            else:
                st.write(content)
                
        except KeyError:
            st.error("Invalid response format from server")
        except Exception as e:
            st.error(f"Connection error: {str(e)}")
