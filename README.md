```markdown
# Titanic Dataset Chatbot

## Overview
This project is a chatbot application that analyzes the Titanic dataset, providing both text responses and visual insights based on user queries. Users can ask questions in plain English (e.g., "What percentage of 1st class passengers survived?" or "What was the average ticket fare?") and receive dynamic answers, including statistics and visualizations.

## Features
- **Natural Language Queries:** Ask questions about Titanic passengers in plain English.
- **Dynamic Statistics:** Calculate mean, count, median, and percentages (with optional filters) on the dataset.
- **Data Visualizations:** Generate histograms and other graphs for data insights.
- **User-Friendly Interface:** Streamlit-based frontend for a clean, chat-like experience.

## Tech Stack
- **Backend:** FastAPI, Pandas, Matplotlib, Seaborn
- **Agent Framework:** LangChain (using Google Gemini via `ChatGoogleGenerativeAI`)
- **Frontend:** Streamlit
- **Language & Environment:** Python 3.x (virtual environment recommended)

## Project Structure
```
titanic_dataset_chat_agent/
├── backend.py          # FastAPI backend code with LangChain agent
├── app.py              # Streamlit frontend code
├── titanic_dataset.csv # Titanic dataset CSV file
├── static/             # Directory for generated images (e.g., histograms)
└── README.md           # This file
```

## Setup & Installation

1. **Clone the Repository:**
   ```bash
   git clone <repository_url>
   cd titanic_dataset_chat_agent
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   If you have a `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```
   Otherwise, install the following packages:
   ```bash
   pip install fastapi uvicorn pandas matplotlib seaborn streamlit langchain langchain_google_genai pydantic
   ```

4. **Configure the API Key:**
   In `backend.py`, update the Google Gemini API key:
   ```python
   GOOGLE_API_KEY = "YOUR_GOOGLE_GEMINI_API_KEY"
   ```
   Replace the placeholder with your actual API key.

5. **Dataset:**
   Ensure the file `titanic_dataset.csv` is placed in the project root.

## Running the Application

### Start the FastAPI Backend
1. In your project directory, run:
   ```bash
   uvicorn backend:app --reload
   ```
2. The backend will run at [http://127.0.0.1:8000](http://127.0.0.1:8000). You can view the API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Start the Streamlit Frontend
1. Open a new terminal (with your virtual environment activated) and run:
   ```bash
   streamlit run app.py
   ```
2. The Streamlit interface will open in your default browser. Use the chat input to ask questions about the Titanic dataset.

## Example Queries
- **Text Queries:**
  - "What was the average ticket fare?"
  - "What percentage of 1st class passengers survived?"
- **Visualization Queries:**
  - "Show a histogram of Age."
  - "Create a histogram of Fare."

## Notes
- The backend uses LangChain with Google Gemini. You might see deprecation warnings regarding LangChain agents; these do not affect functionality.
- When filtering (e.g., "mean,Survived,Pclass==1"), ensure the input format is exactly as specified:  
  **`stat_type,column_name,filter_expression`**
- For binary columns like "Survived", the mean is multiplied by 100 to present a percentage.

## License
This project is licensed under the MIT License. 

## Acknowledgements
- [LangChain](https://python.langchain.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)
- [Google Gemini](https://cloud.google.com/vertex-ai)
```