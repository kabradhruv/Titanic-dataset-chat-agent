# ------------------------------
# Import necessary libraries
# ------------------------------
from fastapi import FastAPI
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from pydantic import BaseModel

# ------------------------------
# Set your Google Gemini API key
# ------------------------------
GOOGLE_API_KEY = "your_api"

# ------------------------------
# Initialize FastAPI application
# ------------------------------
app = FastAPI()

# ------------------------------
# Load the Titanic dataset
# ------------------------------
df = pd.read_csv("titanic_dataset.csv")

# ------------------------------
# Define functions for our tools
# ------------------------------
def calculate_statistic(input_str: str):
    """
    Calculates a statistic for a given column, with optional filtering.
    Expects input in one of the following formats:
      - "stat_type,column_name" 
      - "stat_type,column_name,filter_expression"
    For example:
      - "mean,Survived" or "mean,Survived,Pclass==1"
    
    For binary columns like Survived, "mean" will be returned as a percentage.
    """
    # Split the input string and clean each part (remove extra whitespace/quotes)
    parts = [p.strip().strip("'").strip('"') for p in input_str.split(",")]
    if len(parts) not in [2, 3]:
        return ("Error: Please provide input as 'stat_type,column_name' or "
                "'stat_type,column_name,filter_expression' (e.g., 'mean,Survived,Pclass==1').")
    
    stat_type = parts[0].lower()
    column = parts[1]
    
    # Check if the column exists; try capitalizing if needed
    if column not in df.columns:
        cap_column = column.capitalize()
        if cap_column in df.columns:
            column = cap_column
        else:
            return f"Column '{column}' not found in dataset."
    
    # Apply filter if provided
    if len(parts) == 3:
        filter_expr = parts[2]
        try:
            filtered_df = df.query(filter_expr)
        except Exception as e:
            return f"Error applying filter: {e}"
    else:
        filtered_df = df
    
    # Compute the requested statistic on the filtered dataframe
    if stat_type == "mean":
        try:
            val = filtered_df[column].mean()
            # If the column is binary (like Survived), return as percentage
            if column.lower() == "survived":
                return f"The percentage of survivors is {val * 100:.2f}%."
            else:
                return f"The average {column} is {val:.2f}."
        except Exception as e:
            return f"Error calculating mean: {e}"
    elif stat_type == "count":
        try:
            val = filtered_df[column].count()
            return f"The total number of records for {column} is {val}."
        except Exception as e:
            return f"Error calculating count: {e}"
    elif stat_type == "median":
        try:
            val = filtered_df[column].median()
            return f"The median {column} is {val:.2f}."
        except Exception as e:
            return f"Error calculating median: {e}"
    elif stat_type in ["proportion", "percentage"]:
        # For categorical columns, compute percentage breakdown
        if filtered_df[column].dtype == "object":
            proportions = filtered_df[column].value_counts(normalize=True) * 100
            return {str(category): f"{percentage:.2f}%" for category, percentage in proportions.items()}
        else:
            return f"Proportion can only be calculated for categorical columns, not numeric ones."
    else:
        return "Unsupported statistic type. Use mean, count, median, or percentage."

def generate_histogram(input_str: str):
    """
    Generates a histogram for a given column.
    Expects input as the column name (e.g., "Age").
    """
    column = input_str.strip().strip("'").strip('"')
    if column not in df.columns:
        return f"Column '{column}' not found in dataset."
    
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column].dropna(), bins=20, kde=True)
    plt.xlabel(column)
    plt.title(f"Distribution of {column}")
    
    # Save the plot image in the "static" folder
    image_path = "static/histogram.png"
    os.makedirs("static", exist_ok=True)
    plt.savefig(image_path)
    plt.close()
    
    return image_path  # Return the image path so the frontend can display it


class CalculateStatisticArgs(BaseModel):
    input_str: str  # Expects input like "mean,Survived" or "mean,Survived,Pclass==1"

class GenerateHistogramArgs(BaseModel):
    input_str: str  # Expects the column name (e.g., "Age")

# ------------------------------
# Define LangChain tools
# ------------------------------
tools = [
    Tool(
        name="Calculate Statistic",
        func=calculate_statistic,
        description=("Calculates a statistic for a given column. "
                     "Provide input as 'stat_type,column_name' or "
                     "'stat_type,column_name,filter_expression' (e.g., 'mean,Survived,Pclass==1')."),
        args_schema=CalculateStatisticArgs
    ),
    Tool(
        name="Generate Histogram",
        func=generate_histogram,
        description="Generates a histogram for a given column. Provide input as the column name (e.g., 'Age').",
        args_schema=GenerateHistogramArgs
    )
]

# ------------------------------
# Initialize LangChain agent with Google Gemini
# ------------------------------
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ------------------------------
# Define the API endpoint to handle queries
# ------------------------------
@app.get("/query/")
def handle_query(query: str):
    """
    Processes user queries using the LangChain agent.
    The agent is invoked with handle_parsing_errors=True to manage output parsing errors.
    """
    response = agent.invoke(query, handle_parsing_errors=True)
    return {"response": response}
