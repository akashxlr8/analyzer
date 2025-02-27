import streamlit as st
import pandas as pd
from llm_analyzer import DatasetAnalyzer
from logging_config import get_logger

logger = get_logger("app")

def main():
    st.title('CSV File Analyzer with AI Insights')
    
    # Initialize the analyzer
    analyzer = DatasetAnalyzer()
    
    # File upload widget
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
    
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Display basic information about the dataset
            st.subheader('Dataset Info')
            st.write(f'Number of rows: {df.shape[0]}')
            st.write(f'Number of columns: {df.shape[1]}')
            
            # Display column names
            st.subheader('Columns')
            st.write(df.columns.tolist())
            
            # Preview the data
            st.subheader('Data Preview')
            st.dataframe(df.head())
            
            # Basic statistics
            st.subheader('Statistical Summary')
            st.write(df.describe())
            
            # AI-Generated Questions
            st.subheader('AI-Generated Analysis Questions')
            with st.spinner('Generating questions about your dataset...'):
                questions = analyzer.generate_questions(df)
                for i, q in enumerate(questions, 1):
                    with st.expander(f"{i}. {q.question}"):
                        st.write(f"**Category:** {q.category}")
                        st.write(f"**Reasoning:** {q.reasoning}")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == '__main__':
    main()
