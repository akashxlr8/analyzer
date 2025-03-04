import streamlit as st
import pandas as pd
from llm_analyzer import DatasetAnalyzer
from logging_config import get_logger
from llm_with_tool import CodeEnabledLLM
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
            
            # After showing the AI-generated questions, add a custom question input
            st.subheader('Add Your Own Analysis Question')
            custom_question = st.text_input("Enter your own analytical question:")
            custom_category = st.selectbox(
                "Select a category for your question:",
                ['Trend Analysis', 'Correlation', 'Distribution', 'Outliers', 'Pattern Recognition', 'Custom']
            )
            custom_reasoning = st.text_area("Explain your reasoning for this question:")
            add_question = st.button("Add Question")

            if add_question and custom_question:
                # Create a custom AnalyticalQuestion object
                from llm_analyzer import AnalyticalQuestion
                custom_q = AnalyticalQuestion(
                    question=custom_question,
                    category=custom_category or "Custom",
                    reasoning=custom_reasoning or "No Reasoning Provided"
                )
                # Append to the list of questions
                if 'questions' not in st.session_state:
                    st.session_state.questions = questions
                st.session_state.questions.append(custom_q)
                st.success("Question added successfully!")
                
            # Display all questions (including custom ones)
            st.subheader('All Analysis Questions')
            questions_to_display = st.session_state.get('questions', questions)
            for i, q in enumerate(questions_to_display, 1):
                with st.expander(f"{i}. {q.question}"):
                    st.write(f"**Category:** {q.category}")
                    st.write(f"**Reasoning:** {q.reasoning}")
            
            # After displaying all questions, add a button to analyze with LLM+Calculator
            st.subheader('Analyze Questions with AI')
            selected_question = st.selectbox(
                "Select a question to analyze with AI:",
                [q.question for q in st.session_state.get('questions', questions)]
            )
            analyze_button = st.button("Analyze with Calculator-enabled AI")

            if analyze_button and selected_question:
                try:
                    
                    # Get the selected question object
                    selected_q = next(q for q in st.session_state.get('questions', questions) 
                                     if q.question == selected_question)
                    
                    # Initialize the Code-enabled LLM
                    calculator_llm = CodeEnabledLLM()
                    
                    with st.spinner("Analyzing with AI and calculator..."):
                        # Pass the dataset and question to the LLM with calculator
                        analysis = calculator_llm.analyze_question(df, selected_q.question, selected_q.category)
                        
                        # Display the analysis result in an expandable section
                        with st.expander("Analysis Result", expanded=True):
                            st.markdown(analysis)
                            st.info("This analysis was performed by an AI with access to calculation tools.")
                except ImportError:
                    st.error("The calculator-enabled LLM module is not available. Please make sure llm_with_tool.py exists.")
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == '__main__':
    main()
