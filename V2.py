import streamlit as st
from openai import OpenAI
import psycopg2
from typing import Dict, List, Tuple
from datetime import datetime
import numpy as np
from typing import List, Dict
import os

# Configuration
OPENAI_API_KEY = st.secrets["openai_api_key"]
DB_CONN = st.secrets["database_url"]
TABLE_NAME = 'cyber'
EMBEDDING_MODEL = "text-embedding-ada-002"

# Default System Instruction
DEFAULT_SYSTEM_INSTRUCTION = """You are an AI assistant specialized in cybersecurity incident analysis. Your task is to analyze the given query and related cybersecurity data, and provide a focused, relevant response. Follow these guidelines:

1. Analyze the user's query carefully to understand the specific cybersecurity concern or question.
2. Search through all provided data columns to find information relevant to the query.
3. Use the following analysis framework as appropriate to the query:
   - Threat Assessment: Identify and assess potential threats or security issues.
   - Incident Analysis: Analyze relevant incidents, looking for patterns or connections.
   - Temporal Analysis: Consider timing of events if relevant to the query.
   - Geographical Considerations: Analyze geographical patterns or risks if location data is provided and relevant.
   - User and System Involvement: Assess involvement of users, systems, or networks as pertinent to the query.
   - Data Source Evaluation: Consider the reliability and relevance of data sources if this impacts the analysis.
   - Compliance and Policy: Mention compliance issues or policy violations only if directly relevant.
4. Provide actionable recommendations to the query and the data found.
5. Structure your response to directly address the user's query.
6. Be concise and to the point.
7. If certain aspects of the analysis are not relevant to the query, omit them from your response.
"""

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

class QueryAnalyzer:
    def analyze_query(self, query: str, available_columns: List[str]) -> Dict:
        """Analyze the user query to determine relevant columns and query intention"""
        try:
            current_instruction = st.session_state.system_instruction
            
            prompt = f"""
{current_instruction}

Please analyze this query: "{query}"

Available columns in the database: {', '.join(available_columns)}

Based on the above system instructions and considering cybersecurity context, extract and return a JSON object with the following information:
1. The most relevant columns for this query (only from the available columns list)
2. The main focus of the query from a cybersecurity perspective
3. Any specific data points or metrics mentioned that relate to security incidents
4. Any time frame mentioned
5. Any specific filtering criteria for security analysis

Format the response as a JSON object with these exact keys:
{{
    "relevant_columns": [], # list of column names from available_columns that are most relevant
    "query_focus": "", # main topic or purpose of the query from security perspective
    "specific_data_points": [], # list of specific security-related data points mentioned
    "time_frame": "", # time period mentioned, if any
    "filter_criteria": [] # any specific filtering criteria for security analysis
}}
"""
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": current_instruction},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            
            return eval(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Error analyzing query: {str(e)}")
            return {
                "relevant_columns": [],
                "query_focus": "",
                "specific_data_points": [],
                "time_frame": "",
                "filter_criteria": []
            }

class DatabaseQuerier:
    def __init__(self):
        self.conn = None
        self.available_columns = []

    def connect_to_database(self):
        """Create connection to database"""
        try:
            if not DB_CONN:
                raise ValueError("Database connection string not found")
            self.conn = psycopg2.connect(DB_CONN)
            return True
        except Exception as e:
            st.error(f"Database connection error: {str(e)}")
            return False

    def close_connection(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def get_available_columns(self, table_name: str) -> List[str]:
        """Get list of available columns from the specified table"""
        if not self.conn:
            return []
        
        try:
            with self.conn.cursor() as cur:
                # First, ensure vector extension is available
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                self.conn.commit()
                
                cur.execute(f"""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = %s
                """, (table_name,))
                self.available_columns = [row[0] for row in cur.fetchall()]
                return self.available_columns
        except Exception as e:
            st.error(f"Error fetching columns: {str(e)}")
            return []

    def search_similar_records(self, query_embedding: List[float], relevant_columns: List[str], 
                             table_name: str, limit: int = 5) -> List[Dict]:
        """Search for similar records based on embedding"""
        if not self.conn:
            return []
        
        try:
            with self.conn.cursor() as cur:
                columns_str = ", ".join(relevant_columns) if relevant_columns else "*"
                
                cur.execute(f"""
                    SELECT {columns_str}
                    FROM {table_name}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, limit))
                
                columns = [desc[0] for desc in cur.description]
                results = cur.fetchall()
                
                return [dict(zip(columns, row)) for row in results]
        except Exception as e:
            st.error(f"Error searching records: {str(e)}")
            return []

def get_embedding(text: str) -> List[float]:
    """Get embedding for text using OpenAI's embedding model"""
    try:
        # Include system instruction context if not already present
        if not text.startswith("Context:"):
            text = f"Context: {st.session_state.system_instruction}\n{text}"
            
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error getting embedding: {str(e)}")
        return []

def format_response(query: str, data: List[Dict], analysis: Dict) -> str:
    """Format the data for LLM response generation"""
    formatted_text = f"""
System Context: Using the following instruction for analysis:
{st.session_state.system_instruction}

Query: {query}

Analysis Focus: {analysis['query_focus']}
Time Frame: {analysis.get('time_frame', 'Not specified')}
Security Context: {analysis.get('specific_data_points', [])}

Retrieved Data:
"""
    for idx, record in enumerate(data, 1):
        formatted_text += f"\nRecord {idx}:\n"
        for col, val in record.items():
            formatted_text += f"{col}: {val}\n"
    
    return formatted_text

def get_llm_response(query: str, formatted_data: str) -> str:
    """Get response from OpenAI based on the query and formatted data"""
    try:
        current_instruction = st.session_state.system_instruction
        
        prompt = f"""
As a data analyst, please analyze the following query and data to provide insights:

{formatted_data}

{current_instruction}

Format your response professionally and support your analysis with specific data points.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": current_instruction},
                {"role": "user", "content": formatted_data}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting AI response: {str(e)}"

def check_environment():
    """Check if all required environment variables are set"""
    missing = []
    if not DB_CONN:
        missing.append("DATABASE_URL")
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    
    if missing:
        st.error(f"Missing required environment variables: {', '.join(missing)}")
        return False
    return True

def process_query(query: str, table_name: str) -> Tuple[List[Dict], Dict]:
    """Process a natural language query and return relevant data"""
    analyzer = QueryAnalyzer()
    querier = DatabaseQuerier()
    
    if not querier.connect_to_database():
        return [], {}
    
    try:
        # Get available columns
        available_columns = querier.get_available_columns(table_name)
        
        # Analyze the query with system instructions
        analysis = analyzer.analyze_query(query, available_columns)
        
        # Get embedding for the query with context
        query_with_context = f"""
Context: {st.session_state.system_instruction}
Query: {query}
Analysis Focus: {analysis['query_focus']}
"""
        query_embedding = get_embedding(query_with_context)
        
        # Search for similar records
        results = querier.search_similar_records(
            query_embedding,
            analysis['relevant_columns'],
            table_name
        )
        
        return results, analysis
        
    finally:
        querier.close_connection()

def main():
    st.set_page_config(page_title="Cybersecurity Query System", layout="wide")
    
    # Initialize session state
    if 'system_instruction' not in st.session_state:
        st.session_state.system_instruction = DEFAULT_SYSTEM_INSTRUCTION
    if 'show_instruction_editor' not in st.session_state:
        st.session_state.show_instruction_editor = False

    # Sidebar for system instructions
 
        # Instruction Editor Toggle
        st.subheader("System Instructions")
        toggle_instructions = st.button(
            "View/Edit Instructions" if not st.session_state.show_instruction_editor 
            else "Hide Instructions"
        )
        
        if toggle_instructions:
            st.session_state.show_instruction_editor = not st.session_state.show_instruction_editor
        
        # Show instruction editor when toggled
        if st.session_state.show_instruction_editor:
            st.markdown("---")
            instruction_text = st.text_area(
                "Modify System Instructions:",
                value=st.session_state.system_instruction,
                height=400,
                key="instruction_editor"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Update", use_container_width=True):
                    st.session_state.system_instruction = instruction_text
                    st.success("Instructions updated")
            
            with col2:
                if st.button("Reset Default", use_container_width=True):
                    st.session_state.system_instruction = DEFAULT_SYSTEM_INSTRUCTION
                    st.success("Reset to default")
        
      

    # Main content area
    st.title("Cyber Security Query System")
    
    if not check_environment():
        st.stop()
    
    # Query input area
    query = st.text_input(
        "What would you like to know?",
        placeholder="Enter your cybersecurity query here..."
    )

    if query:
        with st.spinner("Processing your query..."):
            results, analysis = process_query(query, TABLE_NAME)
            
            if results:
                formatted_data = format_response(query, results, analysis)
                response = get_llm_response(query, formatted_data)

                # Display results in tabs
                tab1, tab2, tab3 = st.tabs(["Analysis", "Raw Data", "Query Details"])

                with tab1:
                    st.markdown("Analysis")
                    st.write(response)

                with tab2:
                    st.markdown(" Retrieved Records")
                    st.json(results)
                    
                    st.markdown(" Query Analysis")
                    st.json(analysis)
                    
                with tab3:
                    st.markdown(" System Configuration")
                    with st.expander("View Current System Instructions"):
                        st.write(st.session_state.system_instruction)
                    
                    st.markdown("Analysis Parameters")
                    st.json({
                        "query_focus": analysis['query_focus'],
                        "time_frame": analysis.get('time_frame', 'Not specified'),
                        "data_points": analysis.get('specific_data_points', [])
                    })
            else:
                st.warning("No data found matching your query. Please try rephrasing your question.")

if __name__ == "__main__":
    main()
