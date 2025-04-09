# SHL Assessment Recommendation System
# Main application file (app.py)

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import requests
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import time
import io
import groq
import textwrap

#Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


# Initialize Groq client
class GroqClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = groq.Client(api_key=api_key) if api_key else None
        
    def is_configured(self):
        return self.client is not None
        
    def process(self, prompt, max_tokens=1000):
        if not self.is_configured():
            return "Groq API key not configured."
        try:
            response = self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that specializes in HR and recruitment."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error processing with Groq API: {str(e)}"

# SHL Assessment Recommendation System class
class SHLRecommendationSystem:
    def __init__(self, model_name="all-MiniLM-L6-v2", cache_dir="cache"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.assessment_data = None
        self.assessment_embeddings = None
        self.groq_client = None
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        # Load the sentence transformer model
        self.load_model()
        
        # Create or load assessment data
        self.load_assessment_data()
        
        # Create or load assessment embeddings
        self.load_assessment_embeddings()
        
    def set_groq_client(self, groq_client):
        """Set the Groq client for enhanced processing"""
        self.groq_client = groq_client
    
    def load_model(self):
        """Load the sentence transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            st.success(f"Model {self.model_name} loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            
    def create_assessment_data(self):
        """Create the SHL assessment dataset with official assessment categories"""
        # Official SHL assessment categories
        official_assessments = [
            {
                "id": 1,
                "name": "Behavioral Assessments",
                "description": "Evaluates work styles, preferences, and behavioral tendencies. Identifies key behaviors that drive success in specific roles.",
                "category": "Behavioral",
                "subcategory": "General Behavioral",
                "duration_minutes": 25,
                "question_count": 40,
                "url": "https://www.shl.com/solutions/products/assessments/behavioral-assessments/"
            },
            {
                "id": 2,
                "name": "Virtual Assessment & Development Centers",
                "description": "Immersive virtual experiences that simulate real-world scenarios to evaluate candidates' capabilities and potential.",
                "category": "Development",
                "subcategory": "Assessment Centers",
                "duration_minutes": 120,
                "question_count": 75,
                "url": "https://www.shl.com/solutions/products/assessments/assessment-and-development-centers/"
            },
            {
                "id": 3,
                "name": "Personality Assessment",
                "description": "Comprehensive evaluation of personality traits, work preferences, and interpersonal styles related to job performance.",
                "category": "Personality & Behavior",
                "subcategory": "Personality Profile",
                "duration_minutes": 30,
                "question_count": 60,
                "url": "https://www.shl.com/solutions/products/assessments/personality-assessment/"
            },
            {
                "id": 4,
                "name": "Cognitive Assessment",
                "description": "Measures critical reasoning and problem-solving abilities. Evaluates verbal, numerical, and logical reasoning skills.",
                "category": "Cognitive Ability",
                "subcategory": "General Cognitive",
                "duration_minutes": 35,
                "question_count": 45,
                "url": "https://www.shl.com/solutions/products/assessments/cognitive-assessments/"
            },
            {
                "id": 5,
                "name": "Skills and Simulations",
                "description": "Hands-on assessments that measure practical skills and technical capabilities in real-world contexts.",
                "category": "Technical",
                "subcategory": "Skills Assessment",
                "duration_minutes": 45,
                "question_count": 30,
                "url": "https://www.shl.com/solutions/products/assessments/skills-and-simulations/"
            },
            {
                "id": 6,
                "name": "Job-Focused Assessments",
                "description": "Job-specific assessments tailored to particular roles, combining cognitive, behavioral, and technical evaluations.",
                "category": "Role-Specific",
                "subcategory": "Job-Specific",
                "duration_minutes": 40,
                "question_count": 55,
                "url": "https://www.shl.com/solutions/products/assessments/job-focused-assessments/"
            },
            
            # Behavioral Assessments variants
            {
                "id": 7,
                "name": "Leadership Behavior Assessment",
                "description": "Identifies leadership capabilities, management style, and development areas for leadership positions.",
                "category": "Behavioral",
                "subcategory": "Leadership",
                "duration_minutes": 30,
                "question_count": 45,
                "url": "https://www.shl.com/solutions/products/assessments/behavioral-assessments/"
            },
            {
                "id": 8,
                "name": "Sales Behavior Assessment",
                "description": "Measures sales potential, client relationship capabilities, and driving business results behaviors.",
                "category": "Behavioral",
                "subcategory": "Sales",
                "duration_minutes": 25,
                "question_count": 38,
                "url": "https://www.shl.com/solutions/products/assessments/behavioral-assessments/"
            },
            {
                "id": 9,
                "name": "Collaboration and Teamwork Assessment",
                "description": "Measures ability to collaborate effectively in team environments and work well with others.",
                "category": "Behavioral",
                "subcategory": "Teamwork",
                "duration_minutes": 20,
                "question_count": 35,
                "url": "https://www.shl.com/solutions/products/assessments/behavioral-assessments/"
            },
            
            # Virtual Assessment & Development Centers variants
            {
                "id": 10,
                "name": "Leadership Development Center",
                "description": "Comprehensive virtual assessment center focusing on leadership competencies and potential.",
                "category": "Development",
                "subcategory": "Leadership Development",
                "duration_minutes": 180,
                "question_count": 90,
                "url": "https://www.shl.com/solutions/products/assessments/assessment-and-development-centers/"
            },
            {
                "id": 11,
                "name": "Graduate Assessment Center",
                "description": "Virtual assessment center designed for early career professionals and recent graduates.",
                "category": "Development",
                "subcategory": "Graduate Assessment",
                "duration_minutes": 120,
                "question_count": 70,
                "url": "https://www.shl.com/solutions/products/assessments/assessment-and-development-centers/"
            },
            
            # Personality Assessment variants
            {
                "id": 12,
                "name": "Occupational Personality Questionnaire",
                "description": "Comprehensive personality assessment evaluating 32 characteristics relevant to workplace performance.",
                "category": "Personality & Behavior",
                "subcategory": "Occupational Personality",
                "duration_minutes": 35,
                "question_count": 104,
                "url": "https://www.shl.com/solutions/products/assessments/personality-assessment/"
            },
            {
                "id": 13,
                "name": "Motivation Questionnaire",
                "description": "Evaluates what energizes and drives an individual in the workplace, helping identify cultural fit.",
                "category": "Personality & Behavior",
                "subcategory": "Work Motivation",
                "duration_minutes": 25,
                "question_count": 60,
                "url": "https://www.shl.com/solutions/products/assessments/personality-assessment/"
            },
            
            # Cognitive Assessment variants
            {
                "id": 14,
                "name": "Inductive Reasoning Assessment",
                "description": "Measures ability to identify patterns and relationships in abstract information and apply them to solve problems.",
                "category": "Cognitive Ability",
                "subcategory": "Inductive Reasoning",
                "duration_minutes": 25,
                "question_count": 24,
                "url": "https://www.shl.com/solutions/products/assessments/cognitive-assessments/"
            },
            {
                "id": 15,
                "name": "Numerical Reasoning Assessment",
                "description": "Evaluates ability to analyze and interpret numerical data, statistics, and make logical conclusions.",
                "category": "Cognitive Ability",
                "subcategory": "Numerical Analysis",
                "duration_minutes": 30,
                "question_count": 18,
                "url": "https://www.shl.com/solutions/products/assessments/cognitive-assessments/"
            },
            {
                "id": 16,
                "name": "Verbal Reasoning Assessment",
                "description": "Measures ability to understand and evaluate written information and make logical conclusions.",
                "category": "Cognitive Ability",
                "subcategory": "Verbal Comprehension",
                "duration_minutes": 30,
                "question_count": 30,
                "url": "https://www.shl.com/solutions/products/assessments/cognitive-assessments/"
            },
            
            # Skills and Simulations variants
            {
                "id": 17,
                "name": "Java Coding Assessment",
                "description": "Hands-on assessment of Java programming skills and problem-solving capabilities.",
                "category": "Technical",
                "subcategory": "Java Programming",
                "duration_minutes": 45,
                "question_count": 15,
                "url": "https://www.shl.com/solutions/products/assessments/skills-and-simulations/"
            },
            {
                "id": 18,
                "name": "Python Programming Assessment",
                "description": "Evaluates Python programming proficiency and problem-solving skills through practical coding challenges.",
                "category": "Technical",
                "subcategory": "Python Programming",
                "duration_minutes": 45,
                "question_count": 15,
                "url": "https://www.shl.com/solutions/products/assessments/skills-and-simulations/"
            },
            {
                "id": 19,
                "name": "Microsoft Office Skills Assessment",
                "description": "Measures proficiency in Microsoft Office applications including Excel, Word, and PowerPoint.",
                "category": "Technical",
                "subcategory": "Office Skills",
                "duration_minutes": 30,
                "question_count": 25,
                "url": "https://www.shl.com/solutions/products/assessments/skills-and-simulations/"
            },
            
            # Job-Focused Assessments variants
            {
                "id": 20,
                "name": "Customer Service & Call Center Assessment",
                "description": "Comprehensive evaluation of customer service skills, problem resolution, and communication effectiveness.",
                "category": "Role-Specific",
                "subcategory": "Customer Service",
                "duration_minutes": 35,
                "question_count": 50,
                "url": "https://www.shl.com/solutions/products/assessments/job-focused-assessments/"
            },
            {
                "id": 21,
                "name": "Sales Professional Assessment",
                "description": "Role-specific assessment for sales positions combining behavioral traits, cognitive skills, and situational judgment.",
                "category": "Role-Specific",
                "subcategory": "Sales Assessment",
                "duration_minutes": 40,
                "question_count": 45,
                "url": "https://www.shl.com/solutions/products/assessments/job-focused-assessments/"
            },
            {
                "id": 22,
                "name": "IT Professional Assessment",
                "description": "Comprehensive assessment for IT roles, evaluating technical knowledge, problem-solving, and professional skills.",
                "category": "Role-Specific",
                "subcategory": "Information Technology",
                "duration_minutes": 45,
                "question_count": 35,
                "url": "https://www.shl.com/solutions/products/assessments/job-focused-assessments/"
            }
        ]
        
        return pd.DataFrame(official_assessments)
        
    def load_assessment_data(self):
        """Load or create the assessment data"""
        cache_path = os.path.join(self.cache_dir, "assessment_data.pkl")
        
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.assessment_data = pickle.load(f)
        else:
            self.assessment_data = self.create_assessment_data()
            with open(cache_path, 'wb') as f:
                pickle.dump(self.assessment_data, f)
                
    def create_assessment_embeddings(self):
        """Create embeddings for all assessments"""
        # Combine name, description, category and subcategory for a richer embedding
        texts = []
        for _, row in self.assessment_data.iterrows():
            text = f"{row['name']}. {row['description']} Category: {row['category']}. Subcategory: {row['subcategory']}."
            texts.append(text)
            
        embeddings = self.model.encode(texts)
        return embeddings
        
    def load_assessment_embeddings(self):
        """Load or create assessment embeddings"""
        cache_path = os.path.join(self.cache_dir, "assessment_embeddings.pkl")
        
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.assessment_embeddings = pickle.load(f)
        else:
            self.assessment_embeddings = self.create_assessment_embeddings()
            with open(cache_path, 'wb') as f:
                pickle.dump(self.assessment_embeddings, f)
                
    def preprocess_text(self, text):
        """Preprocess the input text"""
        # Tokenize the text
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token for token in tokens if token not in stop_words and token.isalnum()]
        
        # Join the tokens back into a string
        preprocessed_text = ' '.join(filtered_tokens)
        
        return preprocessed_text
        
    def extract_text_from_url(self, url):
        """Extract text content from a URL"""
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract text from paragraphs, headings, and list items
            paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
            text = ' '.join([p.get_text() for p in paragraphs])
            
            return text
        except Exception as e:
            return f"Error extracting text from URL: {str(e)}"
            
    def enhance_with_groq(self, job_description):
        """Use Groq API to enhance job description analysis"""
        if not self.groq_client or not self.groq_client.is_configured():
            return job_description
            
        prompt = f"""
        Analyze this job description and extract key skills, requirements, and competencies 
        that would be relevant for assessment selection. Focus on cognitive abilities, personality traits,
        and job-specific skills:
        
        {job_description}
        
        Provide a concise summary that highlights the most important requirements for assessment selection.
        """
        
        enhanced_description = self.groq_client.process(prompt)
        
        # Combine original and enhanced description for better matching
        combined = f"{job_description}\n\nEnhanced Analysis:\n{enhanced_description}"
        return combined
            
    def recommend_assessments(self, job_description, top_k=5, min_duration=0, max_duration=1000, enhance=False):
        """Recommend assessments based on job description"""
        # Enhance job description using Groq if requested
        if enhance and self.groq_client and self.groq_client.is_configured():
            job_description = self.enhance_with_groq(job_description)
            
        # Preprocess the job description
        preprocessed_text = self.preprocess_text(job_description)
        
        # Create embedding for the job description
        job_embedding = self.model.encode([preprocessed_text])[0]
        
        # Calculate similarity scores
        similarity_scores = cosine_similarity([job_embedding], self.assessment_embeddings)[0]
        
        # Create a DataFrame with assessments and their similarity scores
        results = self.assessment_data.copy()
        results['similarity_score'] = similarity_scores
        
        # Filter by duration if specified
        duration_filtered = results[(results['duration_minutes'] >= min_duration) & 
                                  (results['duration_minutes'] <= max_duration)]
        
        # Sort by similarity score and get top k results
        top_results = duration_filtered.sort_values('similarity_score', ascending=False).head(top_k)
        
        return top_results.reset_index(drop=True)
    
    def explain_recommendation(self, job_description, assessment_id):
        """Use Groq to explain why a specific assessment is recommended"""
        if not self.groq_client or not self.groq_client.is_configured():
            return "Groq API key not configured. Cannot provide detailed explanation."
            
        assessment = self.assessment_data[self.assessment_data['id'] == assessment_id].iloc[0]
        
        prompt = f"""
        I need to explain why the following assessment is recommended for a job with this description:
        
        Job Description:
        {job_description}
        
        Assessment:
        Name: {assessment['name']}
        Description: {assessment['description']}
        Category: {assessment['category']}
        Subcategory: {assessment['subcategory']}
        
        Provide a concise explanation of why this assessment is relevant for this job,
        highlighting specific skills or qualities measured by the assessment that match requirements in the job description.
        """
        
        explanation = self.groq_client.process(prompt, max_tokens=500)
        return explanation
    
    def generate_api_response(self, job_description, top_k=5, min_duration=0, max_duration=1000, enhance=False):
        """Generate a JSON response for API usage"""
        recommendations = self.recommend_assessments(job_description, top_k, min_duration, max_duration, enhance)
        
        # Convert to dictionary for JSON serialization
        results = []
        for _, row in recommendations.iterrows():
            result = {
                "id": int(row['id']),
                "name": row['name'],
                "description": row['description'],
                "category": row['category'],
                "subcategory": row['subcategory'],
                "duration_minutes": int(row['duration_minutes']),
                "question_count": int(row['question_count']),
                "similarity_score": float(row['similarity_score']),
                "url": row['url']
            }
            results.append(result)
            
        response = {
            "timestamp": time.time(),
            "query": job_description,
            "parameters": {
                "top_k": top_k,
                "min_duration": min_duration,
                "max_duration": max_duration,
                "enhanced": enhance
            },
            "results": results
        }
        
        return response
        
# Initialize the Streamlit app
def main():
    st.set_page_config(
        page_title="SHL Assessment Recommendation System",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("SHL Assessment Recommendation System")
    st.subheader("Match job requirements to appropriate assessment tools")
    
    # Initialize session state variables if they don't exist
    if 'recommendation_system' not in st.session_state:
        st.session_state.recommendation_system = SHLRecommendationSystem()
        
    if 'groq_client' not in st.session_state:
        st.session_state.groq_client = None
        
    # API Key configuration in sidebar
    with st.sidebar:
        st.header("Configuration")
        groq_api_key = st.text_input("Enter Groq API Key", type="password")
        
        if groq_api_key:
            st.session_state.groq_client = GroqClient(groq_api_key)
            st.session_state.recommendation_system.set_groq_client(st.session_state.groq_client)
            
            if st.session_state.groq_client.is_configured():
                st.success("Groq API configured successfully!")
            else:
                st.error("Failed to configure Groq API")
        else:
            st.info("Enter your Groq API key for enhanced analysis")
            
        st.divider()
        st.markdown("### Filtering Options")
        min_duration = st.slider("Minimum Duration (minutes)", 0, 120, 0)
        max_duration = st.slider("Maximum Duration (minutes)", 0, 120, 120)
        top_k = st.slider("Number of Recommendations", 1, 20, 5)
        use_groq = st.checkbox("Enhance with Groq Analysis", value=True)
        
    # Create tabs for different input methods
    tabs = st.tabs(["Text Input", "File Upload", "URL Input", "API Documentation"])
    
    # Text Input Tab
    with tabs[0]:
        st.header("Enter Job Description")
        job_description = st.text_area("Paste job description here:", height=200)
        
        if st.button("Generate Recommendations", key="text_button"):
            if job_description:
                with st.spinner("Analyzing job description and generating recommendations..."):
                    try:
                        # Only use Groq enhancement if API key is configured and enhancement is checked
                        enhance = use_groq and st.session_state.groq_client and st.session_state.groq_client.is_configured()
                        
                        recommendations = st.session_state.recommendation_system.recommend_assessments(
                            job_description, 
                            top_k=top_k, 
                            min_duration=min_duration, 
                            max_duration=max_duration,
                            enhance=enhance
                        )
                        
                        display_recommendations(recommendations, job_description)
                    except Exception as e:
                        st.error(f"Error generating recommendations: {str(e)}")
            else:
                st.warning("Please enter a job description")
                
    # File Upload Tab
    with tabs[1]:
        st.header("Upload Job Description File")
        uploaded_file = st.file_uploader("Upload a TXT file containing job description", type=["txt"])
        
        if uploaded_file is not None:
            # Read the file
            file_content = uploaded_file.read().decode("utf-8")
            st.subheader("File Content Preview")
            st.text_area("Preview", file_content[:500] + ("..." if len(file_content) > 500 else ""), height=150)
            
            if st.button("Generate Recommendations", key="file_button"):
                with st.spinner("Analyzing job description and generating recommendations..."):
                    try:
                        # Only use Groq enhancement if API key is configured and enhancement is checked
                        enhance = use_groq and st.session_state.groq_client and st.session_state.groq_client.is_configured()
                        
                        recommendations = st.session_state.recommendation_system.recommend_assessments(
                            file_content, 
                            top_k=top_k, 
                            min_duration=min_duration, 
                            max_duration=max_duration,
                            enhance=enhance
                        )
                        
                        display_recommendations(recommendations, file_content)
                    except Exception as e:
                        st.error(f"Error generating recommendations: {str(e)}")
                        
    # URL Input Tab
    with tabs[2]:
        st.header("Extract Job Description from URL")
        url = st.text_input("Enter URL containing job description:")
        
        if st.button("Extract and Generate Recommendations", key="url_button"):
            if url:
                with st.spinner("Extracting content from URL and generating recommendations..."):
                    try:
                        # Extract text from the URL
                        url_content = st.session_state.recommendation_system.extract_text_from_url(url)
                        
                        if url_content.startswith("Error"):
                            st.error(url_content)
                        else:
                            st.subheader("Extracted Content Preview")
                            st.text_area("Preview", url_content[:500] + ("..." if len(url_content) > 500 else ""), height=150)
                            
                            # Only use Groq enhancement if API key is configured and enhancement is checked
                            enhance = use_groq and st.session_state.groq_client and st.session_state.groq_client.is_configured()
                            
                            recommendations = st.session_state.recommendation_system.recommend_assessments(
                                url_content, 
                                top_k=top_k, 
                                min_duration=min_duration, 
                                max_duration=max_duration,
                                enhance=enhance
                            )
                            
                            display_recommendations(recommendations, url_content)
                    except Exception as e:
                        st.error(f"Error processing URL: {str(e)}")
            else:
                st.warning("Please enter a URL")
                
    # API Documentation Tab
    with tabs[3]:
        st.header("API Documentation")
        st.markdown("""
        ### API Usage
        
        The SHL Assessment Recommendation System can be used programmatically via API calls.
        
        #### Endpoint
        ```
        POST /api/recommend
        ```
        
        #### Request Parameters
        | Parameter | Type | Description |
        |-----------|------|-------------|
        | `job_description` | string | The job description text |
        | `top_k` | integer | Number of recommendations to return (optional, default=5) |
        | `min_duration` | integer | Minimum assessment duration in minutes (optional, default=0) |
        | `max_duration` | integer | Maximum assessment duration in minutes (optional, default=1000) |
        | `enhance` | boolean | Whether to use Groq to enhance recommendations (optional, default=false) |
        
        #### Example Request
        ```json
        {
            "job_description": "Looking for a software developer with strong analytical skills...",
            "top_k": 3,
            "min_duration": 20,
            "max_duration": 40,
            "enhance": true
        }
        ```
        
        #### Example Response
        ```json
        {
            "timestamp": 1617293845.123,
            "query": "Looking for a software developer with strong analytical skills...",
            "parameters": {
                "top_k": 3,
                "min_duration": 20,
                "max_duration": 40,
                "enhanced": true
            },
            "results": [
                {
                    "id": 10,
                    "name": "IT Aptitude Series",
                    "description": "Measures aptitudes specific to information technology roles.",
                    "category": "Role-Specific",
                    "subcategory": "Information Technology",
                    "duration_minutes": 45,
                    "question_count": 35,
                    "similarity_score": 0.89,
                    "url": "https://www.shl.com/solutions/products/assessments/job-focused-assessments/"
                },
                // Additional results...
            ]
        }
        ```
        
        #### Implementation Example (Python)
        ```python
        import requests
        import json
        
        url = "http://your-server/api/recommend"
        payload = {
            "job_description": "Looking for a software developer...",
            "top_k": 3
        }
        
        response = requests.post(url, json=payload)
        results = response.json()
        print(results)
        ```
        """)
        
        # Example API Response
        st.subheader("Try the API Response Generator")
        api_description = st.text_area("Enter a job description to test API response:", height=150)
        
        if st.button("Generate Sample API Response"):
            if api_description:
                with st.spinner("Generating API response..."):
                    try:
                        # Only use Groq enhancement if API key is configured and enhancement is checked
                        enhance = use_groq and st.session_state.groq_client and st.session_state.groq_client.is_configured()
                        
                        api_response = st.session_state.recommendation_system.generate_api_response(
                            api_description,
                            top_k=top_k,
                            min_duration=min_duration,
                            max_duration=max_duration,
                            enhance=enhance
                        )
                        
                        st.subheader("API Response")
                        st.json(api_response)
                        
                        # Option to download the response as JSON
                        json_str = json.dumps(api_response, indent=2)
                        st.download_button(
                            label="Download JSON Response",
                            data=json_str,
                            file_name="shl_recommendations.json",
                            mime="application/json"
                        )
                    except Exception as e:
                        st.error(f"Error generating API response: {str(e)}")
            else:
                st.warning("Please enter a job description")


def display_recommendations(recommendations, job_description):
    """Display recommendation results"""
    if recommendations.empty:
        st.warning("No matching assessments found. Try adjusting your filters.")
        return
        
    st.success(f"Found {len(recommendations)} matching assessments!")
    
    # Display each recommendation in an expandable card
    for i, (_, assessment) in enumerate(recommendations.iterrows()):
        with st.expander(f"{i+1}. {assessment['name']} (Score: {assessment['similarity_score']:.2f})", expanded=(i==0)):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Description:** {assessment['description']}")
                st.markdown(f"**Category:** {assessment['category']} - {assessment['subcategory']}")
                
                # If Groq is configured, provide a button to explain the recommendation
                if st.session_state.groq_client and st.session_state.groq_client.is_configured():
                    if st.button(f"Explain this recommendation", key=f"explain_{assessment['id']}"):
                        with st.spinner("Generating explanation..."):
                            explanation = st.session_state.recommendation_system.explain_recommendation(
                                job_description, assessment['id'])
                            st.markdown("### Explanation")
                            st.markdown(explanation)
                            
            with col2:
                st.metric("Duration", f"{assessment['duration_minutes']} min")
                st.metric("Questions", f"{assessment['question_count']}")
                
            # Display similarity as a progress bar
            st.markdown("**Relevance:**")
            st.progress(float(assessment['similarity_score']))
    
    # Option to download results as CSV
    csv = recommendations.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="shl_recommendations.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()