# SHL Assessment Recommendation System
# app.py - Main application file

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import requests
import json
import nltk
import time
import io
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import logging
import groq
from typing import Dict, List, Any, Optional, Union, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL = "all-MiniLM-L6-v2"
CACHE_DIR = "cache"
NLTK_DATA_DIR = "nltk_data"

# Ensure directories exist
for directory in [CACHE_DIR, NLTK_DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

# Download necessary NLTK data with specific download path
nltk.data.path.append(NLTK_DATA_DIR)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', download_dir=NLTK_DATA_DIR)
    nltk.download('stopwords', download_dir=NLTK_DATA_DIR)


class GroqClient:
    """Client for interacting with Groq API for enhanced job description analysis"""
    
    def __init__(self, api_key: str):
        """Initialize the Groq client"""
        self.api_key = api_key
        self.client = groq.Client(api_key=api_key) if api_key else None
        
    def is_configured(self) -> bool:
        """Check if the client is properly configured"""
        return self.client is not None
        
    def process(self, prompt: str, max_tokens: int = 1000) -> str:
        """Process a prompt through the Groq API"""
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
            logger.error(f"Error processing with Groq API: {str(e)}")
            return f"Error processing with Groq API: {str(e)}"


class SHLRecommendationSystem:
    """System for recommending SHL assessments based on job descriptions"""
    
    def __init__(self, model_name: str = DEFAULT_MODEL, cache_dir: str = CACHE_DIR):
        """Initialize the SHL recommendation system"""
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.assessment_data = None
        self.assessment_embeddings = None
        self.groq_client = None
        
        # Create cache directory if it doesn't exist
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
            
        # Load components with graceful error handling
        self._load_model()
        self._load_assessment_data()
        self._load_assessment_embeddings()
        
    def _load_model(self) -> None:
        """Load the sentence transformer model with error handling"""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Model {self.model_name} loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            st.error(f"Error loading model: {str(e)}")
            self.model = None
    
    def set_groq_client(self, groq_client: GroqClient) -> None:
        """Set the Groq client for enhanced processing"""
        self.groq_client = groq_client
            
    def _create_assessment_data(self) -> pd.DataFrame:
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
        
    def _load_assessment_data(self) -> None:
        """Load or create the assessment data with error handling"""
        cache_path = os.path.join(self.cache_dir, "assessment_data.pkl")
        
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    self.assessment_data = pickle.load(f)
                logger.info("Assessment data loaded from cache.")
            else:
                self.assessment_data = self._create_assessment_data()
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.assessment_data, f)
                logger.info("Assessment data created and cached.")
        except Exception as e:
            logger.error(f"Error loading assessment data: {str(e)}")
            self.assessment_data = self._create_assessment_data()
            logger.info("Created fresh assessment data due to cache error.")
                
    def _create_assessment_embeddings(self) -> np.ndarray:
        """Create embeddings for all assessments with error handling"""
        if self.model is None:
            logger.error("Cannot create embeddings: model not loaded.")
            return np.array([])
            
        # Combine name, description, category and subcategory for a richer embedding
        texts = []
        for _, row in self.assessment_data.iterrows():
            text = f"{row['name']}. {row['description']} Category: {row['category']}. Subcategory: {row['subcategory']}."
            texts.append(text)
            
        try:
            embeddings = self.model.encode(texts)
            logger.info(f"Created {len(embeddings)} assessment embeddings.")
            return embeddings
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            return np.array([])
        
    def _load_assessment_embeddings(self) -> None:
        """Load or create assessment embeddings with error handling"""
        cache_path = os.path.join(self.cache_dir, "assessment_embeddings.pkl")
        
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    self.assessment_embeddings = pickle.load(f)
                logger.info("Assessment embeddings loaded from cache.")
            else:
                self.assessment_embeddings = self._create_assessment_embeddings()
                if len(self.assessment_embeddings) > 0:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(self.assessment_embeddings, f)
                    logger.info("Assessment embeddings created and cached.")
        except Exception as e:
            logger.error(f"Error loading assessment embeddings: {str(e)}")
            self.assessment_embeddings = self._create_assessment_embeddings()
            logger.info("Created fresh assessment embeddings due to cache error.")
            
    def preprocess_text(self, text: str) -> str:
        """Preprocess the input text for better matching"""
        if not text:
            return ""
            
        try:
            # Tokenize the text
            tokens = word_tokenize(text.lower())
            
            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            filtered_tokens = [token for token in tokens if token not in stop_words and token.isalnum()]
            
            # Join the tokens back into a string
            preprocessed_text = ' '.join(filtered_tokens)
            
            return preprocessed_text
        except Exception as e:
            logger.warning(f"Error in text preprocessing: {str(e)}. Using original text.")
            return text.lower()
        
    def extract_text_from_url(self, url: str) -> str:
        """Extract text content from a URL with improved error handling"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # Raise an exception for 4XX/5XX status codes
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
                
            # Extract text from paragraphs, headings, and list items (job description content)
            content_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div.job-description', 'div.description'])
            
            if not content_elements:
                # If no specific elements found, get all visible text
                text = soup.get_text(separator=' ', strip=True)
            else:
                text = ' '.join([elem.get_text(separator=' ', strip=True) for elem in content_elements])
                
            # Clean up the text: remove extra whitespace and normalize newlines
            text = ' '.join(text.split())
            
            if not text:
                return "No content extracted from URL. The page might be using JavaScript to load content or has access restrictions."
                
            return text
        except requests.exceptions.RequestException as e:
            return f"Error accessing URL: {str(e)}"
        except Exception as e:
            return f"Error extracting text from URL: {str(e)}"
            
    def enhance_with_groq(self, job_description: str) -> str:
        """Use Groq API to enhance job description analysis with error handling"""
        if not self.groq_client or not self.groq_client.is_configured():
            return job_description
            
        try:
            prompt = f"""
            Analyze this job description and extract key skills, requirements, and competencies 
            that would be relevant for assessment selection. Focus on cognitive abilities, personality traits,
            and job-specific skills:
            
            {job_description}
            
            Provide a concise summary that highlights the most important requirements for assessment selection.
            """
            
            enhanced_description = self.groq_client.process(prompt)
            
            if enhanced_description.startswith("Error"):
                logger.warning(f"Groq enhancement failed: {enhanced_description}")
                return job_description
                
            # Combine original and enhanced description for better matching
            combined = f"{job_description}\n\nEnhanced Analysis:\n{enhanced_description}"
            return combined
        except Exception as e:
            logger.error(f"Error in Groq enhancement: {str(e)}")
            return job_description
            
    def recommend_assessments(self, 
                             job_description: str, 
                             top_k: int = 5, 
                             min_duration: int = 0, 
                             max_duration: int = 1000, 
                             enhance: bool = False) -> pd.DataFrame:
        """Recommend assessments based on job description with better error handling"""
        if not job_description:
            logger.warning("Empty job description provided")
            return pd.DataFrame()
            
        if self.model is None or self.assessment_data is None or len(self.assessment_embeddings) == 0:
            logger.error("Recommendation system not properly initialized")
            return pd.DataFrame()
            
        try:
            # Enhance job description using Groq if requested
            if enhance and self.groq_client and self.groq_client.is_configured():
                job_description = self.enhance_with_groq(job_description)
                
            # Preprocess the job description
            preprocessed_text = self.preprocess_text(job_description)
            
            if not preprocessed_text:
                logger.warning("Empty preprocessed text after preprocessing")
                return pd.DataFrame()
                
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
            
            if duration_filtered.empty:
                logger.warning(f"No assessments match the duration filters: min={min_duration}, max={max_duration}")
                return pd.DataFrame()
                
            # Sort by similarity score and get top k results
            top_results = duration_filtered.sort_values('similarity_score', ascending=False).head(top_k)
            
            return top_results.reset_index(drop=True)
        except Exception as e:
            logger.error(f"Error in assessment recommendation: {str(e)}")
            return pd.DataFrame()
    
    def explain_recommendation(self, job_description: str, assessment_id: int) -> str:
        """Use Groq to explain why a specific assessment is recommended with better error handling"""
        if not self.groq_client or not self.groq_client.is_configured():
            return "Groq API key not configured. Cannot provide detailed explanation."
            
        try:
            assessment_df = self.assessment_data[self.assessment_data['id'] == assessment_id]
            if assessment_df.empty:
                return f"Assessment with ID {assessment_id} not found."
                
            assessment = assessment_df.iloc[0]
            
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
        except Exception as e:
            logger.error(f"Error explaining recommendation: {str(e)}")
            return f"Error generating explanation: {str(e)}"
    
    def generate_api_response(self, 
                             job_description: str, 
                             top_k: int = 5, 
                             min_duration: int = 0, 
                             max_duration: int = 1000, 
                             enhance: bool = False) -> Dict[str, Any]:
        """Generate a JSON response for API usage with error handling"""
        try:
            recommendations = self.recommend_assessments(
                job_description, top_k, min_duration, max_duration, enhance)
            
            # Check if recommendations is empty
            if recommendations.empty:
                return {
                    "timestamp": time.time(),
                    "query": job_description,
                    "parameters": {
                        "top_k": top_k,
                        "min_duration": min_duration,
                        "max_duration": max_duration,
                        "enhanced": enhance
                    },
                    "results": [],
                    "status": "No matching assessments found"
                }
            
            # Convert to dictionary for JSON serialization
            results = []
            for _, row in recommendations.iterrows():
                result = {}
                if 'id' in row:
                    result["id"] = int(row['id'])
            
                if 'name' in row:
                    result["name"] = row['name']
                
                if 'description' in row:
                    result["description"] = row['description']
                
                if 'category' in row:
                    result["category"] = row['category']
                
                if 'subcategory' in row:
                    result["subcategory"] = row['subcategory']
                
                if 'duration_minutes' in row:
                    result["duration_minutes"] = int(row['duration_minutes'])
                
                if 'question_count' in row:
                    result["question_count"] = int(row['question_count'])
                
                if 'similarity_score' in row:
                    result["similarity_score"] = float(row['similarity_score'])
                
                if 'url' in row:
                    result["url"] = row['url']                
                
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
                "results": results,
                "status": "success"
            }
            
            return response
        except Exception as e:
            logger.error(f"Error generating API response: {str(e)}")
            return {
                "timestamp": time.time(),
                "query": job_description,
                "parameters": {
                    "top_k": top_k,
                    "min_duration": min_duration,
                    "max_duration": max_duration,
                    "enhanced": enhance
                },
                "results": [],
                "status": f"error: {str(e)}"
            }


def display_recommendations(recommendations, job_description):
    """Display recommendation results with improved visuals and error handling"""
    if recommendations.empty:
        st.warning("No matching assessments found. Try adjusting your filters.")
        return
        
    st.success(f"Found {len(recommendations)} matching assessments!")
    
    # Display each recommendation in an expandable card
    for i, (_, assessment) in enumerate(recommendations.iterrows()):
        # Make sure all required keys exist
        if 'name' not in assessment or 'similarity_score' not in assessment:
            st.error(f"Assessment data at index {i} is missing required fields")
            continue
            
        with st.expander(f"{i+1}. {assessment['name']} (Score: {assessment['similarity_score']:.2f})", expanded=(i==0)):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if 'description' in assessment:
                    st.markdown(f"**Description:** {assessment['description']}")
                
                if 'category' in assessment and 'subcategory' in assessment:
                    st.markdown(f"**Category:** {assessment['category']} - {assessment['subcategory']}")
                
                # If Groq is configured, provide a button to explain the recommendation
                if 'groq_client' in st.session_state and st.session_state.groq_client and st.session_state.groq_client.is_configured():
                    if st.button(f"Explain this recommendation", key=f"explain_{assessment.get('id', i)}"):
                        with st.spinner("Generating explanation..."):
                            explanation = st.session_state.recommendation_system.explain_recommendation(
                                job_description, assessment.get('id', i))
                            if explanation.startswith("Error"):
                                st.error(explanation)
                            else:
                                st.markdown("### Explanation")
                                st.markdown(explanation)
                            
            with col2:
                if 'duration_minutes' in assessment:
                    st.metric("Duration", f"{assessment['duration_minutes']} min")
                    
                if 'question_count' in assessment:
                    st.metric("Questions", f"{assessment['question_count']}")
                
                # Check if URL exists before creating link
                if 'url' in assessment and assessment['url']:
                    # Instead of "Learn more", use "Visit Assessment"
                    st.markdown(f"[Visit Assessment]({assessment['url']})", unsafe_allow_html=True)
                
            # Display similarity as a progress bar
            st.markdown("**Relevance:**")
            st.progress(float(assessment['similarity_score']))
    
    # Option to download results as CSV
    csv = recommendations.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="shl_recommendations.csv",
        mime="text/csv"
    )

# Initialize the Streamlit app
# This completes the main() function and adds any missing code

# Initialize the Streamlit app
def main():
    st.set_page_config(
        page_title="SHL Assessment Recommendation System",
        page_icon="🎯",
        layout="wide"
    )
    
    # Add custom CSS for better layout
    st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f0ff;
    }
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("SHL Assessment Recommendation System")
    st.subheader("Match job requirements to appropriate assessment tools")
    
    # Initialize session state variables if they don't exist
    if 'recommendation_system' not in st.session_state:
        st.session_state.recommendation_system = SHLRecommendationSystem()
    
    if 'job_description' not in st.session_state:
        st.session_state.job_description = ""
    
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = pd.DataFrame()
    
    # Sidebar for API key configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Groq API key input
        groq_api_key = st.text_input(
            "Groq API Key (Optional)", 
            type="password",
            help="Enter your Groq API key to enable enhanced analysis capabilities."
        )
        
        # Initialize or update Groq client when API key changes
        if groq_api_key:
            if 'groq_client' not in st.session_state or st.session_state.groq_client.api_key != groq_api_key:
                st.session_state.groq_client = GroqClient(groq_api_key)
                st.session_state.recommendation_system.set_groq_client(st.session_state.groq_client)
                st.success("Groq API configured successfully!")
        else:
            st.session_state.groq_client = None
            st.session_state.recommendation_system.set_groq_client(None)
        
        # Model selection (for future expansion)
        selected_model = st.selectbox(
            "Embedding Model",
            [DEFAULT_MODEL],
            disabled=True,
            help="Currently using the default model. More models will be available in future updates."
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This system recommends SHL assessments based on job descriptions using NLP and semantic similarity.
        
        Upload job descriptions, paste text, or provide URLs to find the most appropriate assessments.
        """)
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Text Input", "📄 File Upload", "🔗 URL Input", "ℹ️ API Documentation"])
    
    with tab1:
        st.header("Job Description Text Input")
        
        # Text area for job description
        job_description = st.text_area(
            "Enter Job Description",
            height=250,
            placeholder="Paste the job description here...",
            value=st.session_state.job_description
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Filter options
            st.subheader("Filter Options")
            top_k = st.slider("Number of Results", min_value=1, max_value=20, value=5)
            min_duration, max_duration = st.slider(
                "Assessment Duration (minutes)",
                min_value=0,
                max_value=180,
                value=(0, 180)
            )
        
        with col2:
            # Enhanced analysis option
            st.subheader("Analysis Options")
            use_enhanced = st.checkbox(
                "Use Enhanced Analysis",
                value=bool(st.session_state.groq_client),
                disabled=not bool(st.session_state.groq_client),
                help="Analyze job descriptions with Groq for better matching (requires API key)"
            )
            
            st.markdown("")  # Spacing
            st.markdown("")  # Spacing
            
            # Search button
            if st.button("Find Matching Assessments", type="primary", use_container_width=True):
                if not job_description:
                    st.error("Please enter a job description.")
                else:
                    with st.spinner("Analyzing job description..."):
                        try:
                            # Update session state
                            st.session_state.job_description = job_description
                            
                            # Get recommendations
                            recommendations = st.session_state.recommendation_system.recommend_assessments(
                                job_description,
                                top_k=top_k,
                                min_duration=min_duration,
                                max_duration=max_duration,
                                enhance=use_enhanced
                            )
                            
                            st.session_state.recommendations = recommendations
                            
                            # Show results directly in this tab
                            if not recommendations.empty:
                                st.subheader("Assessment Recommendations")
                                display_recommendations(recommendations, job_description)
                            else:
                                st.warning("No matching assessments found. Try adjusting your filters.")
                                
                        except Exception as e:
                            st.error(f"Error processing request: {str(e)}")
                            logger.error(f"Error in tab1: {str(e)}")
    
    with tab2:
        st.header("Upload Job Description File")
        
        uploaded_file = st.file_uploader("Choose a text file", type=['txt', 'pdf', 'docx'])
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Filter options
            st.subheader("Filter Options")
            file_top_k = st.slider("Number of Results", min_value=1, max_value=20, value=5, key="file_top_k")
            file_min_duration, file_max_duration = st.slider(
                "Assessment Duration (minutes)",
                min_value=0,
                max_value=180,
                value=(0, 180),
                key="file_duration"
            )
        
        with col2:
            # Enhanced analysis option
            st.subheader("Analysis Options")
            file_use_enhanced = st.checkbox(
                "Use Enhanced Analysis",
                value=bool(st.session_state.groq_client),
                disabled=not bool(st.session_state.groq_client),
                help="Analyze job descriptions with Groq for better matching (requires API key)",
                key="file_enhanced"
            )
        
        if uploaded_file is not None:
            try:
                # Process uploaded file
                with st.spinner("Processing file..."):
                    # For text files
                    if uploaded_file.name.endswith('.txt'):
                        file_content = uploaded_file.read().decode('utf-8')
                    # For PDF files (assuming PyPDF2 is installed)
                    elif uploaded_file.name.endswith('.pdf'):
                        st.warning("PDF processing requires additional libraries. Using text extraction only.")
                        # Note: In a real implementation, you would use PyPDF2 or similar
                        file_content = "PDF extraction not implemented in this version."
                    # For DOCX files (assuming python-docx is installed)
                    elif uploaded_file.name.endswith('.docx'):
                        st.warning("DOCX processing requires additional libraries. Using text extraction only.")
                        # Note: In a real implementation, you would use python-docx
                        file_content = "DOCX extraction not implemented in this version."
                    else:
                        file_content = "Unsupported file type."
                        
                    # Display extracted content
                    with st.expander("Extracted Content", expanded=False):
                        st.text_area("File Content", value=file_content, height=200, disabled=True)
                    
                    # Get recommendations based on file content
                    if st.button("Analyze File Content", type="primary", use_container_width=True):
                        with st.spinner("Analyzing job description..."):
                            recommendations = st.session_state.recommendation_system.recommend_assessments(
                                file_content,
                                top_k=file_top_k,
                                min_duration=file_min_duration,
                                max_duration=file_max_duration,
                                enhance=file_use_enhanced
                            )
                            
                            st.session_state.recommendations = recommendations
                            st.session_state.job_description = file_content
                            
                            # Show results
                            st.subheader("Assessment Recommendations")
                            display_recommendations(recommendations, file_content)
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                logger.error(f"Error in tab2: {str(e)}")
    
    with tab3:
        st.header("Job Description from URL")
        
        url = st.text_input("Enter Job Posting URL", placeholder="https://example.com/job-posting")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Filter options
            st.subheader("Filter Options")
            url_top_k = st.slider("Number of Results", min_value=1, max_value=20, value=5, key="url_top_k")
            url_min_duration, url_max_duration = st.slider(
                "Assessment Duration (minutes)",
                min_value=0,
                max_value=180,
                value=(0, 180),
                key="url_duration"
            )
        
        with col2:
            # Enhanced analysis option
            st.subheader("Analysis Options")
            url_use_enhanced = st.checkbox(
                "Use Enhanced Analysis",
                value=bool(st.session_state.groq_client),
                disabled=not bool(st.session_state.groq_client),
                help="Analyze job descriptions with Groq for better matching (requires API key)",
                key="url_enhanced"
            )
        
        if st.button("Fetch and Analyze URL", type="primary", use_container_width=True):
            if not url:
                st.error("Please enter a URL.")
            else:
                with st.spinner("Fetching content from URL..."):
                    try:
                        # Extract content from URL
                        url_content = st.session_state.recommendation_system.extract_text_from_url(url)
                        
                        # Display extracted content
                        with st.expander("Extracted Content", expanded=False):
                            st.text_area("URL Content", value=url_content, height=200, disabled=True)
                        
                        # Get recommendations
                        if not url_content.startswith("Error"):
                            recommendations = st.session_state.recommendation_system.recommend_assessments(
                                url_content,
                                top_k=url_top_k,
                                min_duration=url_min_duration,
                                max_duration=url_max_duration,
                                enhance=url_use_enhanced
                            )
                            
                            st.session_state.recommendations = recommendations
                            st.session_state.job_description = url_content
                            
                            # Show results
                            st.subheader("Assessment Recommendations")
                            display_recommendations(recommendations, url_content)
                        else:
                            st.error(url_content)
                    except Exception as e:
                        st.error(f"Error processing URL: {str(e)}")
                        logger.error(f"Error in tab3: {str(e)}")
    
    with tab4:
        st.header("API Documentation")
        
        st.markdown("""
        ### SHL Assessment Recommendation API
        
        This application also provides an API endpoint for programmatic access to the recommendation system.
        
        #### Endpoint
        ```
        POST /api/recommend
        ```
        
        #### Request Parameters
        ```json
        {
            "job_description": "Text of the job description",
            "top_k": 5,                 // Optional, defaults to 5
            "min_duration": 0,          // Optional, defaults to 0
            "max_duration": 180,        // Optional, defaults to 180
            "enhance": false            // Optional, requires Groq API key
        }
        ```
        
        #### Response Format
        ```json
        {
            "timestamp": 1649862420,
            "query": "job description text",
            "parameters": {
                "top_k": 5,
                "min_duration": 0,
                "max_duration": 180,
                "enhanced": false
            },
            "results": [
                {
                    "id": 4,
                    "name": "Cognitive Assessment",
                    "description": "...",
                    "category": "Cognitive Ability",
                    "subcategory": "General Cognitive",
                    "duration_minutes": 35,
                    "question_count": 45,
                    "similarity_score": 0.82,
                    "url": "https://www.shl.com/..."
                },
                // Additional results...
            ],
            "status": "success"
        }
        ```
        
        #### Implementation Example (Python)
        ```python
        import requests
        import json
        
        url = "http://your-app-url/api/recommend"
        
        payload = {
            "job_description": "Software Engineer with 5+ years experience...",
            "top_k": 3,
            "enhance": True
        }
        
        response = requests.post(url, json=payload)
        results = response.json()
        
        print(json.dumps(results, indent=2))
        ```
        """)
        
        # Sample API request builder
        st.subheader("API Request Builder")
        
        api_description = st.text_area(
            "Job Description for API Request",
            height=100,
            placeholder="Enter a job description to test the API response..."
        )
        
        api_top_k = st.slider("Number of Results", min_value=1, max_value=20, value=3, key="api_top_k")
        api_enhance = st.checkbox("Use Enhanced Analysis", value=False, key="api_enhance")
        
        if st.button("Generate Sample API Response", type="primary"):
            if not api_description:
                st.error("Please enter a job description.")
            else:
                with st.spinner("Generating API response..."):
                    try:
                        # Generate sample API response
                        response = st.session_state.recommendation_system.generate_api_response(
                            api_description,
                            top_k=api_top_k,
                            enhance=api_enhance
                        )
                        
                        # Display response
                        st.json(response)
                        
                        # Option to download as JSON
                        json_str = json.dumps(response, indent=2)
                        st.download_button(
                            label="Download JSON Response",
                            data=json_str,
                            file_name="api_response.json",
                            mime="application/json"
                        )
                    except Exception as e:
                        st.error(f"Error generating API response: {str(e)}")
                        logger.error(f"Error in tab4: {str(e)}")

# Add an API endpoint using FastAPI integration (if installed)
# Note: This requires additional setup with FastAPI and Uvicorn
# This is a simplified example that would need to be expanded
def setup_api():
    try:
        from fastapi import FastAPI, Request, HTTPException
        from pydantic import BaseModel
        import uvicorn
        from starlette.middleware.cors import CORSMiddleware
        
        # Define API models
        class RecommendationRequest(BaseModel):
            job_description: str
            top_k: int = 5
            min_duration: int = 0
            max_duration: int = 180
            enhance: bool = False
        
        # Initialize API
        api = FastAPI(title="SHL Assessment Recommendation API")
        
        # Add CORS middleware
        api.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Create recommendation system instance for API
        recommendation_system = SHLRecommendationSystem()
        
        # Define API endpoints
        @api.post("/api/recommend")
        async def recommend(request: RecommendationRequest):
            try:
                response = recommendation_system.generate_api_response(
                    request.job_description,
                    top_k=request.top_k,
                    min_duration=request.min_duration,
                    max_duration=request.max_duration,
                    enhance=request.enhance
                )
                return response
            except Exception as e:
                logger.error(f"API error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Run the API server
        uvicorn.run(api, host="0.0.0.0", port=8000)
        
    except ImportError:
        logger.warning("FastAPI or Uvicorn not installed. API functionality disabled.")

# Run the app
if __name__ == "__main__":
    main()
