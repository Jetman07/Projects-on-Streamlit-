import streamlit as st
import pandas as pd
import joblib

# Set the System logo 
st.set_page_config(page_title="AI Impact on studnets", page_icon="AI_impact_logo.PNG", layout="centered")
# Load the pre-trained model using Streamlit's caching mechanism
@st.cache_resource
def load_model():
    try:
        # Replace 'your_model_file.pkl' with the actual filename of your saved model
        return joblib.load('AI_impact.pki')
    except Exception as e:
        print(f"Error Message:{e}")

model = load_model()

st.markdown("=" * 88)
st.image("Picture.PNG")
st.title("AI Student Impact Predictor", width= "content")
st.write("Adjust the features below to predict the student's Burnout Risk Level.")

st.markdown("=" * 88)
st.header("Categorical Features")
col1, col2 = st.columns(2)

with col1:
    major_category = st.selectbox("Major Category", ['Humanities', 'Medical', 'Business', 'STEM', 'Arts'])
    year_of_study = st.selectbox("Year of Study", ['Senior', 'Junior', 'Freshman', 'Sophomore', 'Graduate'])

with col2:
    primary_use_case = st.selectbox("Primary Use Case", ['Copywriting/Drafting', 'Ideation', 'Summarizing_Reading', 'Debugging/Troubleshooting', 'Direct_Answer_Generation'])

st.markdown("=" * 88)
st.header("Numerical Features")
col3, col4 = st.columns(2)

with col3:
    pre_semester_gpa = st.number_input("Pre-Semester GPA", min_value=1.183, max_value=3.998, value=3.000, step=0.1)
    weekly_genai_hours = st.number_input("Weekly GenAI Hours", min_value=0.0, max_value=40.0, value=10.0, step=1.0)
    tool_diversity = st.slider("Tool Diversity", min_value=1, max_value=5, value=2)
    traditional_study_hours = st.number_input("Traditional Study Hours", min_value=1.0, max_value=35.86, value=15.0, step=1.0)

with col4:
    perceived_ai_dependency = st.slider("Perceived AI Dependency (1-10)", min_value=1, max_value=10, value=5)
    anxiety_level_during_exams = st.slider("Anxiety Level During Exams (1-10)", min_value=1, max_value=10, value=5)
    post_semester_gpa = st.number_input("Post-Semester GPA", min_value=1.0, max_value=4.0, value=3.0, step=0.1)
    skill_retention_score = st.number_input("Skill Retention Score", min_value=10.78, max_value=100.0, value=75.0, step=1.0)
    
# 1. Create mapping dictionaries based on how you trained the model
major_mapping = {
    'Arts': 0, 
    'Business': 1, 
    'Humanities': 2, 
    'Medical': 3, 
    'STEM': 4
}

year_mapping = {
    'Freshman': 0, 
    'Sophomore': 1, 
    'Junior': 2, 
    'Senior': 3, 
    'Graduate': 4
}

use_case_mapping = {
    'Copywriting/Drafting': 0, 
    'Debugging/Troubleshooting': 1, 
    'Direct_Answer_Generation': 2, 
    'Ideation': 3, 
    'Summarizing_Reading': 4
}

# 2. Apply the mapping to the Streamlit dropdown inputs
encoded_major = major_mapping[major_category]
encoded_year = year_mapping[year_of_study]
encoded_use_case = use_case_mapping[primary_use_case]


if st.button("Predict Burnout Risk Level"):
    # 3. Use the encoded variables in your input data dictionary
    input_data = {
    'Major_Category': [encoded_major],
    'Year_of_Study': [encoded_year],
    'Pre_Semester_GPA': [pre_semester_gpa],
    'Weekly_GenAI_Hours': [weekly_genai_hours],
    'Primary_Use_Case': [encoded_use_case],
    'Tool_Diversity': [tool_diversity],
    'Traditional_Study_Hours': [traditional_study_hours],
    'Perceived_AI_Dependency': [perceived_ai_dependency],
    'Anxiety_Level_During_Exams': [anxiety_level_during_exams],
    'Post_Semester_GPA': [post_semester_gpa],
    'Skill_Retention_Score': [skill_retention_score]
    }

    # Convert dictionary to DataFrame and pass to model.predict()
    input_df = pd.DataFrame(input_data)
    
    try:
        raw_prediction = model.predict(input_df)[0]
        
        # Define the mapping from model output index to categorical variables
        # Note: Adjust the keys (0, 1, 2) if your specific LabelEncoder mapped them differently
        risk_mapping = {0: 'Low',1: 'Medium',2: 'High'}
        
        # Convert index to category (if the model outputs strings naturally, this handles it via .title())
        if isinstance(raw_prediction, (int, float)) or str(raw_prediction).isdigit():
            predicted_category = risk_mapping.get(int(raw_prediction), "Unknown")
        else:
            predicted_category = str(raw_prediction).title()
            
        st.markdown("---")
        
        # Display the tailored statement and solution based on the predicted category
        if predicted_category == 'Low':
            st.success(f"**Prediction: {predicted_category} Burnout Risk**")
            st.write("**Statement:** The student is currently maintaining a healthy academic workload and demonstrating safe, balanced interactions with GenAI tools.")
            st.write("**Solution:** Continue practicing current study habits. Maintain a healthy balance between traditional study methods and AI assistance, and schedule regular breaks to preserve this steady baseline.")
            
        elif predicted_category == 'Medium':
            st.warning(f"**Prediction: {predicted_category} Burnout Risk**")
            st.write("**Statement:** The student is exhibiting moderate stress indicators and may be developing a slight over-reliance on AI, leading to increased exam anxiety.")
            st.write("**Solution:** Implement strict timeboxing for AI usage (e.g., limit GenAI hours per week). Encourage active recall and traditional study sessions to rebuild skill retention and reduce dependency before it escalates.")
            
        elif predicted_category == 'High':
            st.snow()
            st.error(f"**Prediction: {predicted_category} Burnout Risk**")
            st.write("**Statement:** The student is experiencing severe academic strain, likely compounded by high AI dependency, low skill retention, and extreme exam anxiety.")
            st.write("**Solution:** Immediate intervention is recommended. Suggest academic counseling, significantly reduce AI-assisted study time in favor of foundational learning, and introduce dedicated mental health support and stress-management techniques.")
        
        else:
            st.info(f"**Prediction Result:** {raw_prediction}")
            
    except Exception as e:
        st.error(f"An error occurred during prediction:{e}. Please ensure your model was trained on these exact columns")
