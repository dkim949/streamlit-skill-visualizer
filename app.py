import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json

st.set_page_config(layout="wide", page_title="Dongin Kim - Data Science Portfolio")

# Custom CSS to improve the design
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E90FF;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #4682B4;
    }
    .contact-info {
        text-align: center;
        color: #708090;
        font-size: 0.9rem;
        margin-top: 2rem;
    }
    .stExpander {
        border: 1px solid #E0E0E0;
        border-radius: 5px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# 타이틀에 이름 추가
st.markdown(
    "<h1 class='main-header'>Dongin Kim - Data Science Skill Portfolio</h1>",
    unsafe_allow_html=True,
)


# Load skills from JSON file
def load_skills():
    try:
        with open("./assets/skills.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(
            "skills.json file not found. Please create this file with your skills data."
        )
        return {}


# 스킬 데이터 로드
skills = load_skills()

# 상단 두 열로 나누기
top_left, top_right = st.columns(2)

with top_left:
    st.markdown(
        "<h2 class='sub-header'>Skills Proficiency</h2>", unsafe_allow_html=True
    )
    # 스킬을 레벨에 따라 내림차순으로 정렬
    sorted_skills = sorted(skills.items(), key=lambda x: x[1]["level"], reverse=True)

    fig = go.Figure()
    for skill, data in sorted_skills:
        fig.add_trace(
            go.Bar(
                y=[skill],
                x=[data["level"]],
                orientation="h",
                text=f"{data['level']}/10",
                textposition="auto",
                showlegend=False,
            )
        )
    fig.update_layout(
        barmode="stack",
        height=400,
        xaxis_range=[0, 10],
        yaxis={
            "categoryorder": "array",
            "categoryarray": [s[0] for s in sorted_skills[::-1]],
        },
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

with top_right:
    st.markdown("<h2 class='sub-header'>Skill Details</h2>", unsafe_allow_html=True)
    for skill, data in skills.items():
        with st.expander(f"{skill} - {data['level']}/10"):
            st.write(f"Tools: {', '.join(data['tools'])}")

# 하단 두 열로 나누기
bottom_left, bottom_right = st.columns(2)

with bottom_left:
    st.markdown("<h2 class='sub-header'>Key Projects</h2>", unsafe_allow_html=True)
    projects = [
        {
            "name": "Delivery Time Estimation Model",
            "description": "Developed a machine learning model designed to predict delivery times in a high-volume, time-sensitive environment such as food delivery services. The goal was to improve operational efficiency and enhance customer satisfaction by providing more accurate delivery time estimates.",
            "skills": [
                "Python",
                "scikit-learn",
                "TensorFlow",
                "PyTorch",
                "Gradient Boosting",
                "Random Forest",
                "Custom Loss Function",
                "Real-Time Data Handling",
                "Feature Engineering",
                "Model Optimization",
                "Inductive Bias via Multi-Task Learning",
                "Exploratory Data Analysis",
                "Stakeholder Engagement",
            ],
        },
        {
            "name": "Anomaly Detection in Transactional Data",
            "description": "Developed an integrated system to detect anomalies in transactional data for a food delivery service.",
            "skills": ["Python", "Statistical Forecasting", "Data Visualization"],
        },
        {
            "name": "Customer Churn Prediction",
            "description": "Developed a machine learning model to predict customer churn with 85% accuracy.",
            "skills": ["Python", "scikit-learn", "Tableau"],
        },
        {
            "name": "Comic Book Recommendation System",
            "description": "Developed a recommendation system for a newly launched mobile webtoon service.",
            "skills": ["Python", "LightFM", "Collaborative Filtering"],
        },
        {
            "name": "Customer Lifetime Value Analysis & Segmentation",
            "description": "This project supports marketing decision-making through in-depth customer lifetime value analysis and segmentation. The goal was to identify key customer segments and optimize marketing strategies accordingly.",
            "skills": [
                "Python",
                "R",
                "Scikit-learn",
                "Clustering",
                "Classification",
                "Tableau",
            ],
        },
        {
            "name": "Inverse Cooking: Recipe Generation from Food Images",
            "description": "Implemented a deep learning model that generates cooking recipes from food images. This project, based on the CVPR 2019 paper, uses computer vision and natural language processing techniques to predict ingredients and generate cooking instructions from images.",
            "skills": [
                "Python",
                "PyTorch",
                "Computer Vision",
                "NLP",
                "Deep Learning",
                "CLIP",
            ],
        },
        {
            "name": "AutoML Interactive Dashboard",
            "description": "Developed an interactive Streamlit dashboard that automates the entire data analysis and machine learning model application process. Users can upload data, perform exploratory data analysis, train multiple models using AutoML techniques, and visualize results in real-time.",
            "skills": [
                "Python",
                "Streamlit",
                "AutoML",
                "Pandas",
                "Scikit-learn",
                "Plotly",
            ],
        },
    ]

    for project in projects:
        with st.expander(project["name"]):
            st.write(project["description"])
            st.write(f"Skills used: {', '.join(project['skills'])}")

with bottom_right:
    st.markdown(
        "<h2 class='sub-header'>Currently Learning</h2>", unsafe_allow_html=True
    )
    st.write("- Advanced Natural Language Processing techniques")
    st.write("- Deep Reinforcement Learning")
    st.write("- MLOps practices and tools")

# 연락처 정보 추가
st.markdown(
    """
<div class='contact-info'>
    For more information, please contact: 
    <a href='mailto:donginkim88@gmail.com'>donginkim88@gmail.com</a> or 
    visit <a href='https://linkedin.com/in/donginkim88' target='_blank'>LinkedIn Profile</a>
</div>
""",
    unsafe_allow_html=True,
)
