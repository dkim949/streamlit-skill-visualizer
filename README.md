# Data Science Skill Portfolio - Streamlit App

This Streamlit application showcases my data science skills, projects, and ongoing learning areas. It's designed to provide a comprehensive overview of my capabilities and experiences in the field of data science.

## Features

- Interactive visualization of skill proficiencies
- Detailed breakdown of skills and tools
- Showcase of key projects with descriptions
- Current learning focus areas

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/data-science-portfolio.git
   ```

2. Navigate to the project directory:
   ```
   cd data-science-portfolio
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `skills.json` file in the project root directory with your skills data. The format should be:
   ```json
   {
     "ML/DL": {
       "level": 9,
       "tools": ["scikit-learn", "TensorFlow", "Keras", "PyTorch", "PyTorch Lightning", "XGBoost", "LightGBM", "CatBoost"]
     },
     "Statistics": {
       "level": 8,
       "tools": ["scikit-learn", "statsmodels"]
     },
     "Time Series Analysis": {
       "level": 7,
       "tools": ["statsmodels", "pmdarima", "arch", "tensorflow-ts", "sktime", "prophet"]
     },
     ...
   }
   ```
   Note: This file is ignored by git for privacy reasons.

## Usage

Run the Streamlit app: