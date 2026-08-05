# Netflix Advanced Recommendation System

This project builds an end-to-end Netflix content analysis and recommendation system using Python and machine learning. It combines data cleaning, feature engineering, exploratory data analysis (EDA), content-based recommendation, and clustering to uncover patterns in Netflix titles and generate relevant recommendations.

## Project Overview

The workflow is organized into four notebooks:

1. Data Cleaning and Preparation
   - Loads the raw Netflix dataset
   - Standardizes column names
   - Handles missing values and date parsing
   - Saves a cleaned dataset

2. Feature Engineering and EDA
   - Extracts useful numeric features such as duration, genre count, cast count, and title length
   - Analyzes content distribution by type, genre, country, and release year
   - Generates visualizations and saves plots

3. Recommendation System
   - Builds a content-based recommender using TF-IDF and cosine similarity
   - Combines title, director, cast, country, rating, and genre information into a single feature text
   - Returns the most similar titles for a given movie or show

4. Content Segmentation and Clustering
   - Applies K-Means clustering on engineered features
   - Reduces dimensionality with PCA for visualization
   - Segments Netflix content into meaningful groups

## Project Structure

- data/data/
  - netflix_titles.csv: original dataset
  - netflix_cleaned.csv: cleaned dataset
  - netflix_cleaned_featured.csv: enriched dataset with engineered features and cluster labels
  - netflix_featured.csv: feature-engineered dataset used for recommendation and clustering

- notebook/
  - 1_Data_Cleaning_and_Preparation.ipynb
  - 2_Feature_Engineering_and_EDA.ipynb
  - 3_Recommendation_System.ipynb
  - 4_Content_Segmentation_and_Clustering.ipynb

- plots/
  - Visualizations generated during EDA and analysis

- requirements.txt
  - Python dependencies for the project

## Technologies Used

- Python
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- scikit-learn

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Open the notebooks in the following order:

1. notebook/1_Data_Cleaning_and_Preparation.ipynb
2. notebook/2_Feature_Engineering_and_EDA.ipynb
3. notebook/3_Recommendation_System.ipynb
4. notebook/4_Content_Segmentation_and_Clustering.ipynb

## Recommendation Example

In the recommendation notebook, you can generate similar titles by calling the recommender function with a known title. For example:

```python
recommend_titles("The Crown", top_n=5)
```

## Key Outputs

- Cleaned Netflix dataset
- Feature-engineered dataset for modeling
- Content-based recommendation engine
- Clustering and segmentation results
- EDA plots for trends and distributions

## Future Improvements

Possible next steps for this project include:

- Adding a collaborative filtering recommender
- Building a hybrid recommender system
- Deploying the model as a web app or API
- Evaluating recommendation quality with precision@k and recall@k

## Notes

Some notebooks reference local file paths. If you run them in a different environment, make sure the dataset paths are updated accordingly.
