import pandas as pd

# Step 1: Load all data (assuming tab-separated files)
skills_df = pd.read_csv('data/Skills.txt', sep='\t')
occupation_df = pd.read_csv('data/Occupation Data.txt', sep='\t')
interests_df = pd.read_csv('data/Interests.txt', sep='\t')
education_df = pd.read_csv('data/Education, Training, and Experience.txt', sep='\t')
workstyles_df = pd.read_csv('data/Work Styles.txt', sep='\t')

# Step 2: Check duplicates and aggregate duplicates in skills_df

print("Checking duplicates in Skills data...")
duplicates_skills = skills_df.duplicated(subset=['O*NET-SOC Code', 'Element Name'], keep=False)
print(f"Number of duplicate rows in skills_df: {duplicates_skills.sum()}")

# Convert 'Data Value' to numeric (coerce errors to NaN)
skills_df['Data Value'] = pd.to_numeric(skills_df['Data Value'], errors='coerce')

# Aggregate duplicates by mean of 'Data Value'
skills_agg = skills_df.groupby(['O*NET-SOC Code', 'Element Name'], as_index=False)['Data Value'].mean()

# Step 3: Pivot skills data to wide format
skills_wide = skills_agg.pivot(index='O*NET-SOC Code', columns='Element Name', values='Data Value').reset_index()

# Repeat duplicate handling for interests_df
print("Checking duplicates in Interests data...")
duplicates_interests = interests_df.duplicated(subset=['O*NET-SOC Code', 'Element Name'], keep=False)
print(f"Number of duplicate rows in interests_df: {duplicates_interests.sum()}")
interests_df['Data Value'] = pd.to_numeric(interests_df['Data Value'], errors='coerce')
interests_agg = interests_df.groupby(['O*NET-SOC Code', 'Element Name'], as_index=False)['Data Value'].mean()
interests_wide = interests_agg.pivot(index='O*NET-SOC Code', columns='Element Name', values='Data Value').reset_index()

# Repeat duplicate handling for education_df
print("Checking duplicates in Education data...")
duplicates_education = education_df.duplicated(subset=['O*NET-SOC Code', 'Element Name'], keep=False)
print(f"Number of duplicate rows in education_df: {duplicates_education.sum()}")
education_df['Data Value'] = pd.to_numeric(education_df['Data Value'], errors='coerce')
education_agg = education_df.groupby(['O*NET-SOC Code', 'Element Name'], as_index=False)['Data Value'].mean()
education_wide = education_agg.pivot(index='O*NET-SOC Code', columns='Element Name', values='Data Value').reset_index()

# Repeat duplicate handling for workstyles_df
print("Checking duplicates in Work Styles data...")
duplicates_workstyles = workstyles_df.duplicated(subset=['O*NET-SOC Code', 'Element Name'], keep=False)
print(f"Number of duplicate rows in workstyles_df: {duplicates_workstyles.sum()}")
workstyles_df['Data Value'] = pd.to_numeric(workstyles_df['Data Value'], errors='coerce')
workstyles_agg = workstyles_df.groupby(['O*NET-SOC Code', 'Element Name'], as_index=False)['Data Value'].mean()
workstyles_wide = workstyles_agg.pivot(index='O*NET-SOC Code', columns='Element Name', values='Data Value').reset_index()

# Step 4: Select relevant columns from occupation data
occupation_small = occupation_df[['O*NET-SOC Code', 'Title', 'Description']]

# Step 5: Merge all dataframes on 'O*NET-SOC Code'

df = occupation_small.merge(skills_wide, on='O*NET-SOC Code', how='inner')
df = df.merge(interests_wide, on='O*NET-SOC Code', how='inner', suffixes=('', '_interests'))
df = df.merge(education_wide, on='O*NET-SOC Code', how='inner', suffixes=('', '_education'))
df = df.merge(workstyles_wide, on='O*NET-SOC Code', how='inner', suffixes=('', '_workstyles'))

# Step 6: Check final dataframe shape and missing values
print(df.head())
print(f"Final shape after merge: {df.shape}")
print("Missing values per column:")
print(df.isnull().sum())

# Step 7: Drop rows with missing values (optional)
df_clean = df.dropna()
print(f"Shape after dropping missing values: {df_clean.shape}")
df_final = df_clean  # final cleaned and merged dataset


# Step 8: (Optional) Encoding categorical columns if needed
# Replace 'CategoryColumnName' with your actual categorical column name if any
# If you have categorical columns to encode, use:
# df_encoded = pd.get_dummies(df_clean, columns=['CategoryColumnName'])
# print(df_encoded.head())

# If no encoding needed now, just keep df_clean for further processing
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Step 1: User input dictionary (you can modify this)
user_profile = {
    'Active Learning': 4.5,
    'Critical Thinking': 5.0,
    'Complex Problem Solving': 4.8,
    'Reading Comprehension': 4.7,
    'Writing': 4.2,
    'Social Orientation': 3.5,
    'Stress Tolerance': 4.0,
    'Independence': 4.3,
    'Initiative': 4.6,
    'Analytical Thinking': 5.0
}

# Step 2: Convert to DataFrame for easy comparison
user_df = pd.DataFrame([user_profile])

# Step 3: Filter only the selected features from the cleaned occupation data
feature_cols = list(user_profile.keys())
occupation_features = df_clean[feature_cols]

# Step 4: Calculate cosine similarity between user and all occupations
similarities = cosine_similarity(user_df, occupation_features)

# Step 5: Attach similarity scores to the occupation data
df_clean['Similarity'] = similarities[0]

# Step 6: Sort by similarity
top_matches = df_clean.sort_values(by='Similarity', ascending=False)

# Step 7: Display top 5 recommendations
print("Top 5 Career Recommendations:")
print(top_matches[['Title', 'Description', 'Similarity']].head(5))
# Assuming df_final is your cleaned merged DataFrame
# Get feature columns excluding ID, title, and description
feature_columns = df_final.columns.difference(['O*NET-SOC Code', 'Title', 'Description'])

print("Number of features to rate:", len(feature_columns))
print("Sample feature columns:", feature_columns[:10].tolist())
# Create an empty dictionary to store user ratings
user_profile = {}

print("Please rate yourself on the following features from 1 (lowest) to 5 (highest):")

for feature in feature_columns:
    while True:
        try:
            rating = float(input(f"{feature}: "))
            if 1 <= rating <= 5:
                user_profile[feature] = rating
                break
            else:
                print("Please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

# Convert user profile dictionary to DataFrame for easy comparison later
import pandas as pd
user_profile_df = pd.DataFrame(user_profile, index=[0])

print("\nUser Profile Ratings:")
print(user_profile_df)
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Extract career features matrix (only features columns)
career_features = df_final[feature_columns].values

# Convert user profile to numpy array
user_vector = user_profile_df[feature_columns].values

# Calculate cosine similarity between user and all careers
similarities = cosine_similarity(user_vector, career_features)

# similarities is a 2D array with shape (1, number_of_careers)
# Flatten it to 1D
similarities = similarities.flatten()

# Add similarity scores to df_final
df_final['SimilarityScore'] = similarities

# Sort careers by similarity descending
recommended_careers = df_final.sort_values(by='SimilarityScore', ascending=False)

# Show top 10 recommended careers
print(recommended_careers[['O*NET-SOC Code', 'Title', 'SimilarityScore']].head(10))

