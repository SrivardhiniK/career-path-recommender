import streamlit as st
import pandas as pd

st.title("Career Path Recommender")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your career dataset CSV", type=["csv"])

if uploaded_file is not None:
    df_final = pd.read_csv(uploaded_file)

    st.write("Dataset loaded. Preview:")
    st.dataframe(df_final.head())

    # Show columns in dataset
    all_columns = df_final.columns.tolist()
    st.write(f"Columns in dataset: {all_columns}")

    # Define your feature lists here (or read from file)
    # For demo, assume these are your feature lists (you can adjust)
    skills_feature_list = ['Achievement/Effort', 'Active Learning', 'Active Listening']
    interests_feature_list = ['Analytical Thinking', 'Artistic', 'Attention to Detail']
    education_feature_list = ['Education Level', 'Training Time']
    workstyles_feature_list = ['Dependability', 'Integrity', 'Stress Tolerance']

    # Filter feature lists to existing columns in df_final
    def filter_existing_columns(feature_list):
        return [col for col in feature_list if col in df_final.columns]

    skills_features = filter_existing_columns(skills_feature_list)
    interests_features = filter_existing_columns(interests_feature_list)
    education_features = filter_existing_columns(education_feature_list)
    workstyles_features = filter_existing_columns(workstyles_feature_list)

    st.write("Using these Skills features:", skills_features)
    st.write("Using these Interests features:", interests_features)
    st.write("Using these Education features:", education_features)
    st.write("Using these Workstyles features:", workstyles_features)

    # Input weights
    st.subheader("Set weights for each category (0 to 1)")
    weight_skills = st.slider("Weight for Skills", 0.0, 1.0, 0.25)
    weight_interests = st.slider("Weight for Interests", 0.0, 1.0, 0.25)
    weight_education = st.slider("Weight for Education", 0.0, 1.0, 0.25)
    weight_workstyles = st.slider("Weight for Workstyles", 0.0, 1.0, 0.25)

    if st.button("Calculate Recommendations"):
        # Calculate weighted scores
        df_final['skills_score'] = df_final[skills_features].mean(axis=1) * weight_skills
        df_final['interests_score'] = df_final[interests_features].mean(axis=1) * weight_interests
        df_final['education_score'] = df_final[education_features].mean(axis=1) * weight_education
        df_final['workstyles_score'] = df_final[workstyles_features].mean(axis=1) * weight_workstyles

        df_final['weighted_score'] = (
            df_final['skills_score'] +
            df_final['interests_score'] +
            df_final['education_score'] +
            df_final['workstyles_score']
        )

        top_recommendations = df_final.sort_values(by='weighted_score', ascending=False)

        st.subheader("Top 10 Career Recommendations")
        st.dataframe(top_recommendations[['O*NET-SOC Code', 'Title', 'weighted_score']].head(10))

else:
    st.info("Please upload the career dataset CSV file to proceed.")
