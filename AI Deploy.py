import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Screening System", page_icon="📄")

st.title("📄 AI Resume Screening System")
st.write("Upload resumes dataset and enter job description to get top matching candidates.")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s+#./]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

uploaded_file = st.file_uploader("Upload Resume CSV File", type=["csv"])

job_description = st.text_area(
    "Enter Job Description",
    height=180,
    value="""We are hiring a Machine Learning Engineer.
Required skills: Python, SQL, Machine Learning, NLP, Scikit-learn,
Data Analysis, Feature Engineering, Docker, Cloud Computing."""
)

top_n = st.slider("Select Top Candidates", 5, 50, 10)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.fillna("")

    required_cols = [
        "Full_name", "Phone_number", "Email", "Candidate_position",
        "Education", "Major", "Soft_skill", "Professional_skill",
        "Language_skill", "Experience", "Certificate",
        "Achievement", "Project", "Activity"
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
    else:
        df["resume_text"] = (
            df["Candidate_position"].astype(str) + " " +
            df["Education"].astype(str) + " " +
            df["Major"].astype(str) + " " +
            df["Soft_skill"].astype(str) + " " +
            df["Professional_skill"].astype(str) + " " +
            df["Language_skill"].astype(str) + " " +
            df["Experience"].astype(str) + " " +
            df["Certificate"].astype(str) + " " +
            df["Achievement"].astype(str) + " " +
            df["Project"].astype(str) + " " +
            df["Activity"].astype(str)
        )

        df["clean_resume"] = df["resume_text"].apply(clean_text)
        clean_jd = clean_text(job_description)

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2)
        )

        tfidf_matrix = vectorizer.fit_transform(df["clean_resume"])
        jd_vector = vectorizer.transform([clean_jd])

        scores = cosine_similarity(jd_vector, tfidf_matrix).flatten()

        df["Match Score (%)"] = (scores * 100).round(2)

        result = df.sort_values(
            by="Match Score (%)",
            ascending=False
        )

        final_result = result[
            [
                "Full_name",
                "Phone_number",
                "Email",
                "Candidate_position",
                "Education",
                "Major",
                "Professional_skill",
                "Experience",
                "Project",
                "Match Score (%)"
            ]
        ].head(top_n)

        st.subheader("🏆 Top Matching Candidates")
        st.dataframe(final_result, use_container_width=True)

        csv = final_result.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Shortlisted Candidates CSV",
            data=csv,
            file_name="shortlisted_candidates.csv",
            mime="text/csv"
        )
else:
    st.info("Please upload your resumes CSV file.")