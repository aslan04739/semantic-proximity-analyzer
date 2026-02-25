#!/usr/bin/env python3
"""
Semantic Proximity Analyzer - Streamlit Dashboard
Clean, minimalistic interface for semantic analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

st.set_page_config(
    page_title="Semantic Proximity Analyzer",
    page_icon="⚪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for minimalistic white theme
st.markdown("""
<style>
    .main {
        background-color: #ffffff;
    }
    h1 {
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 600;
        color: #111111;
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    h2, h3 {
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 500;
        color: #111111;
    }
    p {
        color: #666666;
        line-height: 1.6;
    }
    .stButton>button {
        background-color: #111111;
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 500;
        border-radius: 6px;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #333333;
    }
    .uploadedFile {
        border: 1px solid #e7e7e7;
        border-radius: 8px;
        padding: 1rem;
    }
    .stDataFrame {
        border: 1px solid #e7e7e7;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


def analyze_semantic_proximity(df):
    """
    Analyze semantic proximity between queries and landing pages
    """
    if 'Query' not in df.columns or 'Landing page' not in df.columns:
        st.error("CSV must contain 'Query' and 'Landing page' columns")
        return None
    
    # Clean and prepare data
    df = df.copy()
    df['Query'] = df['Query'].fillna('').astype(str)
    df['Landing page'] = df['Landing page'].fillna('').astype(str)
    
    # Extract page titles from URLs (simplified)
    df['Page_Text'] = df['Landing page'].apply(
        lambda x: x.split('/')[-1].replace('-', ' ').replace('_', ' ')
    )
    
    # Combine query and page text for analysis
    queries = df['Query'].tolist()
    pages = df['Page_Text'].tolist()
    
    # Calculate TF-IDF and cosine similarity
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    # Vectorize
    all_texts = queries + pages
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Calculate similarity for each query-page pair
    similarities = []
    n = len(queries)
    for i in range(n):
        query_vec = tfidf_matrix[i]
        page_vec = tfidf_matrix[n + i]
        sim = cosine_similarity(query_vec, page_vec)[0][0]
        similarities.append(sim * 100)  # Convert to percentage
    
    df['Semantic_Score'] = similarities
    
    # Add performance metrics if available
    if 'Clicks' in df.columns:
        df['Clicks'] = pd.to_numeric(df['Clicks'], errors='coerce').fillna(0)
    if 'Impressions' in df.columns:
        df['Impressions'] = pd.to_numeric(df['Impressions'], errors='coerce').fillna(0)
    if 'Position' in df.columns:
        df['Position'] = pd.to_numeric(df['Position'], errors='coerce').fillna(0)
    
    return df


def main():
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    
    # Header
    st.markdown("# ⚪ Semantic Proximity Analyzer")
    st.markdown("Upload your Google Search Console data to analyze semantic alignment")
    
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose GSC CSV file",
        type=['csv'],
        help="Export your GSC data as CSV with Query, Landing page, Clicks, Impressions, and Position columns",
        key="file_uploader"
    )
    
    # Clear results when new file is uploaded
    if uploaded_file is not None and st.session_state.uploaded_data is None:
        st.session_state.results = None
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_data = df
            
            st.success(f"✓ Loaded {len(df):,} rows")
            
            # Show preview
            with st.expander("Preview data"):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Analyze button
            col1, col2 = st.columns([1, 4])
            with col1:
                analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)
            with col2:
                if st.session_state.results is not None:
                    if st.button("Clear Results", use_container_width=False):
                        st.session_state.results = None
                        st.rerun()
            
            # Run analysis if button clicked
            if analyze_clicked:
                with st.spinner("Analyzing..."):
                    result_df = analyze_semantic_proximity(df)
                    if result_df is not None:
                        st.session_state.results = result_df
            
            # Display results if available
            if st.session_state.results is not None:
                result_df = st.session_state.results
                
                st.success("✓ Analysis complete")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    avg_score = result_df['Semantic_Score'].mean()
                    st.metric("Avg Semantic Score", f"{avg_score:.1f}%")
                with col2:
                    high_score = (result_df['Semantic_Score'] > 70).sum()
                    st.metric("High Alignment", f"{high_score}")
                with col3:
                    low_score = (result_df['Semantic_Score'] < 30).sum()
                    st.metric("Low Alignment", f"{low_score}")
                with col4:
                    total_queries = len(result_df)
                    st.metric("Total Queries", f"{total_queries:,}")
                
                st.markdown("---")
                
                # Results tabs
                tab1, tab2, tab3 = st.tabs(["All Results", "Top Opportunities", "Low Alignment"])
                
                with tab1:
                    st.markdown("### All Results")
                    display_cols = ['Query', 'Landing page', 'Semantic_Score']
                    if 'Clicks' in result_df.columns:
                        display_cols.extend(['Clicks', 'Impressions', 'Position'])
                    
                    st.dataframe(
                        result_df[display_cols].sort_values('Semantic_Score', ascending=False),
                        use_container_width=True,
                        height=400
                    )
                
                with tab2:
                    st.markdown("### Top Opportunities")
                    st.markdown("Queries with high impressions but low semantic alignment")
                    
                    if 'Impressions' in result_df.columns:
                        opportunities = result_df[
                            (result_df['Semantic_Score'] < 50) &
                            (result_df['Impressions'] > result_df['Impressions'].median())
                        ].sort_values('Impressions', ascending=False).head(20)
                        
                        st.dataframe(
                            opportunities[display_cols],
                            use_container_width=True,
                            height=400
                        )
                    else:
                        st.info("Impressions data not available in CSV")
                
                with tab3:
                    st.markdown("### Low Alignment Issues")
                    st.markdown("Queries with semantic scores below 30%")
                    
                    low_alignment = result_df[
                        result_df['Semantic_Score'] < 30
                    ].sort_values('Semantic_Score').head(20)
                    
                    st.dataframe(
                        low_alignment[display_cols],
                        use_container_width=True,
                        height=400
                    )
                
                st.markdown("---")
                
                # Download button
                csv_buffer = io.BytesIO()
                result_df.to_csv(csv_buffer, index=False, encoding='utf-8')
                csv_buffer.seek(0)
                
                st.download_button(
                    label="Download Results (CSV)",
                    data=csv_buffer,
                    file_name="semantic_analysis_results.csv",
                    mime="text/csv",
                    use_container_width=False
                )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Make sure your CSV contains at least 'Query' and 'Landing page' columns")


if __name__ == "__main__":
    main()
