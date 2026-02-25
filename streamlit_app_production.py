#!/usr/bin/env python3
"""
Semantic Proximity Analyzer - Streamlit Production App
Full-featured semantic analysis with embeddings and modern visualizations
Deploy on: https://streamlit.io/cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import textwrap
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# ============= PAGE CONFIG =============
st.set_page_config(
    page_title="Semantic Proximity Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============= CUSTOM STYLING =============
st.markdown("""
<style>
    body { background-color: #ffffff; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
    h1 { font-weight: 700; color: #0f172a; font-size: 2.2rem; margin-bottom: 0.3rem; }
    h2 { font-weight: 600; color: #0f172a; font-size: 1.6rem; margin-top: 2rem; margin-bottom: 0.8rem; }
    h3 { font-weight: 600; color: #111827; font-size: 1.1rem; }
    .stButton>button { background-color: #1f2937; color: white; border: none; padding: 10px 20px; 
                       font-weight: 600; border-radius: 8px; transition: all 0.2s; }
    .stButton>button:hover { background-color: #111827; }
    .metric-card { background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); padding: 20px; 
                   border-radius: 10px; border-left: 4px solid #0284c7; }
    .info-box { background: #f8f8f8; padding: 16px; border-radius: 8px; border-left: 4px solid #1f2937; }
</style>
""", unsafe_allow_html=True)

# ============= LOAD MODEL (CACHED) =============
@st.cache_resource
def load_embedding_model():
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        return None

# ============= UTILITY FUNCTIONS =============
def load_keywords_excel(file_path_or_df):
    """Extract keywords from Excel/CSV file (handles multi-sheet structures)"""
    keywords = []
    try:
        # Handle file objects from st.file_uploader
        if hasattr(file_path_or_df, 'name'):
            xls = pd.ExcelFile(file_path_or_df)
            sheets = xls.sheet_names
            
            for sheet_name in sheets:
                # Read with skiprows to skip headers - "Mot-clé" header is at row 3
                df = pd.read_excel(file_path_or_df, sheet_name=sheet_name, skiprows=3)
                
                # Find the keyword column (should contain "Mot" or similar)
                keyword_col = None
                for col in df.columns:
                    if isinstance(col, str) and ('mot' in col.lower() or 'word' in col.lower()):
                        keyword_col = col
                        break
                
                # If not found, try second column (usually index 1)
                if not keyword_col and len(df.columns) > 1:
                    keyword_col = df.columns[1]
                
                if keyword_col:
                    # Extract keywords, skip NaNs and metadata
                    sheet_keywords = df[keyword_col].dropna().astype(str).tolist()
                    keywords.extend([k.strip() for k in sheet_keywords 
                                   if k.strip() and len(k.strip()) > 2 and not k.strip().isdigit()])
    except Exception as e:
        st.warning(f"Could not parse Excel file: {e}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            unique_keywords.append(k)
    
    return unique_keywords

def parse_french_number(val):
    """Convert French formatted numbers to float (e.g., '1 234,5' → 1234.5)"""
    if pd.isna(val) or val == '':
        return 0.0
    val = str(val).strip()
    # Remove spaces (French thousands separator)
    val = val.replace(' ', '')
    # Replace comma with dot (French decimal)
    val = val.replace(',', '.')
    try:
        return float(val)
    except:
        return 0.0

def get_best_url_for_keyword(gsc_df, keyword):
    """Find best ranking URL for keyword in GSC data (with fuzzy matching)"""
    from difflib import SequenceMatcher
    
    # Clean up data first
    gsc_clean = gsc_df.copy()
    for col in ['Clicks', 'Impressions', 'Position']:
        if col in gsc_clean.columns:
            gsc_clean[col] = gsc_clean[col].apply(parse_french_number)
    
    # Find query column
    query_col = next((col for col in ['Query', 'Queries', 'Top queries', 'query'] 
                     if col in gsc_clean.columns), None)
    
    if not query_col:
        return None
    
    # Find matching queries using fuzzy matching
    keyword_lower = keyword.lower().strip()
    matching_rows = []
    
    for idx, row in gsc_clean.iterrows():
        query_lower = str(row[query_col]).lower().strip()
        
        # Check if keyword is contained in query (flexible matching)
        # Or if query contains most words from keyword
        keyword_words = keyword_lower.split()
        query_words = query_lower.split()
        
        # Count how many keyword words appear in query
        word_match = sum(1 for word in keyword_words if word in query_lower)
        word_match_ratio = word_match / len(keyword_words) if keyword_words else 0
        
        # Include if at least 50% of keyword words match or keyword is in query
        if word_match_ratio >= 0.5 or keyword_lower in query_lower:
            matching_rows.append(idx)
    
    if not matching_rows:
        return None
    
    matching = gsc_clean.loc[matching_rows].copy()
    
    # Calculate performance score
    matching['Score'] = (
        matching.get('Clicks', 0) * 2 +
        matching.get('Impressions', 0) * 0.5 -
        matching.get('Position', 0) * 0.1
    )
    
    best_row = matching.loc[matching['Score'].idxmax()]
    
    # Find page column
    page_col = next((col for col in ['Landing page', 'Page', 'Landing Page', 'URL', 'Top pages', 'page'] 
                    if col in gsc_clean.columns), None)
    
    if not page_col:
        return None
    
    url = str(best_row[page_col])
    return {
        'url': url,
        'clicks': int(best_row.get('Clicks', 0)),
        'impressions': int(best_row.get('Impressions', 0)),
        'position': float(best_row.get('Position', 0)),
    }

def fetch_page_content(url):
    """Fetch page title, meta, and content"""
    try:
        if not url.startswith('http'):
            url = 'https://' + url
        
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
        response = requests.get(url, headers=headers, timeout=8, allow_redirects=True)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        title = soup.title.string if soup.title else ""
        meta_desc = ""
        for meta in soup.find_all('meta'):
            if meta.get('name') == 'description':
                meta_desc = meta.get('content', '')
                break
        
        body = soup.find('body')
        text = ' '.join([p.get_text() for p in body.find_all('p')])[:500] if body else ""
        
        parsed = urlparse(url)
        url_text = (parsed.netloc + parsed.path)[:100]
        
        return {
            'title': title or "",
            'meta_description': meta_desc,
            'content': text,
            'url_text': url_text,
            'full_url': url,
        }
    except:
        return None

def calculate_semantic_proximity(keyword, page_data, model):
    """Calculate semantic score using embeddings"""
    if not page_data or not model:
        return 0
    
    try:
        keyword_embedding = model.encode([keyword])[0]
        
        texts_to_compare = [
            page_data.get('title', ''),
            page_data.get('meta_description', ''),
            page_data.get('content', '')[:200],
            page_data.get('url_text', ''),
        ]
        
        page_embeddings = model.encode(texts_to_compare)
        
        similarities = []
        for emb in page_embeddings:
            sim = cosine_similarity([keyword_embedding], [emb])[0][0]
            similarities.append(max(0, sim))
        
        weights = [0.35, 0.25, 0.30, 0.10]
        score = sum(s * w for s, w in zip(similarities, weights)) * 100
        
        return round(score, 2)
    except:
        return 0

def generate_charts(result_df):
    """Generate modern charts without text overlap"""
    charts = {}
    
    if result_df.empty:
        return charts
    
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'axes.edgecolor': '#d1d5db',
        'grid.alpha': 0.3,
    })
    
    # 1) Quality Snapshot
    excellent = (result_df['Proximity_Score'] >= 80).sum()
    good = ((result_df['Proximity_Score'] >= 60) & (result_df['Proximity_Score'] < 80)).sum()
    fair = ((result_df['Proximity_Score'] >= 40) & (result_df['Proximity_Score'] < 60)).sum()
    poor = (result_df['Proximity_Score'] < 40).sum()
    total = len(result_df)
    
    quality_data = {
        'Excellent (80+)': excellent,
        'Good (60-79)': good,
        'Fair (40-59)': fair,
        'Low (<40)': poor,
    }
    
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    colors = ['#10b981', '#84cc16', '#f59e0b', '#ef4444']
    bars = ax.barh(list(quality_data.keys()), list(quality_data.values()), color=colors)
    ax.set_title('Keyword Quality Distribution')
    ax.set_xlabel('Count')
    ax.grid(axis='x')
    ax.invert_yaxis()
    for i, bar in enumerate(bars):
        width = bar.get_width()
        pct = (width / total * 100) if total > 0 else 0
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, f"{int(width)} ({pct:.0f}%)", 
               va='center', fontsize=9)
    charts['quality'] = fig
    
    # 2) Score Distribution
    fig, ax = plt.subplots(figsize=(10.5, 5), constrained_layout=True)
    ax.hist(result_df['Proximity_Score'], bins=20, color='#1d4ed8', edgecolor='white', alpha=0.9)
    ax.set_title('Score Distribution')
    ax.set_xlabel('Semantic Proximity Score')
    ax.set_ylabel('Keyword Count')
    ax.grid(axis='y')
    charts['distribution'] = fig
    
    # 3) Clicks vs Score (scatter)
    if 'Clicks' in result_df.columns:
        fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
        scatter = ax.scatter(result_df['Proximity_Score'], result_df['Clicks'],
                           s=80, c=result_df['Proximity_Score'], cmap='viridis', alpha=0.7, edgecolors='white')
        ax.set_title('Clicks vs Semantic Score')
        ax.set_xlabel('Semantic Proximity Score')
        ax.set_ylabel('Clicks')
        ax.grid()
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Score')
        charts['clicks'] = fig
    
    return charts

# ============= MAIN APP =============
def main():
    st.title("📊 Semantic Proximity Analyzer")
    st.markdown("*Analyze keyword-to-page semantic alignment using AI embeddings*")
    
    # Overview
    with st.expander("ℹ️ How it works", expanded=False):
        st.markdown("""
        ### What This Tool Does
        
        1. **Takes your priority keywords** (the ones you want to rank for)
        2. **Finds the best ranking URL** for each keyword in your GSC data
        3. **Fetches page content** (title, meta, body text)
        4. **Calculates semantic proximity** using AI embeddings (0-100 score)
        5. **Generates insights** with modern visualizations and exportable reports
        
        ### Required Inputs
        
        - **GSC Data (CSV)**: Export from Google Search Console with Query, Page, Clicks, Impressions, Position
        - **Priority Keywords (Excel/CSV)**: Your strategic keywords list with column named "Mot-clé" or "Keyword"
        
        ### Output
        
        - Semantic proximity scores (0-100) for each keyword
        - Quality distribution charts
        - Top/Bottom performing keywords
        - CSV & Excel export with full data
        """)
    
    st.markdown("---")
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'charts' not in st.session_state:
        st.session_state.charts = {}
    
    # Instructions
    st.info("📋 **Required Files:** Upload both your GSC export CSV and your strategic priority keywords file (Excel or CSV)")
    
    # File uploads
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1️⃣ GSC Data (CSV)")
        st.caption("Export from Google Search Console with Query, Page, Clicks, Impressions, Position columns")
        gsc_file = st.file_uploader("Upload GSC CSV", type=['csv'], key='gsc', help="Export your GSC data as CSV with queries and landing pages")
    
    with col2:
        st.subheader("2️⃣ Priority Keywords (Excel/CSV) ⭐")
        st.caption("Your strategic keywords list - the app will find best URLs for each and analyze semantic alignment")
        keywords_file = st.file_uploader("Upload Keywords File", type=['xlsx', 'xls', 'csv'], key='keywords', help="Excel or CSV file with your priority keywords in 'Mot-clé' or 'Keyword' column")
    
    st.markdown("---")
    
    # Analyze button
    if st.button("🚀 Analyze Semantic Proximity", type="primary", use_container_width=True):
        if gsc_file and keywords_file:
            with st.spinner("🔄 Analyzing... (this may take a minute)"):
                try:
                    # Load files
                    gsc_df = pd.read_csv(gsc_file, dtype={'Clicks': str, 'Impressions': str, 'Position': str})
                    
                    # Load keywords from file (handles Excel and CSV)
                    keywords = load_keywords_excel(keywords_file)
                    
                    if not keywords:
                        st.error("❌ Could not extract keywords from file")
                        st.warning("Make sure your file has a column named 'Mot-clé', 'Keyword', or 'Keywords'")
                        return
                    
                    st.success(f"✅ Loaded {len(keywords)} priority keywords from file")
                    with st.expander("📋 View loaded keywords"):
                        st.write(", ".join(keywords[:20]))
                        if len(keywords) > 20:
                            st.write(f"... and {len(keywords) - 20} more")
                    
                    st.info(f"🔍 Now matching {len(keywords)} keywords with GSC data and analyzing semantic proximity...")
                    
                    # Load model
                    model = load_embedding_model()
                    if not model:
                        st.error("❌ Failed to load embedding model")
                        return
                    
                    # Analyze
                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, keyword in enumerate(keywords):
                        status_text.text(f"Analyzing: {keyword} ({i+1}/{len(keywords)})")
                        
                        url_info = get_best_url_for_keyword(gsc_df, keyword)
                        if not url_info:
                            continue
                        
                        page_data = fetch_page_content(url_info['url'])
                        if not page_data:
                            continue
                        
                        proximity = calculate_semantic_proximity(keyword, page_data, model)
                        
                        results.append({
                            'Keyword': keyword,
                            'URL': url_info['url'],
                            'Proximity_Score': proximity,
                            'Clicks': int(url_info['clicks']),
                            'Impressions': int(url_info['impressions']),
                            'Position': round(url_info['position'], 1),
                        })
                        
                        progress_bar.progress((i + 1) / len(keywords))
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    if results:
                        result_df = pd.DataFrame(results)
                        st.session_state.results = result_df
                        st.session_state.charts = generate_charts(result_df)
                        st.success(f"✅ Analyzed {len(results)} keywords!")
                    else:
                        st.error("❌ No matching keywords found in GSC data")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    with st.expander("Show detailed error"):
                        st.code(traceback.format_exc())
        else:
            st.error("⚠️ **Both files are required to run the analysis:**")
            if not gsc_file:
                st.write("- ❌ GSC Data CSV is missing")
            if not keywords_file:
                st.write("- ❌ **Priority Keywords file is missing** (Excel or CSV with your strategic keywords)")
            
            with st.expander("💡 What are Priority Keywords?"):
                st.markdown("""
                **Priority Keywords** are your strategic target keywords - the ones you want to rank for.
                
                **File Format:**
                - Excel (.xlsx) or CSV file
                - Should contain a column named: `Mot-clé`, `Keyword`, or `Keywords`
                - One keyword per row
                
                **Example:**
                ```
                Mot-clé
                interim
                offre emploi
                agence interim
                recherche emploi
                ```
                
                **The app will:**
                1. Find the best ranking URL in GSC for each keyword
                2. Fetch the page content
                3. Calculate semantic proximity score (0-100)
                4. Generate insights and recommendations
                """)
    
    st.markdown("---")
    
    # Display results
    if st.session_state.results is not None:
        result_df = st.session_state.results
        
        st.subheader("📈 Analysis Results")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Score", f"{result_df['Proximity_Score'].mean():.1f}")
        with col2:
            st.metric("Keywords Analyzed", len(result_df))
        with col3:
            st.metric("Avg Clicks", f"{result_df['Clicks'].mean():.1f}")
        with col4:
            st.metric("Avg Impressions", f"{result_df['Impressions'].mean():.1f}")
        
        st.markdown("---")
        
        # Charts
        st.subheader("📊 Visualizations")
        
        col1, col2 = st.columns(2)
        with col1:
            if 'quality' in st.session_state.charts:
                st.pyplot(st.session_state.charts['quality'], use_container_width=True)
        with col2:
            if 'distribution' in st.session_state.charts:
                st.pyplot(st.session_state.charts['distribution'], use_container_width=True)
        
        if 'clicks' in st.session_state.charts:
            st.pyplot(st.session_state.charts['clicks'], use_container_width=True)
        
        st.markdown("---")
        
        # Tables
        st.subheader("🎯 Top & Bottom Keywords")
        
        tab1, tab2 = st.tabs(["Top 20 (Best Aligned)", "Bottom 20 (Needs Work)"])
        
        with tab1:
            top_20 = result_df.nlargest(20, 'Proximity_Score')
            st.dataframe(top_20, use_container_width=True, hide_index=True)
        
        with tab2:
            bottom_20 = result_df.nsmallest(20, 'Proximity_Score')
            st.dataframe(bottom_20, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Download
        st.subheader("⬇️ Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV",
                data=csv,
                file_name=f"semantic_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Excel export
            buffer = pd.ExcelWriter('temp.xlsx', engine='openpyxl')
            result_df.to_excel(buffer, sheet_name='Results', index=False)
            
            summary = pd.DataFrame({
                'Metric': ['Total Keywords', 'Avg Score', 'Excellent (80+)', 'Good (60-80)', 'Fair (40-60)', 'Poor (<40)'],
                'Value': [
                    len(result_df),
                    f"{result_df['Proximity_Score'].mean():.1f}",
                    (result_df['Proximity_Score'] >= 80).sum(),
                    ((result_df['Proximity_Score'] >= 60) & (result_df['Proximity_Score'] < 80)).sum(),
                    ((result_df['Proximity_Score'] >= 40) & (result_df['Proximity_Score'] < 60)).sum(),
                    (result_df['Proximity_Score'] < 40).sum(),
                ]
            })
            summary.to_excel(buffer, sheet_name='Summary', index=False)
            buffer.close()
            
            with open('temp.xlsx', 'rb') as f:
                st.download_button(
                    label="📊 Download Excel",
                    data=f.read(),
                    file_name=f"semantic_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.ms-excel",
                    use_container_width=True
                )

if __name__ == '__main__':
    main()
