#!/usr/bin/env python3
"""
Semantic Proximity Analyzer - Flask Dashboard
Analyzes keyword-to-page semantic alignment using embeddings and cosine similarity
Multi-language support: English, French, Arabic
"""

import os
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from datetime import datetime
import io
import base64
import textwrap
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from translations import get_text

def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path.cwd()))
    return Path(__file__).parent

BASE_DIR = get_base_dir()
TEMPLATE_DIR = BASE_DIR / "templates"

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

UPLOAD_FOLDER = Path(tempfile.gettempdir()) / "semantic_uploads"
RESULTS_FOLDER = Path.home() / "Semantic_Analysis_Results"
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
MAX_FILE_SIZE = 50 * 1024 * 1024

UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

analysis_jobs = {}

# Load embedding model
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except:
    embedding_model = None


def allowed_file(filename, extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions


def load_keywords(filepath):
    """Load keywords from Excel or CSV file"""
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath, header=None)
    
    # Handle structured Excel files with metadata/headers
    # Look for a row with "mot-clé" or "keyword" in column 1
    header_row = None
    for idx, row in df.iterrows():
        row_str = ' '.join([str(v).lower() for v in row if pd.notna(v)])
        if 'mot-clé' in row_str or 'keyword' in row_str or 'mot cle' in row_str:
            header_row = idx
            break
    
    # If we found a header row, get the keywords from the column after it
    if header_row is not None and header_row + 1 < len(df):
        # Get data rows after header
        data_df = df.iloc[header_row + 1:]
        # Try column 1 first (common for "Mot-clé" columns), then column 0
        if len(data_df.columns) > 1:
            keywords_col = data_df.iloc[:, 1]
        else:
            keywords_col = data_df.iloc[:, 0]
    else:
        # Fallback: get first column
        keywords_col = df.iloc[:, 0]
    
    keywords = keywords_col.tolist()
    # Filter out headers, empty values, and try to get actual keyword strings (not just numbers)
    result = []
    for k in keywords:
        k_str = str(k).strip()
        if k and k_str and k_str not in ['#', 'mot-clé', 'keyword', 'mot cle', 'name', 'keywords']:
            # Skip rows that look like just row numbers or metadata
            if not (k_str.isdigit() and len(k_str) < 3):
                result.append(k_str)
    
    return result


def get_best_url_for_keyword(gsc_df, keyword):
    """Find best ranking URL for a keyword from GSC data"""
    query_cols = ['Query', 'Queries', 'Top queries', 'query']
    
    # Find query column
    query_col = next((col for col in query_cols if col in gsc_df.columns), None)
    if not query_col:
        return None
    
    # Filter rows matching keyword
    matching = gsc_df[gsc_df[query_col].str.contains(keyword, case=False, na=False)]
    if matching.empty:
        return None
    
    # Score by clicks, impressions, CTR, position
    matching = matching.copy()
    
    # Normalize metrics for scoring
    if 'Clicks' in matching.columns:
        matching['Clicks'] = pd.to_numeric(matching['Clicks'], errors='coerce').fillna(0)
    if 'Impressions' in matching.columns:
        matching['Impressions'] = pd.to_numeric(matching['Impressions'], errors='coerce').fillna(0)
    if 'Position' in matching.columns:
        matching['Position'] = pd.to_numeric(matching['Position'], errors='coerce').fillna(0)
    
    # Calculate score: clicks + impressions - (position * 0.1)
    matching['Score'] = (
        matching.get('Clicks', 0) * 2 +
        matching.get('Impressions', 0) * 0.5 -
        matching.get('Position', 0) * 0.1
    )
    
    best_row = matching.loc[matching['Score'].idxmax()]
    
    page_cols = ['Landing page', 'Page', 'Landing Page', 'URL', 'Top pages', 'page']
    page_col = next((col for col in page_cols if col in gsc_df.columns), None)
    
    if not page_col:
        return None
    
    url = str(best_row[page_col])
    return {
        'url': url,
        'clicks': best_row.get('Clicks', 0),
        'impressions': best_row.get('Impressions', 0),
        'position': best_row.get('Position', 0),
    }


def extract_location_from_url(url):
    """Extract potential location/city info from URL"""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain_parts = parsed.netloc.split('.')
        
        # Check for city/country codes (e.g., london.example.com, example.co.uk)
        locations = []
        
        # Common city/region subdomains
        cities = ['london', 'newyork', 'paris', 'berlin', 'tokyo', 'sydney', 'toronto', 
                 'vancouver', 'moscow', 'mumbai', 'singapore', 'dubai', 'hongkong',
                 'sanfrancisco', 'seattle', 'austin', 'chicago', 'boston', 'miami']
        
        for part in domain_parts:
            if part.lower() in cities:
                locations.append(part.capitalize())
        
        # Path-based location detection (e.g., example.com/london)
        path_parts = parsed.path.strip('/').split('/')
        for part in path_parts[:2]:  # Check first 2 path segments
            if part.lower() in cities:
                locations.append(part.capitalize())
                break
        
        return locations[0] if locations else 'Multi-market'
    except:
        return 'Unknown'


def generate_geo_insights(result_df):
    """Generate geographic SEO insights"""
    insights = {
        'locations': {},
        'top_locations': [],
        'multi_market': True
    }
    
    # Group by location
    if len(result_df) > 0:
        result_df['Location'] = result_df['URL'].apply(extract_location_from_url)
        location_groups = result_df.groupby('Location').agg({
            'Proximity_Score': ['mean', 'count'],
            'Clicks': 'sum',
            'Impressions': 'sum',
            'Position': 'mean'
        }).round(2)
        
        for loc in location_groups.index:
            insights['locations'][loc] = {
                'keywords': int(location_groups.loc[loc, ('Proximity_Score', 'count')]),
                'avg_score': float(location_groups.loc[loc, ('Proximity_Score', 'mean')]),
                'total_clicks': int(location_groups.loc[loc, ('Clicks', 'sum')]),
                'total_impressions': int(location_groups.loc[loc, ('Impressions', 'sum')]),
                'avg_position': float(location_groups.loc[loc, ('Position', 'mean')])
            }
        
        # Top locations by performance
        top_locs = location_groups.sort_values(('Proximity_Score', 'mean'), ascending=False).head(5)
        insights['top_locations'] = [
            {'location': loc, 'score': float(top_locs.loc[loc, ('Proximity_Score', 'mean')])}
            for loc in top_locs.index
        ]
    
    return insights


def fetch_page_content(url):
    """Fetch page content, title, and metadata"""
    try:
        if not url.startswith('http'):
            url = 'https://' + url
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract content
        title = soup.title.string if soup.title else ""
        meta_desc = ""
        meta_title = ""
        
        for meta in soup.find_all('meta'):
            if meta.get('name') == 'description':
                meta_desc = meta.get('content', '')
            if meta.get('property') == 'og:title':
                meta_title = meta.get('content', '')
        
        # Get main content
        body = soup.find('body')
        text = ' '.join([p.get_text() for p in body.find_all('p')])[:500] if body else ""
        
        # URL domain/path
        parsed = urlparse(url)
        url_text = (parsed.netloc + parsed.path)[:100]
        
        return {
            'title': title or meta_title or "",
            'meta_description': meta_desc,
            'content': text,
            'url_text': url_text,
            'full_url': url,
        }
    except:
        return None


def calculate_semantic_proximity(keyword, page_data, model):
    """Calculate semantic proximity using embeddings"""
    if not page_data or not model:
        return 0
    
    try:
        # Get embedding for keyword
        keyword_embedding = model.encode([keyword])[0]
        
        # Get embeddings for page elements
        texts_to_compare = [
            page_data.get('title', ''),
            page_data.get('meta_description', ''),
            page_data.get('content', '')[:200],
            page_data.get('url_text', ''),
        ]
        
        page_embeddings = model.encode(texts_to_compare)
        
        # Calculate cosine similarities
        similarities = []
        for emb in page_embeddings:
            sim = cosine_similarity([keyword_embedding], [emb])[0][0]
            similarities.append(max(0, sim))
        
        # Weighted average: title > content > description > url
        weights = [0.35, 0.25, 0.30, 0.10]
        score = sum(s * w for s, w in zip(similarities, weights)) * 100
        
        return round(score, 2)
    except:
        return 0


def generate_graphs(result_df):
    """Generate visualization graphs - separated by audience"""
    graphs = {
        'client': [],      # Simple, impactful for decision makers
        'technical': []    # Detailed analysis for consultants
    }
    
    try:
        if result_df is None or result_df.empty:
            return graphs

        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 13,
            'axes.titleweight': 'bold',
            'axes.labelsize': 10,
            'axes.edgecolor': '#d1d5db',
            'axes.linewidth': 0.8,
            'grid.color': '#e5e7eb',
            'grid.linewidth': 0.7,
            'grid.alpha': 0.6,
            'xtick.color': '#374151',
            'ytick.color': '#374151',
        })

        chart_df = result_df.copy()
        chart_df['Keyword'] = chart_df['Keyword'].astype(str).str.strip()
        chart_df['Proximity_Score'] = pd.to_numeric(chart_df['Proximity_Score'], errors='coerce').fillna(0).clip(0, 100)
        for metric_col in ['Clicks', 'Impressions', 'Position']:
            if metric_col in chart_df.columns:
                chart_df[metric_col] = pd.to_numeric(chart_df[metric_col], errors='coerce').fillna(0)

        def wrap_label(text_value, width=24):
            text_value = str(text_value).strip()
            return '\n'.join(textwrap.wrap(text_value, width=width)) if text_value else '-'

        def fig_to_b64(fig):
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', dpi=140, bbox_inches='tight', facecolor='white')
            img_buffer.seek(0)
            return base64.b64encode(img_buffer.getvalue()).decode()

        def score_bucket(score):
            if score >= 80:
                return 'Excellent'
            if score >= 60:
                return 'Good'
            if score >= 40:
                return 'Fair'
            return 'Low'

        chart_df['Score_Bucket'] = chart_df['Proximity_Score'].apply(score_bucket)

        # ===== CLIENT-FRIENDLY GRAPHS (Executive Storytelling) =====
        excellent = (chart_df['Proximity_Score'] >= 80).sum()
        good = ((chart_df['Proximity_Score'] >= 60) & (chart_df['Proximity_Score'] < 80)).sum()
        fair = ((chart_df['Proximity_Score'] >= 40) & (chart_df['Proximity_Score'] < 60)).sum()
        poor = (chart_df['Proximity_Score'] < 40).sum()
        total_keywords = len(chart_df)

        # 1) Executive quality mix (clean bars, no overlap)
        quality_rows = [
            ('Excellent (80+)', excellent, '#10b981'),
            ('Good (60-79)', good, '#84cc16'),
            ('Fair (40-59)', fair, '#f59e0b'),
            ('Low (<40)', poor, '#ef4444'),
        ]
        quality_df = pd.DataFrame(quality_rows, columns=['Segment', 'Count', 'Color'])
        quality_df['Share'] = np.where(total_keywords > 0, (quality_df['Count'] / total_keywords) * 100, 0)

        fig, ax = plt.subplots(figsize=(10.8, 5.0), constrained_layout=True)
        bars = ax.barh(quality_df['Segment'], quality_df['Share'], color=quality_df['Color'], height=0.6)
        ax.set_title('Executive Quality Snapshot')
        ax.set_xlabel('Share of Priority Keywords (%)')
        ax.set_xlim(0, 100)
        ax.grid(axis='x')
        ax.invert_yaxis()
        for index, bar in enumerate(bars):
            pct = quality_df.iloc[index]['Share']
            cnt = int(quality_df.iloc[index]['Count'])
            ax.text(min(pct + 1.2, 96), bar.get_y() + bar.get_height() / 2, f"{pct:.0f}% ({cnt})", va='center', fontsize=9, color='#111827')
        graphs['client'].append({'name': 'Executive Quality Snapshot', 'data': fig_to_b64(fig)})
        plt.close(fig)

        # 2) Opportunity board (business-first)
        if 'Impressions' in chart_df.columns:
            opportunity_df = chart_df.copy()
            max_impr = max(opportunity_df['Impressions'].max(), 1)
            opportunity_df['Opportunity_Index'] = (100 - opportunity_df['Proximity_Score']) * np.log1p(opportunity_df['Impressions']) / np.log1p(max_impr)
            opportunity_df = opportunity_df[opportunity_df['Impressions'] > 0].nlargest(8, 'Opportunity_Index').sort_values('Opportunity_Index')

            if not opportunity_df.empty:
                fig, ax = plt.subplots(figsize=(11.0, 6.8), constrained_layout=True)
                y_pos = np.arange(len(opportunity_df))
                labels = [wrap_label(k, 32) for k in opportunity_df['Keyword']]
                ax.hlines(y=y_pos, xmin=0, xmax=opportunity_df['Opportunity_Index'], color='#93c5fd', linewidth=2)
                ax.scatter(opportunity_df['Opportunity_Index'], y_pos, color='#1d4ed8', s=80, zorder=3)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels, fontsize=9)
                ax.set_xlabel('Opportunity Index (higher = higher expected SEO upside)')
                ax.set_title('Client Opportunity Board')
                ax.grid(axis='x')
                for index, row in enumerate(opportunity_df.itertuples(index=False)):
                    ax.text(row.Opportunity_Index + 0.02, index, f"S {row.Proximity_Score:.0f} | I {int(row.Impressions):,}", va='center', fontsize=8, color='#374151')
                graphs['client'].append({'name': 'Top SEO Opportunities', 'data': fig_to_b64(fig)})
                plt.close(fig)

        # 3) Client impact matrix (minimal labels to avoid clutter)
        if 'Impressions' in chart_df.columns and 'Clicks' in chart_df.columns:
            fig, ax = plt.subplots(figsize=(10.8, 6.8), constrained_layout=True)
            bubble_size = np.clip((chart_df['Clicks'] + 1) * 9, 30, 900)
            scatter = ax.scatter(
                chart_df['Proximity_Score'],
                chart_df['Impressions'],
                s=bubble_size,
                c=chart_df['Proximity_Score'],
                cmap='RdYlGn',
                alpha=0.78,
                edgecolors='white',
                linewidth=0.8,
            )

            impression_line = chart_df['Impressions'].median()
            ax.axvline(70, linestyle='--', color='#6b7280', linewidth=1)
            ax.axhline(impression_line, linestyle='--', color='#6b7280', linewidth=1)
            ax.set_title('Client Impact Matrix: Visibility vs Semantic Fit')
            ax.set_xlabel('Pure Semantic Score')
            ax.set_ylabel('Impressions')
            ax.set_xlim(0, 100)
            ax.grid()
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Pure Semantic Score')

            graphs['client'].append({'name': 'Client Impact Matrix', 'data': fig_to_b64(fig)})
            plt.close(fig)

        # ===== TECHNICAL/CONSULTANT GRAPHS =====
        # 4) Score distribution
        fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
        bins = np.arange(0, 105, 5)
        ax.hist(chart_df['Proximity_Score'], bins=bins, color='#1d4ed8', edgecolor='white', alpha=0.9)
        ax.set_xlabel('Pure Semantic Score')
        ax.set_ylabel('Keyword Count')
        ax.set_title('Score Distribution (5-point buckets)')
        ax.set_xlim(0, 100)
        ax.grid(axis='y')
        q25, q50, q75 = chart_df['Proximity_Score'].quantile([0.25, 0.5, 0.75])
        for value, label in [(q25, 'Q1'), (q50, 'Median'), (q75, 'Q3')]:
            ax.axvline(value, color='#6b7280', linestyle='--', linewidth=1)
            ax.text(value + 0.5, ax.get_ylim()[1] * 0.92, label, fontsize=8, color='#374151')
        graphs['technical'].append({'name': 'Score Distribution', 'data': fig_to_b64(fig)})
        plt.close(fig)

        # 5) Clicks correlation (hexbin style)
        if 'Clicks' in chart_df.columns:
            fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
            hb = ax.hexbin(
                chart_df['Proximity_Score'],
                chart_df['Clicks'],
                gridsize=24,
                cmap='viridis',
                mincnt=1,
            )
            if len(chart_df) >= 2:
                trend = np.polyfit(chart_df['Proximity_Score'], chart_df['Clicks'], 1)
                x_line = np.linspace(0, 100, 120)
                y_line = trend[0] * x_line + trend[1]
                ax.plot(x_line, y_line, color='#111827', linewidth=1.8, linestyle='--', label='Trend')
                ax.legend(frameon=False, loc='upper left')
            ax.set_xlabel('Pure Semantic Score')
            ax.set_ylabel('Clicks')
            ax.set_title('Clicks vs Semantic Score (density)')
            ax.set_xlim(0, 100)
            ax.grid()
            cbar = plt.colorbar(hb, ax=ax)
            cbar.set_label('Point Density')
            graphs['technical'].append({'name': 'Clicks vs Score', 'data': fig_to_b64(fig)})
            plt.close(fig)

        # 6) SERP position correlation (clean scatter)
        if 'Position' in chart_df.columns:
            fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
            scatter = ax.scatter(
                chart_df['Proximity_Score'],
                chart_df['Position'],
                alpha=0.7,
                s=65,
                c=chart_df['Proximity_Score'],
                cmap='plasma',
                edgecolors='white',
                linewidth=0.6,
            )
            ax.set_xlabel('Pure Semantic Score')
            ax.set_ylabel('SERP Position (lower is better)')
            ax.set_title('SERP Position vs Semantic Score')
            ax.set_xlim(0, 100)
            ax.invert_yaxis()
            ax.grid()
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Pure Semantic Score')
            graphs['technical'].append({'name': 'Position vs Score', 'data': fig_to_b64(fig)})
            plt.close(fig)

        # 7) Geographic performance
        chart_df['Location'] = chart_df['URL'].apply(extract_location_from_url)
        geo_perf = chart_df.groupby('Location').agg(
            Avg_Score=('Proximity_Score', 'mean'),
            Keyword_Count=('Keyword', 'count')
        ).round(2).sort_values('Avg_Score', ascending=False).head(10)

        if not geo_perf.empty:
            fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
            labels = [wrap_label(loc, 18) for loc in geo_perf.index]
            colors = ['#10b981' if score >= 70 else '#f59e0b' if score >= 50 else '#ef4444' for score in geo_perf['Avg_Score']]
            bars = ax.barh(labels, geo_perf['Avg_Score'], color=colors)
            ax.set_xlim(0, 100)
            ax.set_xlabel('Average Pure Semantic Score')
            ax.set_title('Geographic Performance')
            ax.grid(axis='x')
            for idx, (bar, count) in enumerate(zip(bars, geo_perf['Keyword_Count'])):
                width = bar.get_width()
                ax.text(width + 1, idx, f"{width:.0f} ({int(count)} kw)", va='center', fontsize=8)
            graphs['technical'].append({'name': 'Geographic Performance', 'data': fig_to_b64(fig)})
            plt.close(fig)

        # 8) Market distribution donut with legend (no text overlap)
        if chart_df['Location'].nunique() > 1:
            market_dist = chart_df['Location'].value_counts().head(8)
            fig, ax = plt.subplots(figsize=(10.8, 6.2), constrained_layout=True)
            wedges, _ = ax.pie(
                market_dist.values,
                startangle=90,
                wedgeprops={'width': 0.42, 'edgecolor': 'white'},
                colors=plt.cm.Set3(np.linspace(0.05, 0.95, len(market_dist)))
            )
            ax.set_title('Keyword Distribution Across Markets')
            legend_labels = [f"{loc} ({count})" for loc, count in market_dist.items()]
            ax.legend(wedges, legend_labels, title='Markets', loc='lower center', bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=2)
            graphs['technical'].append({'name': 'Multi-Market Distribution', 'data': fig_to_b64(fig)})
            plt.close(fig)
    
    except Exception as e:
        print(f"Graph generation error: {e}")
    
    return graphs

def run_analysis(job_id, gsc_path, keywords_path, output_dir):
    """Run analysis in background"""
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        analysis_jobs[job_id]['status'] = 'processing'
        analysis_jobs[job_id]['progress'] = 10
        
        # Load data
        gsc_df = pd.read_csv(gsc_path)
        keywords = load_keywords(keywords_path)
        analysis_jobs[job_id]['progress'] = 20
        
        results = []
        
        # Analyze each keyword
        for i, keyword in enumerate(keywords):
            # Find best URL
            url_info = get_best_url_for_keyword(gsc_df, keyword)
            if not url_info:
                continue
            
            # Fetch page content
            page_data = fetch_page_content(url_info['url'])
            if not page_data:
                continue
            
            # Calculate proximity
            proximity = calculate_semantic_proximity(keyword, page_data, embedding_model)
            
            results.append({
                'Keyword': keyword,
                'URL': url_info['url'],
                'Proximity_Score': proximity,
                'Clicks': url_info['clicks'],
                'Impressions': url_info['impressions'],
                'Position': url_info['position'],
                'Page_Title': page_data['title'][:50],
                'Meta_Description': page_data['meta_description'][:50],
            })
            
            progress = 20 + (i + 1) / len(keywords) * 60
            analysis_jobs[job_id]['progress'] = int(progress)
        
        if not results:
            raise ValueError("No matching keywords found in GSC data")
        
        result_df = pd.DataFrame(results)
        analysis_jobs[job_id]['progress'] = 85
        
        # Save CSV
        csv_file = output_dir / "results.csv"
        result_df.to_csv(csv_file, index=False)
        
        # Generate graphs
        graphs = generate_graphs(result_df)
        
        # Save Excel
        excel_file = output_dir / "results.xlsx"
        with pd.ExcelWriter(excel_file) as writer:
            result_df.to_excel(writer, sheet_name='Results', index=False)
            
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
            summary.to_excel(writer, sheet_name='Summary', index=False)
        
        analysis_jobs[job_id].update({
            'status': 'completed',
            'progress': 100,
            'result_df': result_df,
            'graphs': graphs,
            'output_dir': str(output_dir),
        })
        
    except Exception as e:
        analysis_jobs[job_id] = {'status': 'error', 'error': str(e)}
        print(f"Analysis error: {e}")


@app.route('/')
def index():
    """Main upload page"""
    lang = request.args.get('lang', 'en')
    return render_template('index.html', lang=lang, get_text=lambda key: get_text(lang, key))


@app.route('/upload', methods=['POST'])
def upload():
    """Handle file uploads"""
    lang = request.form.get('lang', 'en')
    
    if 'gsc' not in request.files or 'keywords' not in request.files:
        return "Missing files", 400
    
    gsc_file = request.files['gsc']
    keywords_file = request.files['keywords']
    
    if not gsc_file or not keywords_file:
        return "No files", 400
    
    if not (allowed_file(gsc_file.filename, {'csv'}) and 
            allowed_file(keywords_file.filename, {'xlsx', 'xls', 'csv'})):
        return "Invalid file types", 400
    
    job_id = str(uuid.uuid4())[:8]
    upload_dir = Path(app.config['UPLOAD_FOLDER']) / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    gsc_path = upload_dir / secure_filename(gsc_file.filename)
    keywords_path = upload_dir / secure_filename(keywords_file.filename)
    
    gsc_file.save(str(gsc_path))
    keywords_file.save(str(keywords_path))
    
    output_dir = RESULTS_FOLDER / job_id
    
    # Initialize job
    analysis_jobs[job_id] = {
        'status': 'queued',
        'progress': 0,
        'error': None,
        'result_df': None,
        'graphs': {},
        'output_dir': str(output_dir),
    }
    
    # Start analysis
    thread = threading.Thread(
        target=run_analysis,
        args=(job_id, str(gsc_path), str(keywords_path), str(output_dir))
    )
    thread.daemon = True
    thread.start()
    
    return redirect(url_for('results', job_id=job_id, lang=lang))


@app.route('/api/status/<job_id>')
def get_status(job_id):
    """Get job status"""
    if job_id not in analysis_jobs:
        return jsonify({'status': 'not_found'}), 404
    
    job = analysis_jobs[job_id]
    return jsonify({
        'status': job.get('status'),
        'progress': job.get('progress', 0),
        'error': job.get('error'),
    })


@app.route('/results/<job_id>')
def results(job_id):
    """Display results"""
    lang = request.args.get('lang', 'en')
    
    job = analysis_jobs.get(job_id, {})
    status = job.get('status')
    
    if not status:
        return render_template('loading.html', job_id=job_id, lang=lang)
    
    if status == 'error':
        return f"{get_text(lang, 'error_prefix')} {job.get('error')}", 400
    
    if status != 'completed':
        return render_template('loading.html', job_id=job_id, lang=lang)
    
    result_df = job.get('result_df')
    graphs = job.get('graphs', {})
    
    if result_df is None:
        return render_template('loading.html', job_id=job_id, lang=lang)
    
    # Prepare metrics
    avg_score = result_df['Proximity_Score'].mean()
    avg_clicks = result_df['Clicks'].mean()
    avg_impressions = result_df['Impressions'].mean()
    metrics = {
        'avg_score': avg_score,
        'keywords_analyzed': len(result_df),
        'avg_clicks': avg_clicks,
        'avg_impressions': avg_impressions,
    }
    
    # Prepare detailed tables with rows only - Enhanced for Geo-SEO
    tables = {
        'top_20': [[
            i + 1,
            str(row['Keyword']).strip(),  # Ensure keyword is string
            row['URL'],
            f"{row['Proximity_Score']:.2f}",
            int(row['Clicks']),
            int(row['Impressions']),
            f"{int(row['Position'])}" if row['Position'] > 0 else '-'
        ] for i, (_, row) in enumerate(result_df.nlargest(20, 'Proximity_Score').iterrows())],
        'bottom_20': [[
            i + 1,
            str(row['Keyword']).strip(),  # Ensure keyword is string
            row['URL'],
            f"{row['Proximity_Score']:.2f}",
            int(row['Clicks']),
            int(row['Impressions']),
            f"{int(row['Position'])}" if row['Position'] > 0 else '-'
        ] for i, (_, row) in enumerate(result_df.nsmallest(20, 'Proximity_Score').iterrows())]
    }
    
    # Generate geo-SEO insights
    geo_insights = generate_geo_insights(result_df)
    
    return render_template('results.html', 
                         job_id=job_id,
                         lang=lang,
                         metrics=metrics, 
                         graphs=graphs, 
                         tables=tables,
                         geo_insights=geo_insights,
                         get_text=lambda key: get_text(lang, key))


@app.route('/download/<job_id>/<filename>')
def download_file(job_id, filename):
    """Download result file"""
    job = analysis_jobs.get(job_id, {})
    output_dir = Path(job.get('output_dir', ''))
    
    filepath = output_dir / filename
    if not filepath.exists():
        return "File not found", 404
    
    return send_file(filepath, as_attachment=True, download_name=filename)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    app.run(debug=False, port=args.port, host='127.0.0.1')
