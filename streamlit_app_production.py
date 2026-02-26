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
import re
import unicodedata
from translations import get_text


APP_I18N = {
    'en': {
        'lang_selector': 'Language',
        'how_it_works': 'How it works',
        'required_files_info': 'Required Files: Upload both your GSC export CSV and your strategic priority keywords file (Excel or CSV)',
        'upload_gsc': 'Upload GSC CSV',
        'upload_keywords': 'Upload Keywords File',
        'matching_sensitivity': 'Keyword Matching Sensitivity',
        'strict': 'Strict',
        'balanced': 'Balanced',
        'wide': 'Wide',
        'visualizations': 'Visualizations',
        'client_graphs': 'Client Insights Graphs',
        'technical_graphs': 'Technical Graphs',
        'graph_opportunity': 'Top SEO Opportunities',
        'graph_position': 'Position vs Semantic Score',
        'graph_bucket': 'Semantic Quality Mix',
        'opportunity_idx': 'Opportunity Index',
        'serp_position': 'SERP Position (lower is better)',
        'count': 'Count',
        'gsc_caption': 'Export from Google Search Console with Query, Page, Clicks, Impressions, Position columns',
        'keywords_caption': 'Your strategic keywords list - the app will find best URLs for each and analyze semantic alignment',
        'matching_help': 'Strict = very close query match only, Balanced = recommended default, Wide = broader match coverage',
        'analyzing_spinner': 'Analyzing... (this may take a minute)',
        'missing_gsc_columns': 'Could not find required GSC columns (Query + Page)',
        'detected_columns': 'Detected columns',
        'cannot_extract_keywords': 'Could not extract keywords from file',
        'keywords_column_hint': "Make sure your file has a column named 'Mot-clé', 'Keyword', or 'Keywords'",
        'loaded_keywords': 'Loaded',
        'priority_keywords_loaded_suffix': 'priority keywords from file',
        'view_loaded_keywords': 'View loaded keywords',
        'matching_info_prefix': 'Now matching',
        'matching_info_suffix': 'keywords with GSC data and analyzing semantic proximity...',
        'failed_load_model': 'Failed to load embedding model',
        'retry_wide': 'No matches found with the selected mode. Retrying automatically with Wide mode...',
        'analyzing_status': 'Analyzing',
        'analyzed_success_prefix': 'Analyzed',
        'analyzed_success_mid': 'keywords (matched out of',
        'analyzed_success_suffix': ') using mode',
        'fetch_warning_prefix': 'matched URLs could not be fetched live. Fallback semantic inputs were used (query + URL text).',
        'fetch_details': 'Fetch failures details',
        'no_gsc_match': 'keywords had no GSC match',
        'no_matching_keywords': 'No matching keywords found in GSC data',
        'matched_but_not_analyzable': 'Keywords matched in GSC, but no rows were analyzable after processing.',
        'quick_diagnostic': 'Quick diagnostic',
        'diag_loaded_keywords': 'Loaded strategic keywords',
        'diag_gsc_rows': 'GSC rows available',
        'diag_query_col': 'GSC query column used',
        'diag_matched_before_fetch': 'Matched keywords before fetch',
        'diag_fetch_failures': 'URL fetch failures',
        'sample_gsc_queries': 'Sample GSC queries (first 20)',
        'sample_strategic_keywords': 'Sample strategic keywords (first 20)',
        'show_detailed_error': 'Show detailed error',
        'both_files_required': 'Both files are required to run the analysis',
        'missing_gsc_file': 'GSC Data CSV is missing',
        'missing_keywords_file': 'Priority Keywords file is missing (Excel or CSV with your strategic keywords)',
        'what_are_priority_keywords': 'What are Priority Keywords?',
    },
    'fr': {
        'lang_selector': 'Langue',
        'how_it_works': 'Comment ça marche',
        'required_files_info': 'Fichiers requis : importez votre export GSC CSV et votre fichier de mots-clés stratégiques (Excel ou CSV)',
        'upload_gsc': 'Importer le CSV GSC',
        'upload_keywords': 'Importer le fichier de mots-clés',
        'matching_sensitivity': 'Sensibilité du matching des mots-clés',
        'strict': 'Strict',
        'balanced': 'Équilibré',
        'wide': 'Large',
        'visualizations': 'Visualisations',
        'client_graphs': 'Graphes orientés client',
        'technical_graphs': 'Graphes techniques',
        'graph_opportunity': 'Top opportunités SEO',
        'graph_position': 'Position vs Score sémantique',
        'graph_bucket': 'Répartition qualité sémantique',
        'opportunity_idx': "Indice d'opportunité",
        'serp_position': 'Position SERP (plus bas = meilleur)',
        'count': 'Volume',
        'gsc_caption': 'Export Google Search Console avec colonnes Query, Page, Clicks, Impressions, Position',
        'keywords_caption': 'Votre liste de mots-clés stratégiques – l’app trouve les meilleures URLs et analyse l’alignement sémantique',
        'matching_help': 'Strict = correspondances très proches, Équilibré = recommandé, Large = couverture plus large',
        'analyzing_spinner': 'Analyse en cours... (cela peut prendre une minute)',
        'missing_gsc_columns': 'Colonnes GSC requises introuvables (Query + Page)',
        'detected_columns': 'Colonnes détectées',
        'cannot_extract_keywords': 'Impossible d’extraire les mots-clés du fichier',
        'keywords_column_hint': "Vérifiez qu’une colonne existe avec 'Mot-clé', 'Keyword' ou 'Keywords'",
        'loaded_keywords': 'Chargé',
        'priority_keywords_loaded_suffix': 'mots-clés prioritaires depuis le fichier',
        'view_loaded_keywords': 'Voir les mots-clés chargés',
        'matching_info_prefix': 'Matching de',
        'matching_info_suffix': 'mots-clés avec les données GSC et analyse de proximité sémantique...',
        'failed_load_model': 'Échec du chargement du modèle d’embedding',
        'retry_wide': 'Aucun match avec le mode sélectionné. Nouvelle tentative automatique en mode Large...',
        'analyzing_status': 'Analyse',
        'analyzed_success_prefix': 'Analysé',
        'analyzed_success_mid': 'mots-clés (matchés sur',
        'analyzed_success_suffix': ') avec le mode',
        'fetch_warning_prefix': 'URLs matchées non récupérées en live. Fallback utilisé (query + texte URL).',
        'fetch_details': 'Détails des échecs de fetch',
        'no_gsc_match': 'mots-clés sans match GSC',
        'no_matching_keywords': 'Aucun mot-clé correspondant trouvé dans les données GSC',
        'matched_but_not_analyzable': 'Des mots-clés sont matchés dans GSC, mais aucune ligne n’est analysable après traitement.',
        'quick_diagnostic': 'Diagnostic rapide',
        'diag_loaded_keywords': 'Mots-clés stratégiques chargés',
        'diag_gsc_rows': 'Lignes GSC disponibles',
        'diag_query_col': 'Colonne query GSC utilisée',
        'diag_matched_before_fetch': 'Mots-clés matchés avant fetch',
        'diag_fetch_failures': 'Échecs de fetch URL',
        'sample_gsc_queries': 'Exemples de requêtes GSC (20 premières)',
        'sample_strategic_keywords': 'Exemples de mots-clés stratégiques (20 premiers)',
        'show_detailed_error': 'Afficher l’erreur détaillée',
        'both_files_required': 'Les deux fichiers sont requis pour lancer l’analyse',
        'missing_gsc_file': 'Le fichier CSV GSC est manquant',
        'missing_keywords_file': 'Le fichier de mots-clés prioritaires est manquant (Excel ou CSV)',
        'what_are_priority_keywords': 'Que sont les mots-clés prioritaires ?',
    },
    'ar': {
        'lang_selector': 'اللغة',
        'how_it_works': 'كيف تعمل الأداة',
        'required_files_info': 'الملفات المطلوبة: ارفع ملف GSC CSV وملف الكلمات المفتاحية الاستراتيجية (Excel أو CSV)',
        'upload_gsc': 'رفع ملف GSC CSV',
        'upload_keywords': 'رفع ملف الكلمات المفتاحية',
        'matching_sensitivity': 'حساسية مطابقة الكلمات المفتاحية',
        'strict': 'صارم',
        'balanced': 'متوازن',
        'wide': 'واسع',
        'visualizations': 'الرسوم البيانية',
        'client_graphs': 'رسوم موجهة للعميل',
        'technical_graphs': 'رسوم تقنية',
        'graph_opportunity': 'أفضل فرص SEO',
        'graph_position': 'الترتيب مقابل الدرجة الدلالية',
        'graph_bucket': 'توزيع الجودة الدلالية',
        'opportunity_idx': 'مؤشر الفرصة',
        'serp_position': 'ترتيب SERP (الأقل أفضل)',
        'count': 'العدد',
        'gsc_caption': 'تصدير من Google Search Console مع الأعمدة Query وPage وClicks وImpressions وPosition',
        'keywords_caption': 'قائمة كلماتك المفتاحية الاستراتيجية - الأداة تختار أفضل URL وتحلل المحاذاة الدلالية',
        'matching_help': 'صارم = مطابقة دقيقة، متوازن = موصى به، واسع = تغطية أوسع',
        'analyzing_spinner': 'جارٍ التحليل... (قد يستغرق ذلك دقيقة)',
        'missing_gsc_columns': 'تعذر العثور على أعمدة GSC المطلوبة (Query + Page)',
        'detected_columns': 'الأعمدة المكتشفة',
        'cannot_extract_keywords': 'تعذر استخراج الكلمات المفتاحية من الملف',
        'keywords_column_hint': "تأكد من وجود عمود باسم 'Mot-clé' أو 'Keyword' أو 'Keywords'",
        'loaded_keywords': 'تم تحميل',
        'priority_keywords_loaded_suffix': 'كلمة مفتاحية استراتيجية من الملف',
        'view_loaded_keywords': 'عرض الكلمات المفتاحية المحمّلة',
        'matching_info_prefix': 'جارٍ مطابقة',
        'matching_info_suffix': 'كلمة مفتاحية مع بيانات GSC وتحليل التقارب الدلالي...',
        'failed_load_model': 'فشل تحميل نموذج التضمين',
        'retry_wide': 'لم يتم العثور على مطابقات بالوضع الحالي. إعادة المحاولة تلقائياً بوضع واسع...',
        'analyzing_status': 'تحليل',
        'analyzed_success_prefix': 'تم تحليل',
        'analyzed_success_mid': 'كلمة مفتاحية (مطابقة من أصل',
        'analyzed_success_suffix': ') باستخدام الوضع',
        'fetch_warning_prefix': 'عنوان URL مطابق لم يتم جلبه مباشرة. تم استخدام fallback (query + نص URL).',
        'fetch_details': 'تفاصيل فشل الجلب',
        'no_gsc_match': 'كلمات مفتاحية بدون مطابقة في GSC',
        'no_matching_keywords': 'لم يتم العثور على كلمات مفتاحية مطابقة في بيانات GSC',
        'matched_but_not_analyzable': 'تمت مطابقة كلمات مفتاحية في GSC ولكن لا توجد صفوف قابلة للتحليل بعد المعالجة.',
        'quick_diagnostic': 'تشخيص سريع',
        'diag_loaded_keywords': 'الكلمات الاستراتيجية المحمّلة',
        'diag_gsc_rows': 'صفوف GSC المتاحة',
        'diag_query_col': 'عمود الاستعلام المستخدم من GSC',
        'diag_matched_before_fetch': 'الكلمات المطابقة قبل الجلب',
        'diag_fetch_failures': 'إخفاقات جلب URL',
        'sample_gsc_queries': 'عينات استعلامات GSC (أول 20)',
        'sample_strategic_keywords': 'عينات الكلمات الاستراتيجية (أول 20)',
        'show_detailed_error': 'عرض الخطأ التفصيلي',
        'both_files_required': 'الملفان مطلوبان لتشغيل التحليل',
        'missing_gsc_file': 'ملف GSC CSV مفقود',
        'missing_keywords_file': 'ملف الكلمات المفتاحية الاستراتيجية مفقود (Excel أو CSV)',
        'what_are_priority_keywords': 'ما هي الكلمات المفتاحية الاستراتيجية؟',
    },
}


def t(language, key):
    if key in APP_I18N.get(language, {}):
        return APP_I18N[language][key]
    return get_text(language, key)

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
def normalize_text(text):
    if pd.isna(text):
        return ""
    value = str(text).lower().strip()
    value = unicodedata.normalize('NFKD', value)
    value = ''.join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def detect_column(columns, candidates):
    normalized = {col: normalize_text(col) for col in columns}
    for candidate in candidates:
        candidate_norm = normalize_text(candidate)
        for original, norm in normalized.items():
            if norm == candidate_norm or candidate_norm in norm:
                return original
    return None


def extract_keywords_from_dataframe(df):
    if df is None or df.empty:
        return []

    keyword_col = detect_column(
        df.columns,
        ['mot cle', 'mot-clé', 'mot clé', 'keyword', 'keywords', 'query']
    )

    if keyword_col is None:
        if len(df.columns) > 1:
            keyword_col = df.columns[1]
        else:
            keyword_col = df.columns[0]

    values = df[keyword_col].dropna().astype(str).tolist()
    cleaned = []
    for value in values:
        text = value.strip()
        if not text or len(text) < 2 or text.isdigit():
            continue
        normalized = normalize_text(text)
        if normalized in {'mot cle', 'keyword', 'keywords', 'nan'}:
            continue
        cleaned.append(text)
    return cleaned


def load_keywords_excel(file_obj):
    keywords = []
    try:
        filename = (getattr(file_obj, 'name', '') or '').lower()

        if filename.endswith('.csv'):
            file_obj.seek(0)
            df = pd.read_csv(file_obj, sep=None, engine='python')
            keywords.extend(extract_keywords_from_dataframe(df))

            if len(keywords) < 3:
                file_obj.seek(0)
                df_no_header = pd.read_csv(file_obj, sep=None, engine='python', header=None)
                keywords.extend(extract_keywords_from_dataframe(df_no_header))
        else:
            file_obj.seek(0)
            xls = pd.ExcelFile(file_obj)
            for sheet_name in xls.sheet_names:
                file_obj.seek(0)
                df = pd.read_excel(file_obj, sheet_name=sheet_name)
                keywords.extend(extract_keywords_from_dataframe(df))

                file_obj.seek(0)
                df_skip = pd.read_excel(file_obj, sheet_name=sheet_name, skiprows=3)
                keywords.extend(extract_keywords_from_dataframe(df_skip))

    except Exception as e:
        st.warning(f"Could not parse keywords file: {e}")

    seen = set()
    unique_keywords = []
    for item in keywords:
        key = normalize_text(item)
        if key and key not in seen:
            seen.add(key)
            unique_keywords.append(item.strip())
    return unique_keywords

def parse_french_number(val):
    """Convert French formatted numbers to float (e.g., '1 234,5' → 1234.5)"""
    if pd.isna(val) or val == '':
        return 0.0
    val = str(val).strip()
    # Remove spaces (French thousands separator)
    val = val.replace(' ', '')
    val = val.replace('\u00a0', '')
    val = val.replace('\u202f', '')
    # Replace comma with dot (French decimal)
    val = val.replace(',', '.')
    val = val.replace('%', '')
    try:
        return float(val)
    except:
        return 0.0

def prepare_gsc_data(gsc_df):
    prepared = gsc_df.copy()

    query_col = detect_column(prepared.columns, ['query', 'queries', 'top queries'])
    page_col = detect_column(prepared.columns, ['page', 'landing page', 'url', 'top pages'])

    if not query_col or not page_col:
        return None, None, None

    for col in ['Clicks', 'Impressions', 'Position']:
        metric_col = detect_column(prepared.columns, [col])
        if metric_col:
            prepared[metric_col] = prepared[metric_col].apply(parse_french_number)

    prepared['_query_norm'] = prepared[query_col].apply(normalize_text)
    prepared = prepared[prepared['_query_norm'].str.len() > 0].copy()

    return prepared, query_col, page_col


def get_best_url_for_keyword(prepared_gsc_df, keyword, query_col, page_col, matching_mode='Balanced'):
    keyword_norm = normalize_text(keyword)
    if not keyword_norm:
        return None

    keyword_tokens = [token for token in keyword_norm.split() if len(token) > 1]
    if not keyword_tokens:
        return None

    mode = (matching_mode or 'Balanced').lower()
    if mode == 'strict':
        min_overlap = 0.75
    elif mode == 'wide':
        min_overlap = 0.30
    else:
        min_overlap = 0.45

    def keyword_match_score(query_norm):
        if not query_norm:
            return 0.0

        if keyword_norm == query_norm:
            return 1.0
        if keyword_norm in query_norm:
            return 0.95

        overlap = sum(1 for token in keyword_tokens if token in query_norm)
        overlap_ratio = overlap / len(keyword_tokens)
        if overlap_ratio >= min_overlap:
            return 0.5 + (overlap_ratio * 0.4)

        return 0.0

    scored = prepared_gsc_df.copy()
    scored['_match_score'] = scored['_query_norm'].apply(keyword_match_score)
    matched = scored[scored['_match_score'] > 0].copy()

    if matched.empty:
        return None

    clicks_col = detect_column(matched.columns, ['Clicks'])
    impressions_col = detect_column(matched.columns, ['Impressions'])
    position_col = detect_column(matched.columns, ['Position'])

    matched['_perf_score'] = (
        matched.get(clicks_col, 0) * 2
        + matched.get(impressions_col, 0) * 0.5
        - matched.get(position_col, 0) * 0.1
    )
    matched['_final_score'] = (matched['_match_score'] * 1000) + matched['_perf_score']
    best_row = matched.loc[matched['_final_score'].idxmax()]

    return {
        'url': str(best_row[page_col]),
        'query': str(best_row[query_col]),
        'clicks': int(best_row.get(clicks_col, 0)) if clicks_col else 0,
        'impressions': int(best_row.get(impressions_col, 0)) if impressions_col else 0,
        'position': float(best_row.get(position_col, 0)) if position_col else 0.0,
    }

def fetch_page_content(url):
    """Fetch page title, meta, and content"""
    try:
        request_url = url
        if not request_url.startswith('http'):
            request_url = 'https://' + request_url
        
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
        response = requests.get(request_url, headers=headers, timeout=8, allow_redirects=True)
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
        
        parsed = urlparse(request_url)
        url_text = (parsed.netloc + parsed.path)[:100]
        
        return {
            'title': title or "",
            'meta_description': meta_desc,
            'content': text,
            'url_text': url_text,
            'full_url': request_url,
        }, None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)}"

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

def generate_charts(result_df, labels=None):
    """Generate modern charts without text overlap"""
    charts = {}
    labels = labels or {}
    
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
    ax.set_title(labels.get('graph_quality', 'Keyword Quality Distribution'))
    ax.set_xlabel(labels.get('count', 'Count'))
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
    ax.set_title(labels.get('graph_distribution', 'Score Distribution'))
    ax.set_xlabel(labels.get('col_score', 'Semantic Proximity Score'))
    ax.set_ylabel(labels.get('count', 'Keyword Count'))
    ax.grid(axis='y')
    charts['distribution'] = fig
    
    # 3) Clicks vs Score (scatter)
    if 'Clicks' in result_df.columns:
        fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
        scatter = ax.scatter(result_df['Proximity_Score'], result_df['Clicks'],
                           s=80, c=result_df['Proximity_Score'], cmap='viridis', alpha=0.7, edgecolors='white')
        ax.set_title(labels.get('graph_clicks_vs_score', 'Clicks vs Semantic Score'))
        ax.set_xlabel(labels.get('col_score', 'Semantic Proximity Score'))
        ax.set_ylabel(labels.get('col_clicks', 'Clicks'))
        ax.grid()
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(labels.get('col_score', 'Score'))
        charts['clicks'] = fig

    if 'Position' in result_df.columns:
        fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
        scatter = ax.scatter(
            result_df['Proximity_Score'],
            result_df['Position'],
            s=75,
            c=result_df['Proximity_Score'],
            cmap='plasma',
            alpha=0.7,
            edgecolors='white'
        )
        ax.set_title(labels.get('graph_position', 'Position vs Semantic Score'))
        ax.set_xlabel(labels.get('col_score', 'Semantic Proximity Score'))
        ax.set_ylabel(labels.get('serp_position', 'SERP Position (lower is better)'))
        ax.invert_yaxis()
        ax.grid()
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(labels.get('col_score', 'Score'))
        charts['position'] = fig

    if 'Impressions' in result_df.columns:
        opp_df = result_df.copy()
        max_impr = max(opp_df['Impressions'].max(), 1)
        opp_df['Opportunity_Index'] = (100 - opp_df['Proximity_Score']) * np.log1p(opp_df['Impressions']) / np.log1p(max_impr)
        opp_df = opp_df[opp_df['Impressions'] > 0].nlargest(10, 'Opportunity_Index').sort_values('Opportunity_Index')
        if not opp_df.empty:
            fig, ax = plt.subplots(figsize=(10.8, 6.0), constrained_layout=True)
            y_pos = np.arange(len(opp_df))
            ax.hlines(y=y_pos, xmin=0, xmax=opp_df['Opportunity_Index'], color='#93c5fd', linewidth=2)
            ax.scatter(opp_df['Opportunity_Index'], y_pos, color='#1d4ed8', s=85, zorder=3)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(opp_df['Keyword'].astype(str).str.slice(0, 36), fontsize=9)
            ax.set_xlabel(labels.get('opportunity_idx', 'Opportunity Index'))
            ax.set_title(labels.get('graph_opportunity', 'Top SEO Opportunities'))
            ax.grid(axis='x')
            charts['opportunity'] = fig

    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    ax.pie(
        [excellent, good, fair, poor],
        labels=['80+', '60-79', '40-59', '<40'],
        autopct='%1.0f%%',
        colors=['#10b981', '#84cc16', '#f59e0b', '#ef4444'],
        startangle=90
    )
    ax.set_title(labels.get('graph_bucket', 'Semantic Quality Mix'))
    charts['bucket'] = fig
    
    return charts

# ============= MAIN APP =============
def main():
    lang_options = {
        'Français': 'fr',
        'English': 'en',
        'العربية': 'ar',
    }
    selected_lang_label = st.selectbox(
        "Language / Langue / اللغة",
        list(lang_options.keys()),
        index=0,
    )
    language = lang_options[selected_lang_label]

    if language == 'ar':
        st.markdown(
            """
            <style>
            .stApp { direction: rtl; }
            </style>
            """,
            unsafe_allow_html=True,
        )

    st.title(f"📊 {t(language, 'title')}")
    st.markdown(f"*{t(language, 'subtitle')}*")
    
    # Overview
    with st.expander(f"ℹ️ {t(language, 'how_it_works')}", expanded=False):
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
    st.info(f"📋 **{t(language, 'required_files_info')}**")
    
    # File uploads
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"1️⃣ {t(language, 'gsc_label')}")
        st.caption(t(language, 'gsc_caption'))
        gsc_file = st.file_uploader(t(language, 'upload_gsc'), type=['csv'], key='gsc', help="Export your GSC data as CSV with queries and landing pages")
    
    with col2:
        st.subheader(f"2️⃣ {t(language, 'keywords_label')} ⭐")
        st.caption(t(language, 'keywords_caption'))
        keywords_file = st.file_uploader(t(language, 'upload_keywords'), type=['xlsx', 'xls', 'csv'], key='keywords', help="Excel or CSV file with your priority keywords in 'Mot-clé' or 'Keyword' column")

    matching_mode = st.select_slider(
        f"🎯 {t(language, 'matching_sensitivity')}",
        options=[t(language, 'strict'), t(language, 'balanced'), t(language, 'wide')],
        value=t(language, 'balanced'),
        help=t(language, 'matching_help')
    )
    
    st.markdown("---")
    
    # Analyze button
    if st.button(f"🚀 {t(language, 'analyze_btn')}", type="primary", use_container_width=True):
        if gsc_file and keywords_file:
            with st.spinner(f"🔄 {t(language, 'analyzing_spinner')}"):
                try:
                    # Load files
                    gsc_df = pd.read_csv(gsc_file, dtype={'Clicks': str, 'Impressions': str, 'Position': str})

                    prepared_gsc, query_col, page_col = prepare_gsc_data(gsc_df)
                    if prepared_gsc is None:
                        st.error(f"❌ {t(language, 'missing_gsc_columns')}")
                        st.write(f"{t(language, 'detected_columns')}: {', '.join(gsc_df.columns.astype(str).tolist())}")
                        return
                    
                    # Load keywords from file (handles Excel and CSV)
                    keywords = load_keywords_excel(keywords_file)
                    
                    if not keywords:
                        st.error(f"❌ {t(language, 'cannot_extract_keywords')}")
                        st.warning(t(language, 'keywords_column_hint'))
                        return
                    
                    st.success(f"✅ {t(language, 'loaded_keywords')} {len(keywords)} {t(language, 'priority_keywords_loaded_suffix')}")
                    with st.expander(f"📋 {t(language, 'view_loaded_keywords')}"):
                        st.write(", ".join(keywords[:20]))
                        if len(keywords) > 20:
                            st.write(f"... and {len(keywords) - 20} more")
                    
                    st.info(
                        f"🔍 {t(language, 'matching_info_prefix')} {len(keywords)} "
                        f"{t(language, 'matching_info_suffix')} (mode: {matching_mode})"
                    )
                    
                    # Load model
                    model = load_embedding_model()
                    if not model:
                        st.error(f"❌ {t(language, 'failed_load_model')}")
                        return
                    
                    # Analyze
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    reverse_mode = {
                        t(language, 'strict'): 'Strict',
                        t(language, 'balanced'): 'Balanced',
                        t(language, 'wide'): 'Wide',
                    }
                    selected_mode_en = reverse_mode.get(matching_mode, 'Balanced')
                    mode_attempts = [selected_mode_en]
                    if selected_mode_en != 'Wide':
                        mode_attempts.append('Wide')

                    results = []
                    unmatched_keywords = []
                    matched_keywords_count = 0
                    fetch_failures = 0
                    fetch_failure_details = []
                    effective_mode = selected_mode_en

                    for attempt_index, attempt_mode in enumerate(mode_attempts):
                        if attempt_index > 0:
                            st.warning(t(language, 'retry_wide'))

                        results = []
                        unmatched_keywords = []
                        matched_keywords_count = 0
                        fetch_failures = 0
                        fetch_failure_details = []

                        for i, keyword in enumerate(keywords):
                            status_text.text(
                                f"{t(language, 'analyzing_status')} ({attempt_mode}): {keyword} ({i+1}/{len(keywords)})"
                            )

                            url_info = get_best_url_for_keyword(
                                prepared_gsc,
                                keyword,
                                query_col,
                                page_col,
                                matching_mode=attempt_mode,
                            )
                            if not url_info:
                                unmatched_keywords.append(keyword)
                                continue

                            matched_keywords_count += 1

                            page_data, fetch_error = fetch_page_content(url_info['url'])
                            if not page_data:
                                fetch_failures += 1
                                fetch_failure_details.append({
                                    'Keyword': keyword,
                                    'Matched_Query': url_info.get('query', ''),
                                    'URL': url_info.get('url', ''),
                                    'Fetch_Error': fetch_error or 'Unknown error',
                                })
                                page_data = {
                                    'title': url_info.get('query', ''),
                                    'meta_description': '',
                                    'content': url_info.get('query', ''),
                                    'url_text': url_info.get('url', ''),
                                    'full_url': url_info.get('url', ''),
                                }

                            proximity = calculate_semantic_proximity(keyword, page_data, model)

                            results.append({
                                'Keyword': keyword,
                                'Matched_Query': url_info['query'],
                                'URL': url_info['url'],
                                'Proximity_Score': proximity,
                                'Clicks': int(url_info['clicks']),
                                'Impressions': int(url_info['impressions']),
                                'Position': round(url_info['position'], 1),
                            })

                            progress_bar.progress((i + 1) / len(keywords))

                        if results:
                            effective_mode = attempt_mode
                            break
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    if results:
                        result_df = pd.DataFrame(results)
                        st.session_state.results = result_df
                        chart_labels = {
                            'graph_quality': t(language, 'graph_quality'),
                            'graph_distribution': t(language, 'graph_distribution'),
                            'graph_clicks_vs_score': t(language, 'graph_clicks_vs_score'),
                            'graph_position': t(language, 'graph_position'),
                            'graph_opportunity': t(language, 'graph_opportunity'),
                            'graph_bucket': t(language, 'graph_bucket'),
                            'col_score': t(language, 'col_score'),
                            'col_clicks': t(language, 'col_clicks'),
                            'count': t(language, 'count'),
                            'opportunity_idx': t(language, 'opportunity_idx'),
                            'serp_position': t(language, 'serp_position'),
                        }
                        st.session_state.charts = generate_charts(result_df, labels=chart_labels)
                        st.success(
                            f"✅ {t(language, 'analyzed_success_prefix')} {len(results)} "
                            f"{t(language, 'analyzed_success_mid')} {len(keywords)}"
                            f"{t(language, 'analyzed_success_suffix')}: {effective_mode}"
                        )

                        if fetch_failures > 0:
                            st.warning(
                                f"⚠️ {fetch_failures} {t(language, 'fetch_warning_prefix')}"
                            )
                            with st.expander(f"🔎 {t(language, 'fetch_details')}"):
                                st.dataframe(
                                    pd.DataFrame(fetch_failure_details),
                                    use_container_width=True,
                                    hide_index=True,
                                )

                        if unmatched_keywords:
                            with st.expander(f"⚠️ {len(unmatched_keywords)} {t(language, 'no_gsc_match')}"):
                                st.write(", ".join(unmatched_keywords[:50]))
                                if len(unmatched_keywords) > 50:
                                    st.write(f"... and {len(unmatched_keywords) - 50} more")
                    else:
                        if matched_keywords_count == 0:
                            st.error(f"❌ {t(language, 'no_matching_keywords')}")
                        else:
                            st.error(f"❌ {t(language, 'matched_but_not_analyzable')}")
                        st.info(f"🔎 {t(language, 'quick_diagnostic')}")
                        st.write(f"- {t(language, 'diag_loaded_keywords')}: {len(keywords)}")
                        st.write(f"- {t(language, 'diag_gsc_rows')}: {len(prepared_gsc)}")
                        st.write(f"- {t(language, 'diag_query_col')}: {query_col}")
                        st.write(f"- {t(language, 'diag_matched_before_fetch')}: {matched_keywords_count}")
                        st.write(f"- {t(language, 'diag_fetch_failures')}: {fetch_failures}")
                        sample_queries = prepared_gsc[query_col].dropna().astype(str).head(20).tolist()
                        with st.expander(t(language, 'sample_gsc_queries')):
                            for query in sample_queries:
                                st.write(f"- {query}")
                        with st.expander(t(language, 'sample_strategic_keywords')):
                            for keyword in keywords[:20]:
                                st.write(f"- {keyword}")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    with st.expander(t(language, 'show_detailed_error')):
                        st.code(traceback.format_exc())
        else:
            st.error(f"⚠️ **{t(language, 'both_files_required')}**")
            if not gsc_file:
                st.write(f"- ❌ {t(language, 'missing_gsc_file')}")
            if not keywords_file:
                st.write(f"- ❌ **{t(language, 'missing_keywords_file')}**")
            
            with st.expander(f"💡 {t(language, 'what_are_priority_keywords')}"):
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
        
        st.subheader(f"📈 {t(language, 'results_title')}")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(t(language, 'avg_score'), f"{result_df['Proximity_Score'].mean():.1f}")
        with col2:
            st.metric(t(language, 'keywords_analyzed'), len(result_df))
        with col3:
            st.metric(t(language, 'avg_clicks'), f"{result_df['Clicks'].mean():.1f}")
        with col4:
            st.metric(t(language, 'avg_impressions'), f"{result_df['Impressions'].mean():.1f}")
        
        st.markdown("---")
        
        # Charts
        st.subheader(f"📊 {t(language, 'visualizations')}")
        
        st.markdown(f"**{t(language, 'client_graphs')}**")
        col1, col2 = st.columns(2)
        with col1:
            if 'quality' in st.session_state.charts:
                st.pyplot(st.session_state.charts['quality'], use_container_width=True)
            if 'opportunity' in st.session_state.charts:
                st.pyplot(st.session_state.charts['opportunity'], use_container_width=True)
        with col2:
            if 'bucket' in st.session_state.charts:
                st.pyplot(st.session_state.charts['bucket'], use_container_width=True)
            if 'distribution' in st.session_state.charts:
                st.pyplot(st.session_state.charts['distribution'], use_container_width=True)

        st.markdown(f"**{t(language, 'technical_graphs')}**")
        col3, col4 = st.columns(2)
        with col3:
            if 'clicks' in st.session_state.charts:
                st.pyplot(st.session_state.charts['clicks'], use_container_width=True)
        with col4:
            if 'position' in st.session_state.charts:
                st.pyplot(st.session_state.charts['position'], use_container_width=True)
        
        st.markdown("---")
        
        # Tables
        st.subheader(f"🎯 {t(language, 'detailed_results')}")
        
        tab1, tab2 = st.tabs([t(language, 'top_keywords'), t(language, 'bottom_keywords')])
        
        with tab1:
            top_20 = result_df.nlargest(20, 'Proximity_Score')
            st.dataframe(top_20, use_container_width=True, hide_index=True)
        
        with tab2:
            bottom_20 = result_df.nsmallest(20, 'Proximity_Score')
            st.dataframe(bottom_20, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Download
        st.subheader(f"⬇️ {t(language, 'export_data')}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = result_df.to_csv(index=False)
            st.download_button(
                label=t(language, 'download_csv'),
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
                    label=t(language, 'download_excel'),
                    data=f.read(),
                    file_name=f"semantic_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.ms-excel",
                    use_container_width=True
                )

if __name__ == '__main__':
    main()
