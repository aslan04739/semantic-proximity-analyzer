# Semantic Proximity Analyzer

A powerful AI-powered tool to analyze keyword-to-page semantic alignment using sentence embeddings and semantic similarity scoring.

## 🎯 What It Does

1. **Loads GSC Data**: Import Google Search Console export (CSV)
2. **Matches Keywords**: Identify your priority keywords from Excel/CSV
3. **Selects Best URL**: Uses GSC performance metrics to find the best ranking page per keyword
4. **Analyzes Content**: Fetches page title, meta description, and body content
5. **Calculates Score**: Uses AI embeddings (all-MiniLM-L6-v2) to compute pure semantic proximity (0-100)
6. **Generates Insights**: Modern charts showing quality distribution, score trends, and click correlation
7. **Exports Results**: CSV and Excel with detailed metrics

## 🚀 Quick Start (Local)

### Requirements
- Python 3.11+
- pip or conda

### Installation

```bash
# Clone/download repository
cd semantic-proximity-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements_streamlit.txt
```

### Run Locally

```bash
streamlit run streamlit_app_production.py
```

Open http://localhost:8501 in your browser.

## ☁️ Deploy to Streamlit Cloud (Free)

### Prerequisites
1. GitHub account
2. Streamlit Community Cloud account (free signup at https://streamlit.io/cloud)

### Deployment Steps

#### 1. Push to GitHub

```bash
# Initialize git repo (if not already done)
git init
git add .
git commit -m "Initial commit: Semantic Proximity Analyzer"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR-USERNAME/semantic-proximity-analyzer.git
git branch -M main
git push -u origin main
```

#### 2. Deploy on Streamlit Cloud

1. Visit https://share.streamlit.io
2. Paste your GitHub repo URL
3. Select:
   - **Repository**: `YOUR-USERNAME/semantic-proximity-analyzer`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app_production.py`
4. Click "Deploy"
5. Streamlit automatically watches for changes—every push auto-deploys!

#### 3. Optional: Add Secrets (if needed)

If you need API keys later, add them in Streamlit Cloud dashboard:
1. Go to your app settings
2. Click "Secrets" tab
3. Add key-value pairs (e.g., `api_key = "xxx"`)

In code, access via:
```python
secret = st.secrets["api_key"]
```

## 📊 Features

### Input Files

**GSC Data (CSV)**
- Export from Google Search Console
- Requires columns: Query, Landing page, Clicks, Impressions, Position

**Priority Keywords (Excel/CSV)**
- Single column with keywords
- Column name should contain: "Keyword", "Mot-clé", or "Clé"

### Output Metrics

- **Pure Semantic Score** (0-100): AI embeddings-based similarity between keyword and page content
- **Quality Distribution**: Excellent (80+), Good (60-80), Fair (40-60), Poor (<40)
- **Correlation Charts**: Click trends vs semantic score
- **CSV & Excel**: Full results with sortable tables

## 🔧 Technical Stack

- **Model**: SentenceTransformers (all-MiniLM-L6-v2 embeddings)
- **Similarity**: Cosine similarity between keyword and page elements
- **Weights**: Title (35%) + Meta (25%) + Content (30%) + URL (10%)
- **Visualization**: Matplotlib with modern styling (no text overlaps)
- **Deployment**: Streamlit Cloud (serverless Python)

## 📁 File Structure

```
.
├── streamlit_app_production.py   # Main Streamlit app
├── dashboard_app.py              # Flask backend (optional local alternative)
├── requirements_streamlit.txt    # Python dependencies
├── .streamlit/config.toml        # Streamlit theme & config
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## 🐛 Troubleshooting

**"Model loading stuck"**
- First run downloads the embedding model (~50MB)
- Streamlit caches it, subsequent runs are instant
- On Cloud, takes ~30s on first load

**"Download buttons don't work"**
- Check browser console for errors
- Ensure file doesn't exceed Streamlit's 200MB limit

**"Can't find my keywords column"**
- Excel file must have column named: "Keyword", "Mot-clé", "Clé", or "Keywords"
- Or ensure it's in the first column

**"No matching keywords in GSC"**
- Verify keyword format matches GSC query strings
- Some keywords may not appear in GSC—that's normal

## 📈 Example Workflow

1. Export GSC data for last 3 months
2. Create Excel with 50-100 priority keywords
3. Upload both files
4. Wait for analysis (~1 min for 100 keywords)
5. Review metrics and charts
6. Identify keywords with low semantic alignment
7. Export CSV for content team
8. Optimize pages with low scores

## 🤝 Contributing

Suggestions? Issues? Improvements?
- Fork the repo
- Create feature branch
- Submit pull request

## 📝 License

MIT License - free to use and modify

## 📧 Support

For questions or bugs:
- Check GitHub Issues
- Create new issue with details

---

**Made with ❤️ for SEO teams**
