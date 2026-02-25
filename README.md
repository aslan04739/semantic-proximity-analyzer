# Semantic Proximity Analyzer

A minimalistic macOS dashboard app for analyzing semantic proximity in Google Search Console data.

## Quick Start

1. Launch the app from Spotlight: Search for **"Semantic Analyzer Dashboard"**
2. Upload your GSC CSV file in the dashboard
3. View results and download the analysis

## Building the App

If you need to rebuild the app:

```bash
bash build_dashboard_app.sh
```

The app will be installed to `~/Applications/Semantic Analyzer Dashboard.app`

## Project Files

- `dashboard_embedded_app.py` - Main app entry point (Flask + WebView)
- `dashboard_app.py` - Backend API for analysis
- `templates/index.html` - Minimalistic dashboard UI
- `build_dashboard_app.sh` - PyInstaller build script
- `scripts/create_minimal_icon.py` - Icon generator
- `assets/icon_minimal.icns` - App icon
- `.venv_dashboard/` - Python 3.12 environment for building

## Requirements

- Python 3.12 (for building)
- pandas, scikit-learn, scipy
- Flask, pywebview
- PyInstaller
- Customization guide
- Building from source

### 🔍 **I Want Deep Understanding**
→ Read: [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md)
- How the tool works (5-step pipeline)
- What each graph & table shows
- Metric definitions
- Use cases & workflows

---

## ✨ What You Get

### 📊 **10 Professional Graphs**
- 5 client-ready visualizations (executive, distribution, action matrix, opportunities, pages)
- 5 advanced analysis charts (SERP impact, funnel, efficiency, metrics, breakdown)
- All in French, 200 DPI, publication-ready

### 📑 **Excel Workbook with 5 Sheets**
- Executive summary with KPIs
- Complete keyword analysis  
- Page-level strategic insights
- Quick wins (high-impact opportunities)
- Gap analysis (missing keywords)

### 🎯 **Semantic Alignment Scoring**
- AI-powered embeddings analyze REAL page content
- Scores each keyword 0-100% alignment  
- Identifies optimization opportunities
- Tracks improvement over time

---

## 🔧 What You Need

### Data Files (You Provide):
1. **GSC Entry** (CSV)
   - Exported from Google Search Console
   - Contains: queries, clicks, impressions, positions

2. **Business Keywords** (Excel .xlsx)
   - Sheet named: "Intention Business"
   - Keywords in first column

### System Requirements:
- **OS**: macOS 10.13+ (Intel or Apple Silicon)
- **Storage**: ~2GB (app + dependencies + results)
- **Internet**: For web page scraping
- **RAM**: 4GB+ recommended

---

## 💡 Key Features

✅ **Fully Automated Pipeline**
- Keyword extraction, GSC matching, web scraping, AI analysis, visualization generation

✅ **Real Content Analysis**  
- Scrapes actual page content (not just URLs)
- Fallback to URL structure if scraping unavailable
- Extracts: titles, meta, h1, body text

✅ **Client-Ready Output**
- Beautiful modern graphs in French
- Color-coded Excel with recommendations
- Actionable insights highlighted

✅ **Professional Design**
- Minimalist macOS app interface
- Clean, intuitive workflow
- Non-technical users can operate it

✅ **Reusable & Flexible**
- Run on different keyword lists
- Analyze different GSC properties
- Save results with timestamps
- Track progress over time

---

## 🎯 Use Cases

**For SEO Agencies**
- Audit client sites for content-keyword alignment
- Generate monthly optimization reports
- Identify quick wins for client presentations

**For In-House Teams**  
- Content strategy audits
- Competitive keyword analysis
- Performance tracking over time

**For Freelancers**
- Quick deliverable for clients  
- Professional, shareable results
- Minimal setup time

---

## 📁 File Structure

```
Semantic proximity/
├── dist/Semantic Analyzer.app      ← 🚀 THE APP (double-click to run)
│
├── Python Scripts/
│   ├── semantic_proximity_gsc.py          (core analysis)
│   ├── generate_modern_client_graphs.py   (client visualizations) 
│   ├── generate_advanced_graphs.py        (advanced analysis)
│   └── generate_modern_client_tables.py   (Excel workbook)
│
├── Documentation/
│   ├── README.md                   (you are here)
│   ├── QUICK_START.md              (user guide)
│   ├── SYSTEM_OVERVIEW.md          (technical deep-dive)
│   └── README_APP.md               (developer guide)
│
└── Sample Data/
    ├── Manpower GSC data - Semantic proximity.csv
    └── Mots-cles_Business_Info_1.xlsx
```

---

## 🎨 Inside the App

The minimalistically-designed macOS app provides:

```
┌─────────────────────────────────────┐
│  Semantic Proximity Analyzer         │
│  Analyse sémantique de vos mots-clés │
│                                      │
│  Google Search Console CSV: [Browse] │
│  Excel Mots-clés métier:    [Browse] │
│  Dossier de sortie:         [Browse] │
│                                      │
│  [Progress Bar - Hidden until run]  │
│  [Status Text Area]                 │
│                                      │
│              [Analyser] [Ouvrir]    │
└─────────────────────────────────────┘
```

**Blue Theme**: Professional, modern, accessible  
**French Labels**: Ready for European clients  
**Minimalist Design**: Nothing unnecessary

---

## 🔄 How It Works (Simplified)

```
Your Files
    ↓
[GSC Data] + [Keywords]
    ↓
Match keywords to GSC queries (best URL per keyword)
    ↓
Scrape real page content for each URL
    ↓  
Calculate semantic similarity (AI embeddings)
    ↓
Generate insights (alignment scores, priority ranking)
    ↓
Create visualizations & Excel reports
    ↓
📊 Output Folder with 10 graphs + Excel
```

---

## 🎓 Learning Resources

**First-time use?**
1. Read: QUICK_START.md (5 min read)
2. Prepare your data (5 min)
3. Launch app and run analysis (5-10 min runtime)
4. Review results (10 min exploration)

**Want to understand more?**
→ Read SYSTEM_OVERVIEW.md for:
- Detailed metric explanations
- Use cases & workflows
- Technical architecture
- FAQ section

**Want to customize?**
→ Read README_APP.md for:
- Build instructions
- Code structure
- Customization guide
- Dependency list

---

## ✅ What's Included (Version 1.0)

- ✅ Fully functional macOS application
- ✅ 4 Python analysis scripts
- ✅ 10 graph generation templates
- ✅ Excel workbook generation
- ✅ Web scraping with fallbacks
- ✅ AI embeddings for semantic analysis
- ✅ Complete documentation
- ✅ Sample data for testing

---

## 🚨 Common Questions

**Q: Do I need Python installed?**  
A: No! The .app includes everything bundled. Just double-click and go.

**Q: Is my data secure?**  
A: Yes. Analysis runs locally on your Mac. Only your public web pages are accessed.

**Q: How many keywords can I analyze?**  
A: Tested with 30+ keywords. Larger sets (100+) may take 30+ minutes.

**Q: Can I run this on Windows?**  
A: The .app is macOS only. But the Python scripts work anywhere. See README_APP.md for Windows setup.

**Q: What languages are supported?**  
A: French, English, Spanish, German, and 100+ other languages via SentenceTransformer.

---

## 📊 Example Output

### You'll Receive:

**Graphs** (PNG files @ 200 DPI)
- Executive dashboard showing overall health
- Distribution bar chart showing alignment zones  
- Action matrix identifying quick-wins
- Top opportunities ranked by ROI
- Page performance breakdown
- SERP impact analysis (4-panel)
- Quality funnel visualization
- Efficiency matrix scatter plot
- Top keywords comparison (4-panel)
- Segment performance breakdown

**Excel Workbook** (5 professional sheets)
- KPIs and metrics
- Detailed keyword-level analysis
- Page-by-page strategic overview
- High-impact quick-wins list
- Keyword gap analysis

**Everything in French** (labels, recommendations, insights)

---

## 🚀 Getting Started Now

### Option A: Launch the App
```bash
Double-click: dist/Semantic Analyzer.app
```

### Option B: Command Line (For Developers)
```bash
# Install dependencies
pip install -r requirements_app.txt

# Run Python app directly  
python3 semantic_analyzer_app.py

# Or run single analysis script
python3 semantic_proximity_gsc.py --gsc-csv data.csv --keywords-excel keywords.xlsx
```

---

## 🔗 Navigation

| Need | Read This | Time |
|------|-----------|------|
| How to use | [QUICK_START.md](QUICK_START.md) | 5 min |
| Understanding outputs | [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md) | 15 min |
| Developer setup | [README_APP.md](README_APP.md) | 10 min |
| Ready to start? | Double-click the .app! | 2 min |

---

## 📞 Troubleshooting

**App won't launch?**
→ See "Troubleshooting" section in QUICK_START.md

**File format errors?**  
→ Check "Data Preparation" in QUICK_START.md

**Want to rebuild the app?**
→ See "Building from Source" in README_APP.md

**Technical questions?**
→ Check docstrings in Python files

---

## 📈 Version Info

**Current Version**: 1.0.0  
**Release Date**: February 24, 2024  
**Status**: Production Ready  
**Python Version**: 3.8+  
**macOS Version**: 10.13+

---

## 🎉 Ready?

### 👉 **Start Here:**
1. Review [QUICK_START.md](QUICK_START.md) 
2. Prepare your GSC CSV and Keywords Excel
3. Double-click: `dist/Semantic Analyzer.app`
4. Follow the simple 3-step process
5. Get your professional analysis

**That's it!** The app handles everything else. ✨

---

**Built for professionals. Designed for simplicity. Powered by AI.**

Happy analyzing! 🚀
