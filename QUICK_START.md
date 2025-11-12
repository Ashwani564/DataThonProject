# Quick Start Guide - Datathon Visualization

## 🎯 Generate All Diagrams in One Command

```bash
# Navigate to project
cd /Users/ashwani/Desktop/Rallymate

# Activate virtual environment
source venv/bin/activate

# Generate all 15 diagrams
python visualize_analysis.py
```

## 📊 What Gets Generated

**Location:** `output/diagrams/`
**Total Files:** 15 PNG images
**Total Size:** ~6.7 MB
**Resolution:** 300 DPI (publication quality)

## 📁 Files Created

| File | Description | Use Case |
|------|-------------|----------|
| `01_shot_type_distribution.png` | Bar & pie charts | Overview of shot distribution |
| `02_velocity_comparison.png` | Box plots | Velocity statistics comparison |
| `03_duration_analysis.png` | Multi-panel | Shot duration deep dive |
| `04_angle_correlation_heatmap.png` | Heatmap | Body mechanics relationships |
| `05_velocity_vs_duration.png` | Scatter plots | Performance relationships |
| `06_player_performance.png` | Dashboard | Player-to-player comparison |
| `07_movement_analysis.png` | 4-panel | CoG movement patterns |
| `08_angle_distributions.png` | Violin plots | Angle distribution analysis |
| `09_temporal_analysis.png` | Time series | Performance over time |
| `10_performance_radar.png` | Radar chart | Multi-metric comparison |
| `11_velocity_kde.png` | KDE plots | Smooth distributions |
| `12_summary_statistics.png` | Table | Numerical summary |
| `13_scatter_matrix.png` | Matrix | Pairwise relationships |
| `14_shot_efficiency.png` | 4-panel | Efficiency metrics |
| `15_comprehensive_overview.png` | Dashboard | Executive summary |

## 🎨 Presentation Tips

### For Introductory Slide
Use: `15_comprehensive_overview.png` - Shows everything at a glance

### For Technical Deep Dive
Use: `04_angle_correlation_heatmap.png`, `13_scatter_matrix.png`

### For Performance Comparison
Use: `06_player_performance.png`, `10_performance_radar.png`

### For Trends & Patterns
Use: `09_temporal_analysis.png`, `11_velocity_kde.png`

### For Data Summary
Use: `12_summary_statistics.png`

## ⚡ Fast Regeneration

If you need to regenerate specific plots or all plots:

```bash
# Just run the script again
python visualize_analysis.py
```

The script automatically:
- ✅ Finds the latest CSV file
- ✅ Creates output directory if needed
- ✅ Overwrites old diagrams with fresh ones
- ✅ Handles missing data gracefully

## 🔧 Customization

To modify colors, fonts, or layout:
1. Edit `visualize_analysis.py`
2. Look for the specific plot function (e.g., `plot_shot_type_distribution`)
3. Modify matplotlib/seaborn parameters
4. Rerun: `python visualize_analysis.py`

## 📦 Data Source

The visualizations use data from:
```
output/analysisCSV/analysis_dataDetection_20251112_124805.csv
```

**Data Points:** 240 shots analyzed
**Columns:** 15 metrics including velocity, duration, angles, movement

## 🐛 Troubleshooting

### Error: "No module named 'seaborn'"
```bash
pip install seaborn
```

### Error: "No CSV files found"
First run the analysis:
```bash
python analyze_processed_video.py
```

### Diagrams look pixelated
They're saved at 300 DPI. Ensure you're viewing them at 100% zoom.

## 📖 Full Documentation

For detailed description of each diagram, see:
- `VISUALIZATION_REPORT.md` - Comprehensive documentation
- `README.md` - Project overview

## ✅ Checklist for Datathon

- [x] All 15 diagrams generated
- [x] High resolution (300 DPI)
- [x] Professional color scheme
- [x] Clear labels and titles
- [x] Grid overlays for readability
- [x] Consistent styling across all charts
- [x] Documentation complete

---

**Last Generated:** November 12, 2025
**Script:** `visualize_analysis.py`
**Status:** ✅ Ready for presentation
