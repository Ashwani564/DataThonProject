# Datathon Visualization Report

## Badminton Shot Analysis - Comprehensive Diagrams

This document provides an overview of all generated visualizations for the datathon presentation.

### Generated: November 12, 2025
### Total Diagrams: 15
### Data Source: `output/analysisCSV/analysis_dataDetection_20251112_124805.csv`
### Output Location: `output/diagrams/`

---

## Visualization Catalog

### 1. Shot Type Distribution (`01_shot_type_distribution.png`)
- **Type:** Bar Chart & Pie Chart
- **Purpose:** Shows the overall distribution of different shot types (Forehand vs Backhand)
- **Insights:** Displays both absolute counts and percentage breakdown

### 2. Velocity Comparison (`02_velocity_comparison.png`)
- **Type:** Box Plots (2 panels)
- **Purpose:** Compares max and average racket velocities across shot types
- **Insights:** Shows velocity ranges, medians, and outliers for each shot type

### 3. Duration Analysis (`03_duration_analysis.png`)
- **Type:** Multi-panel Analysis (Histogram, Box Plot, Time Series, Bar Chart)
- **Purpose:** Comprehensive analysis of shot duration patterns
- **Insights:** Duration distributions, statistics by shot type, temporal trends, and player comparisons

### 4. Angle Correlation Heatmap (`04_angle_correlation_heatmap.png`)
- **Type:** Correlation Heatmap
- **Purpose:** Shows relationships between different body angles
- **Insights:** Identifies correlated movements between hip-knee, elbow, and torso angles

### 5. Velocity vs Duration Scatter (`05_velocity_vs_duration.png`)
- **Type:** Scatter Plots (Max & Avg Velocity)
- **Purpose:** Explores relationship between shot duration and velocity
- **Insights:** Reveals patterns in how shot length affects velocity for different shot types

### 6. Player Performance Comparison (`06_player_performance.png`)
- **Type:** Multi-panel Dashboard (Bar Charts & Statistics Table)
- **Purpose:** Compares performance metrics across different players
- **Insights:** Shot counts, velocities, shot type preferences, and comprehensive statistics

### 7. Movement Analysis (`07_movement_analysis.png`)
- **Type:** 4-panel Analysis (Bar Charts, Histograms, Scatter)
- **Purpose:** Analyzes center of gravity movement patterns
- **Insights:** Horizontal vs vertical movement, distributions, and relationships

### 8. Angle Distributions (`08_angle_distributions.png`)
- **Type:** Violin Plots (3 panels)
- **Purpose:** Shows distribution of hip-knee, elbow, and torso angles by shot type
- **Insights:** Reveals angle patterns and variations for different shot types

### 9. Temporal Analysis (`09_temporal_analysis.png`)
- **Type:** Time Series Plots with Moving Average
- **Purpose:** Tracks velocity changes over the course of the match
- **Insights:** Performance trends, fatigue indicators, and shot-by-shot variations

### 10. Performance Metrics Radar (`10_performance_radar.png`)
- **Type:** Radar/Spider Chart
- **Purpose:** Multi-dimensional comparison of normalized performance metrics
- **Insights:** Holistic view of velocity, duration, angles, and movement for each shot type

### 11. Velocity Distribution KDE (`11_velocity_kde.png`)
- **Type:** Kernel Density Estimation Plots
- **Purpose:** Smooth probability distributions of velocity metrics
- **Insights:** Detailed distribution shapes and overlaps between shot types

### 12. Summary Statistics Table (`12_summary_statistics.png`)
- **Type:** Formatted Data Table
- **Purpose:** Comprehensive numerical summary of all key metrics
- **Insights:** Quick reference for average values across all measured parameters

### 13. Scatter Matrix (`13_scatter_matrix.png`)
- **Type:** Pairwise Scatter Plot Matrix
- **Purpose:** Explores relationships between multiple variables simultaneously
- **Insights:** Identifies correlations and patterns across key performance metrics

### 14. Shot Efficiency Analysis (`14_shot_efficiency.png`)
- **Type:** 4-panel Analysis
- **Purpose:** Analyzes efficiency metrics including velocity-per-duration ratios
- **Insights:** Movement efficiency, velocity-angle relationships, and frame duration patterns

### 15. Comprehensive Overview Dashboard (`15_comprehensive_overview.png`)
- **Type:** Executive Dashboard (9 panels)
- **Purpose:** High-level summary of all key findings in one view
- **Insights:** Perfect for presentations - combines pie charts, box plots, time series, and statistics

---

## Key Metrics Analyzed

1. **Velocity Metrics**
   - Max Racket Velocity (px/s)
   - Average Racket Velocity (px/s)

2. **Timing Metrics**
   - Shot Duration (seconds)
   - Start/End Frames
   - Frame Duration

3. **Body Angle Metrics**
   - Hip-Knee Angle (degrees)
   - Elbow Angle (degrees)
   - Torso Angle (degrees)

4. **Movement Metrics**
   - Center of Gravity Horizontal Movement (pixels)
   - Center of Gravity Vertical Movement (pixels)
   - Total Movement (calculated)

5. **Performance Metrics**
   - Velocity Efficiency (velocity/duration)
   - Shot Type Distribution
   - Player Comparisons

---

## Technical Details

- **Resolution:** 300 DPI (publication quality)
- **Format:** PNG with tight bounding boxes
- **Color Palette:** Professional "husl" palette with consistent colors across charts
- **Style:** Seaborn darkgrid for enhanced readability
- **Total Data Points:** 240 shots analyzed

---

## Usage for Datathon Presentation

1. **Overview Slide:** Use `15_comprehensive_overview.png` for introduction
2. **Shot Distribution:** Use `01_shot_type_distribution.png` to show data composition
3. **Performance Metrics:** Use `10_performance_radar.png` for comparative analysis
4. **Detailed Analysis:** Use specific charts (2-14) for deep dives into particular aspects
5. **Summary:** Use `12_summary_statistics.png` for conclusions

---

## Files Generated

All visualization files are saved in: `output/diagrams/`

Total Size: ~6.9 MB
Format: High-resolution PNG files
Naming Convention: Sequential numbering (01-15) for easy presentation ordering

---

**Generated by:** Comprehensive Visualization Script
**Script Location:** `visualize_analysis.py`
**Dependencies:** pandas, matplotlib, seaborn, numpy
