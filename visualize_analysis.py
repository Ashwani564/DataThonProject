#!/usr/bin/env python3
"""
Comprehensive Visualization Script for Badminton Shot Analysis - Datathon Edition
Generates 15+ diagrams using matplotlib and seaborn based on the actual CSV columns
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import glob
from math import pi

# Set style for professional-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_latest_csv():
    """Load the most recent analysis CSV file"""
    csv_files = glob.glob('output/analysisCSV/*.csv')
    if not csv_files:
        raise FileNotFoundError("No CSV files found in output/analysisCSV/")
    latest_csv = max(csv_files, key=lambda x: Path(x).stat().st_mtime)
    print(f"📊 Loading: {latest_csv}")
    return pd.read_csv(latest_csv)

def create_output_dir():
    """Create output directory for diagrams"""
    output_dir = Path('output/diagrams')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def plot_shot_type_distribution(df, output_dir):
    """1. Shot Type Distribution - Bar Chart with Pie"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    shot_counts = df['Shot_Type'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E1D3']
    
    # Bar chart
    bars = ax1.bar(shot_counts.index, shot_counts.values, color=colors[:len(shot_counts)])
    ax1.set_title('Distribution of Shot Types', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Shot Type', fontsize=12)
    ax1.set_ylabel('Number of Shots', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Pie chart
    ax2.pie(shot_counts.values, labels=shot_counts.index, autopct='%1.1f%%',
            colors=colors[:len(shot_counts)], startangle=90, textprops={'fontsize': 12})
    ax2.set_title('Shot Type Percentage', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / '01_shot_type_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: Shot Type Distribution")

def plot_velocity_comparison(df, output_dir):
    """2. Velocity Comparison - Box Plot"""
    plt.figure(figsize=(14, 7))
    
    shot_types = df['Shot_Type'].unique()
    data_max = [df[df['Shot_Type'] == st]['Max_Racket_Velocity_px_s'].values for st in shot_types]
    data_avg = [df[df['Shot_Type'] == st]['Avg_Racket_Velocity_px_s'].values for st in shot_types]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Max velocity box plot
    bp1 = ax1.boxplot(data_max, labels=shot_types, patch_artist=True, showmeans=True)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for patch, color in zip(bp1['boxes'], colors[:len(shot_types)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_title('Max Racket Velocity by Shot Type', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Shot Type', fontsize=12)
    ax1.set_ylabel('Max Velocity (px/s)', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # Avg velocity box plot
    bp2 = ax2.boxplot(data_avg, labels=shot_types, patch_artist=True, showmeans=True)
    for patch, color in zip(bp2['boxes'], colors[:len(shot_types)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_title('Average Racket Velocity by Shot Type', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Shot Type', fontsize=12)
    ax2.set_ylabel('Avg Velocity (px/s)', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '02_velocity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: Velocity Comparison")

def plot_duration_analysis(df, output_dir):
    """3. Shot Duration Analysis"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Duration histogram by shot type
    for shot_type in df['Shot_Type'].unique():
        data = df[df['Shot_Type'] == shot_type]['Duration_Seconds']
        ax1.hist(data, alpha=0.6, label=shot_type, bins=20, edgecolor='black')
    ax1.set_title('Duration Distribution by Shot Type', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Duration (seconds)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Duration box plot
    shot_types = df['Shot_Type'].unique()
    data = [df[df['Shot_Type'] == st]['Duration_Seconds'].values for st in shot_types]
    bp = ax2.boxplot(data, labels=shot_types, patch_artist=True, showmeans=True)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for patch, color in zip(bp['boxes'], colors[:len(shot_types)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_title('Duration Statistics by Shot Type', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Shot Type', fontsize=12)
    ax2.set_ylabel('Duration (seconds)', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    # Duration over shot sequence
    ax3.plot(df['Shot_Number'], df['Duration_Seconds'], marker='o', linestyle='-', 
            markersize=3, alpha=0.6, color='#4ECDC4')
    ax3.set_title('Duration Over Shot Sequence', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Shot Number', fontsize=12)
    ax3.set_ylabel('Duration (seconds)', fontsize=12)
    ax3.grid(alpha=0.3)
    
    # Average duration by player
    if 'Player' in df.columns:
        avg_duration = df.groupby('Player')['Duration_Seconds'].mean()
        ax4.bar(avg_duration.index, avg_duration.values, color='#FF6B6B', alpha=0.7)
        ax4.set_title('Average Duration by Player', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Player', fontsize=12)
        ax4.set_ylabel('Avg Duration (seconds)', fontsize=12)
        ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '03_duration_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: Duration Analysis")

def plot_angle_correlation_heatmap(df, output_dir):
    """4. Angle Correlation Heatmap"""
    plt.figure(figsize=(12, 10))
    
    angle_columns = ['Min_Hip_Knee_Angle_deg', 'Avg_Hip_Knee_Angle_deg', 
                     'Min_Elbow_Angle_deg', 'Avg_Elbow_Angle_deg', 'Avg_Torso_Angle_deg']
    
    correlation_matrix = df[angle_columns].corr()
    
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1)
    
    plt.title('Correlation Between Body Angles', fontsize=16, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / '04_angle_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: Angle Correlation Heatmap")

def plot_velocity_vs_duration(df, output_dir):
    """5. Velocity vs Duration Scatter Plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors_map = {'Forehand': '#FF6B6B', 'Backhand': '#4ECDC4', 'Serve': '#45B7D1'}
    
    # Max velocity vs duration
    for shot_type in df['Shot_Type'].unique():
        data = df[df['Shot_Type'] == shot_type]
        ax1.scatter(data['Duration_Seconds'], data['Max_Racket_Velocity_px_s'], 
                   label=shot_type, alpha=0.6, s=100, 
                   color=colors_map.get(shot_type, '#95E1D3'))
    ax1.set_title('Max Velocity vs Duration', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Duration (seconds)', fontsize=12)
    ax1.set_ylabel('Max Velocity (px/s)', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Avg velocity vs duration
    for shot_type in df['Shot_Type'].unique():
        data = df[df['Shot_Type'] == shot_type]
        ax2.scatter(data['Duration_Seconds'], data['Avg_Racket_Velocity_px_s'], 
                   label=shot_type, alpha=0.6, s=100, 
                   color=colors_map.get(shot_type, '#95E1D3'))
    ax2.set_title('Avg Velocity vs Duration', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Duration (seconds)', fontsize=12)
    ax2.set_ylabel('Avg Velocity (px/s)', fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '05_velocity_vs_duration.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: Velocity vs Duration Scatter")

def plot_player_performance(df, output_dir):
    """6. Player Performance Comparison"""
    if 'Player' not in df.columns or df['Player'].nunique() <= 1:
        print("⊘ Skipped: Player Performance (single player or no player data)")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Total shots by player
    shot_counts = df.groupby('Player').size()
    ax1.bar(shot_counts.index, shot_counts.values, color='#FF6B6B', alpha=0.7)
    ax1.set_title('Total Shots by Player', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Player', fontsize=12)
    ax1.set_ylabel('Number of Shots', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # Avg max velocity by player
    avg_velocity = df.groupby('Player')['Max_Racket_Velocity_px_s'].mean()
    ax2.bar(avg_velocity.index, avg_velocity.values, color='#4ECDC4', alpha=0.7)
    ax2.set_title('Average Max Velocity by Player', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Player', fontsize=12)
    ax2.set_ylabel('Avg Max Velocity (px/s)', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    # Shot type distribution by player
    player_shot_counts = df.groupby(['Player', 'Shot_Type']).size().unstack(fill_value=0)
    player_shot_counts.plot(kind='bar', ax=ax3, color=['#FF6B6B', '#4ECDC4'], width=0.7)
    ax3.set_title('Shot Type Distribution by Player', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Player', fontsize=12)
    ax3.set_ylabel('Number of Shots', fontsize=12)
    ax3.legend(title='Shot Type')
    ax3.tick_params(axis='x', rotation=0)
    ax3.grid(axis='y', alpha=0.3)
    
    # Performance metrics table
    ax4.axis('tight')
    ax4.axis('off')
    player_stats = df.groupby('Player').agg({
        'Shot_Number': 'count',
        'Max_Racket_Velocity_px_s': 'mean',
        'Duration_Seconds': 'mean',
        'Avg_Elbow_Angle_deg': 'mean'
    }).round(2)
    
    table_data = [[idx] + row.tolist() for idx, row in player_stats.iterrows()]
    table = ax4.table(cellText=table_data,
                      colLabels=['Player', 'Shots', 'Avg Max Vel', 'Avg Dur', 'Avg Elbow'],
                      cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    ax4.set_title('Player Statistics Summary', fontsize=14, fontweight='bold', pad=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / '06_player_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: Player Performance Comparison")

def plot_movement_analysis(df, output_dir):
    """7. Center of Gravity Movement Analysis"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    shot_types = df['Shot_Type'].unique()
    x = np.arange(len(shot_types))
    width = 0.35
    
    # Movement by shot type
    horizontal_means = [df[df['Shot_Type'] == st]['CoG_Horizontal_Movement_px'].mean() for st in shot_types]
    vertical_means = [df[df['Shot_Type'] == st]['CoG_Vertical_Movement_px'].mean() for st in shot_types]
    
    ax1.bar(x - width/2, horizontal_means, width, label='Horizontal', color='#FF6B6B', alpha=0.8)
    ax1.bar(x + width/2, vertical_means, width, label='Vertical', color='#4ECDC4', alpha=0.8)
    ax1.set_title('Avg CoG Movement by Shot Type', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Shot Type', fontsize=12)
    ax1.set_ylabel('Movement (pixels)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(shot_types)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Horizontal movement distribution
    for shot_type in shot_types:
        data = df[df['Shot_Type'] == shot_type]['CoG_Horizontal_Movement_px']
        ax2.hist(data, alpha=0.6, label=shot_type, bins=15, edgecolor='black')
    ax2.set_title('Horizontal Movement Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Horizontal Movement (pixels)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Vertical movement distribution
    for shot_type in shot_types:
        data = df[df['Shot_Type'] == shot_type]['CoG_Vertical_Movement_px']
        ax3.hist(data, alpha=0.6, label=shot_type, bins=15, edgecolor='black')
    ax3.set_title('Vertical Movement Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Vertical Movement (pixels)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Horizontal vs Vertical movement scatter
    for shot_type in shot_types:
        data = df[df['Shot_Type'] == shot_type]
        ax4.scatter(data['CoG_Horizontal_Movement_px'], data['CoG_Vertical_Movement_px'],
                   label=shot_type, alpha=0.6, s=50)
    ax4.set_title('Horizontal vs Vertical Movement', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Horizontal Movement (pixels)', fontsize=12)
    ax4.set_ylabel('Vertical Movement (pixels)', fontsize=12)
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '07_movement_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: Movement Analysis")

def plot_angle_distributions(df, output_dir):
    """8. Angle Distributions - Violin Plots"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    shot_types = df['Shot_Type'].unique()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # Hip-Knee angle violin
    data_list = [df[df['Shot_Type'] == st]['Avg_Hip_Knee_Angle_deg'].values for st in shot_types]
    parts = ax1.violinplot(data_list, positions=range(len(shot_types)), showmeans=True, showmedians=True)
    for pc, color in zip(parts['bodies'], colors[:len(shot_types)]):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    ax1.set_title('Hip-Knee Angle Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Shot Type', fontsize=12)
    ax1.set_ylabel('Angle (degrees)', fontsize=12)
    ax1.set_xticks(range(len(shot_types)))
    ax1.set_xticklabels(shot_types)
    ax1.grid(axis='y', alpha=0.3)
    
    # Elbow angle violin
    data_list = [df[df['Shot_Type'] == st]['Avg_Elbow_Angle_deg'].values for st in shot_types]
    parts = ax2.violinplot(data_list, positions=range(len(shot_types)), showmeans=True, showmedians=True)
    for pc, color in zip(parts['bodies'], colors[:len(shot_types)]):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    ax2.set_title('Elbow Angle Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Shot Type', fontsize=12)
    ax2.set_ylabel('Angle (degrees)', fontsize=12)
    ax2.set_xticks(range(len(shot_types)))
    ax2.set_xticklabels(shot_types)
    ax2.grid(axis='y', alpha=0.3)
    
    # Torso angle violin
    data_list = [df[df['Shot_Type'] == st]['Avg_Torso_Angle_deg'].values for st in shot_types]
    parts = ax3.violinplot(data_list, positions=range(len(shot_types)), showmeans=True, showmedians=True)
    for pc, color in zip(parts['bodies'], colors[:len(shot_types)]):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    ax3.set_title('Torso Angle Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Shot Type', fontsize=12)
    ax3.set_ylabel('Angle (degrees)', fontsize=12)
    ax3.set_xticks(range(len(shot_types)))
    ax3.set_xticklabels(shot_types)
    ax3.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '08_angle_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: Angle Distributions")

def plot_temporal_analysis(df, output_dir):
    """9. Temporal Analysis - Shot Sequence"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Max velocity over time
    ax1.plot(df['Shot_Number'], df['Max_Racket_Velocity_px_s'], 
            marker='o', linestyle='-', linewidth=1, markersize=3, alpha=0.6, color='#FF6B6B')
    ax1.set_title('Max Velocity Over Shot Sequence', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Shot Number', fontsize=12)
    ax1.set_ylabel('Max Velocity (px/s)', fontsize=12)
    ax1.grid(alpha=0.3)
    
    # Avg velocity over time
    ax2.plot(df['Shot_Number'], df['Avg_Racket_Velocity_px_s'], 
            marker='o', linestyle='-', linewidth=1, markersize=3, alpha=0.6, color='#4ECDC4')
    ax2.set_title('Avg Velocity Over Shot Sequence', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Shot Number', fontsize=12)
    ax2.set_ylabel('Avg Velocity (px/s)', fontsize=12)
    ax2.grid(alpha=0.3)
    
    # Shot type sequence
    colors_map = {'Forehand': '#FF6B6B', 'Backhand': '#4ECDC4', 'Serve': '#45B7D1'}
    for shot_type in df['Shot_Type'].unique():
        mask = df['Shot_Type'] == shot_type
        ax3.scatter(df[mask]['Shot_Number'], df[mask]['Max_Racket_Velocity_px_s'], 
                   label=shot_type, s=50, alpha=0.7, color=colors_map.get(shot_type, '#95E1D3'))
    ax3.set_title('Velocity by Shot Type Over Time', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Shot Number', fontsize=12)
    ax3.set_ylabel('Max Velocity (px/s)', fontsize=12)
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Moving average
    window = min(10, len(df) // 5)
    if window > 1:
        df_sorted = df.sort_values('Shot_Number')
        moving_avg = df_sorted['Max_Racket_Velocity_px_s'].rolling(window=window).mean()
        ax4.plot(df_sorted['Shot_Number'], df_sorted['Max_Racket_Velocity_px_s'], 
                alpha=0.3, label='Actual', color='gray', linewidth=1)
        ax4.plot(df_sorted['Shot_Number'], moving_avg, 
                linewidth=2, label=f'{window}-shot Moving Avg', color='#FF6B6B')
        ax4.set_title('Velocity Trend (Moving Average)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Shot Number', fontsize=12)
        ax4.set_ylabel('Max Velocity (px/s)', fontsize=12)
        ax4.legend()
        ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '09_temporal_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: Temporal Analysis")

def plot_performance_radar(df, output_dir):
    """10. Performance Metrics Radar Chart"""
    plt.figure(figsize=(12, 12))
    
    shot_types = df['Shot_Type'].unique()
    categories = ['Velocity', 'Duration', 'Hip-Knee\nAngle', 'Elbow\nAngle', 'Torso\nAngle', 'Movement']
    
    angles = [n / len(categories) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]
    
    ax = plt.subplot(111, polar=True)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, shot_type in enumerate(shot_types):
        shot_data = df[df['Shot_Type'] == shot_type]
        
        # Normalize metrics (0-1 scale)
        values = [
            shot_data['Max_Racket_Velocity_px_s'].mean() / (df['Max_Racket_Velocity_px_s'].max() + 0.001),
            shot_data['Duration_Seconds'].mean() / (df['Duration_Seconds'].max() + 0.001),
            shot_data['Avg_Hip_Knee_Angle_deg'].mean() / 180,
            shot_data['Avg_Elbow_Angle_deg'].mean() / 180,
            shot_data['Avg_Torso_Angle_deg'].mean() / 180,
            (shot_data['CoG_Horizontal_Movement_px'].abs().mean() + 
             shot_data['CoG_Vertical_Movement_px'].abs().mean()) / 
            (df['CoG_Horizontal_Movement_px'].abs().max() + df['CoG_Vertical_Movement_px'].abs().max() + 0.001)
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=shot_type, 
               color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[idx % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    ax.set_ylim(0, 1)
    plt.title('Performance Metrics Comparison (Normalized)', 
             fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / '10_performance_radar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: Performance Metrics Radar")

def plot_velocity_kde(df, output_dir):
    """11. Velocity Distribution with KDE"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Max velocity KDE
    for shot_type in df['Shot_Type'].unique():
        data = df[df['Shot_Type'] == shot_type]['Max_Racket_Velocity_px_s']
        data.plot(kind='kde', ax=axes[0], label=shot_type, linewidth=2)
    axes[0].set_title('Max Racket Velocity Distribution (KDE)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Max Velocity (px/s)', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Avg velocity KDE
    for shot_type in df['Shot_Type'].unique():
        data = df[df['Shot_Type'] == shot_type]['Avg_Racket_Velocity_px_s']
        data.plot(kind='kde', ax=axes[1], label=shot_type, linewidth=2)
    axes[1].set_title('Average Racket Velocity Distribution (KDE)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Avg Velocity (px/s)', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '11_velocity_kde.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: Velocity Distribution KDE")

def plot_summary_statistics(df, output_dir):
    """12. Summary Statistics Table"""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')
    
    summary_data = []
    for shot_type in df['Shot_Type'].unique():
        shot_data = df[df['Shot_Type'] == shot_type]
        summary_data.append([
            shot_type,
            len(shot_data),
            f"{shot_data['Max_Racket_Velocity_px_s'].mean():.1f}",
            f"{shot_data['Avg_Racket_Velocity_px_s'].mean():.1f}",
            f"{shot_data['Duration_Seconds'].mean():.2f}",
            f"{shot_data['Avg_Hip_Knee_Angle_deg'].mean():.1f}°",
            f"{shot_data['Avg_Elbow_Angle_deg'].mean():.1f}°",
            f"{shot_data['Avg_Torso_Angle_deg'].mean():.1f}°",
            f"{shot_data['CoG_Horizontal_Movement_px'].mean():.1f}",
            f"{shot_data['CoG_Vertical_Movement_px'].mean():.1f}"
        ])
    
    columns = ['Shot Type', 'Count', 'Avg Max\nVel', 'Avg\nVel', 'Avg\nDur', 
              'Avg Hip-\nKnee', 'Avg\nElbow', 'Avg\nTorso',
              'Avg H.\nMove', 'Avg V.\nMove']
    
    table = ax.table(cellText=summary_data, colLabels=columns, 
                    cellLoc='center', loc='center',
                    colWidths=[0.11, 0.08, 0.11, 0.11, 0.09, 0.11, 0.09, 0.09, 0.11, 0.11])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    colors = ['#FFE5E5', '#E5F5F5', '#E5E5FF']
    for i, row in enumerate(summary_data):
        for j in range(len(columns)):
            table[(i+1, j)].set_facecolor(colors[i % len(colors)])
    
    plt.title('Summary Statistics by Shot Type', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / '12_summary_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: Summary Statistics Table")

def plot_scatter_matrix(df, output_dir):
    """13. Advanced Scatter Matrix"""
    plt.figure(figsize=(16, 16))
    
    metrics = ['Max_Racket_Velocity_px_s', 'Duration_Seconds', 
              'Avg_Elbow_Angle_deg', 'Avg_Torso_Angle_deg']
    
    shot_type_colors = {st: f'C{i}' for i, st in enumerate(df['Shot_Type'].unique())}
    colors = df['Shot_Type'].map(shot_type_colors)
    
    pd.plotting.scatter_matrix(df[metrics], figsize=(16, 16), 
                              alpha=0.6, diagonal='kde', c=colors, s=50)
    
    plt.suptitle('Scatter Matrix of Key Performance Metrics', 
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / '13_scatter_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: Scatter Matrix")

def plot_shot_efficiency(df, output_dir):
    """14. Shot Efficiency Analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Velocity efficiency (velocity/duration)
    df['Velocity_Efficiency'] = df['Max_Racket_Velocity_px_s'] / (df['Duration_Seconds'] + 0.001)
    
    for shot_type in df['Shot_Type'].unique():
        data = df[df['Shot_Type'] == shot_type]['Velocity_Efficiency']
        axes[0, 0].hist(data, alpha=0.6, label=shot_type, bins=15, edgecolor='black')
    axes[0, 0].set_title('Velocity Efficiency (Velocity/Duration)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Efficiency (px/s²)', fontsize=10)
    axes[0, 0].set_ylabel('Frequency', fontsize=10)
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(alpha=0.3)
    
    # Total movement
    df['Movement_Total'] = np.sqrt(df['CoG_Horizontal_Movement_px']**2 + df['CoG_Vertical_Movement_px']**2)
    
    shot_types = df['Shot_Type'].unique()
    movement_means = [df[df['Shot_Type'] == st]['Movement_Total'].mean() for st in shot_types]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    axes[0, 1].bar(shot_types, movement_means, color=colors[:len(shot_types)], alpha=0.7)
    axes[0, 1].set_title('Total CoG Movement by Shot Type', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Shot Type', fontsize=10)
    axes[0, 1].set_ylabel('Total Movement (px)', fontsize=10)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Velocity vs Elbow Angle
    for shot_type in df['Shot_Type'].unique():
        data = df[df['Shot_Type'] == shot_type]
        axes[1, 0].scatter(data['Avg_Elbow_Angle_deg'], data['Max_Racket_Velocity_px_s'],
                         label=shot_type, alpha=0.6, s=50)
    axes[1, 0].set_title('Velocity vs Elbow Angle', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Avg Elbow Angle (degrees)', fontsize=10)
    axes[1, 0].set_ylabel('Max Velocity (px/s)', fontsize=10)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(alpha=0.3)
    
    # Frame duration analysis
    df['Frame_Duration'] = df['End_Frame'] - df['Start_Frame']
    for shot_type in shot_types:
        data = df[df['Shot_Type'] == shot_type]['Frame_Duration']
        axes[1, 1].hist(data, alpha=0.6, label=shot_type, bins=15, edgecolor='black')
    axes[1, 1].set_title('Frame Duration Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Frame Duration', fontsize=10)
    axes[1, 1].set_ylabel('Frequency', fontsize=10)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle('Shot Efficiency Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / '14_shot_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: Shot Efficiency Analysis")

def plot_comprehensive_overview(df, output_dir):
    """15. Comprehensive Overview Dashboard"""
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Shot type pie
    ax1 = fig.add_subplot(gs[0, 0])
    shot_counts = df['Shot_Type'].value_counts()
    ax1.pie(shot_counts.values, labels=shot_counts.index, autopct='%1.1f%%', 
            startangle=90, colors=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_title('Shot Distribution', fontsize=12, fontweight='bold')
    
    # 2. Velocity box
    ax2 = fig.add_subplot(gs[0, 1])
    data = [df[df['Shot_Type'] == st]['Max_Racket_Velocity_px_s'].values 
            for st in df['Shot_Type'].unique()]
    bp = ax2.boxplot(data, labels=df['Shot_Type'].unique(), patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#FF6B6B', '#4ECDC4']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_title('Max Velocity', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', labelsize=8)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Duration histogram
    ax3 = fig.add_subplot(gs[0, 2])
    for st in df['Shot_Type'].unique():
        ax3.hist(df[df['Shot_Type'] == st]['Duration_Seconds'], alpha=0.6, bins=15)
    ax3.set_title('Duration Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Duration (s)', fontsize=10)
    ax3.grid(alpha=0.3)
    
    # 4. Velocity over time
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.plot(df['Shot_Number'], df['Max_Racket_Velocity_px_s'], 
            marker='o', linestyle='-', markersize=3, alpha=0.6)
    ax4.set_title('Velocity Over Time', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Shot Number', fontsize=10)
    ax4.set_ylabel('Max Velocity', fontsize=10)
    ax4.grid(alpha=0.3)
    
    # 5. Angle comparison
    ax5 = fig.add_subplot(gs[1, 2])
    angle_means = {
        'Hip-Knee': df['Avg_Hip_Knee_Angle_deg'].mean(),
        'Elbow': df['Avg_Elbow_Angle_deg'].mean(),
        'Torso': df['Avg_Torso_Angle_deg'].mean()
    }
    ax5.bar(angle_means.keys(), angle_means.values(), color=['#FF6B6B', '#4ECDC4', '#95E1D3'])
    ax5.set_title('Average Angles', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Angle (degrees)', fontsize=10)
    ax5.tick_params(axis='x', labelsize=8, rotation=45)
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Movement scatter
    ax6 = fig.add_subplot(gs[2, 0])
    for st in df['Shot_Type'].unique():
        data = df[df['Shot_Type'] == st]
        ax6.scatter(data['CoG_Horizontal_Movement_px'], data['CoG_Vertical_Movement_px'],
                   label=st, alpha=0.6, s=30)
    ax6.set_title('CoG Movement', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Horizontal', fontsize=9)
    ax6.set_ylabel('Vertical', fontsize=9)
    ax6.legend(fontsize=8)
    ax6.grid(alpha=0.3)
    
    # 7. Statistics table
    ax7 = fig.add_subplot(gs[2, 1:])
    ax7.axis('tight')
    ax7.axis('off')
    
    table_data = []
    for st in df['Shot_Type'].unique():
        data = df[df['Shot_Type'] == st]
        table_data.append([
            st,
            len(data),
            f"{data['Max_Racket_Velocity_px_s'].mean():.1f}",
            f"{data['Duration_Seconds'].mean():.2f}",
            f"{data['Avg_Elbow_Angle_deg'].mean():.1f}°"
        ])
    
    table = ax7.table(cellText=table_data,
                      colLabels=['Type', 'Count', 'Avg Vel', 'Avg Dur', 'Avg Elbow'],
                      cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    plt.suptitle('BADMINTON SHOT ANALYSIS - COMPREHENSIVE OVERVIEW', 
                fontsize=18, fontweight='bold')
    plt.savefig(output_dir / '15_comprehensive_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: Comprehensive Overview Dashboard")

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("  BADMINTON SHOT ANALYSIS - DATATHON VISUALIZATION")
    print("="*60 + "\n")
    
    try:
        df = load_latest_csv()
        print(f"📈 Loaded {len(df)} shots from CSV\n")
        
        output_dir = create_output_dir()
        print(f"📁 Output directory: {output_dir}\n")
        
        print("🎨 Generating visualizations...\n")
        
        plot_shot_type_distribution(df, output_dir)
        plot_velocity_comparison(df, output_dir)
        plot_duration_analysis(df, output_dir)
        plot_angle_correlation_heatmap(df, output_dir)
        plot_velocity_vs_duration(df, output_dir)
        plot_player_performance(df, output_dir)
        plot_movement_analysis(df, output_dir)
        plot_angle_distributions(df, output_dir)
        plot_temporal_analysis(df, output_dir)
        plot_performance_radar(df, output_dir)
        plot_velocity_kde(df, output_dir)
        plot_summary_statistics(df, output_dir)
        plot_scatter_matrix(df, output_dir)
        plot_shot_efficiency(df, output_dir)
        plot_comprehensive_overview(df, output_dir)
        
        print("\n" + "="*60)
        print(f"✅ Successfully generated 15 diagrams!")
        print(f"📊 Location: {output_dir}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
