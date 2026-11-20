import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set aesthetic styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['figure.titlesize'] = 14

COVER_TYPE_NAMES = {
    1: 'Spruce/Fir',
    2: 'Lodgepole Pine',
    3: 'Ponderosa Pine',
    4: 'Cottonwood/Willow',
    5: 'Aspen',
    6: 'Douglas-fir',
    7: 'Krummholz'
}

def run_eda_and_quality():
    print("Loading train and test datasets...")
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    # 1. Dataset Shape & Dtypes
    n_train_rows, n_train_cols = train_df.shape
    n_test_rows, n_test_cols = test_df.shape
    
    missing_train = train_df.isnull().sum().sum()
    missing_test = test_df.isnull().sum().sum()
    
    dup_train = train_df.duplicated(subset=['Id']).sum()
    dup_test = test_df.duplicated(subset=['Id']).sum()
    dup_train_features = train_df.drop(columns=['Id', 'Cover_Type']).duplicated().sum()
    
    # Check ID uniqueness
    train_id_unique = train_df['Id'].nunique() == len(train_df)
    test_id_unique = test_df['Id'].nunique() == len(test_df)
    id_overlap = len(set(train_df['Id']).intersection(set(test_df['Id'])))
    
    # Wilderness & Soil columns
    wilderness_cols = [c for c in train_df.columns if c.startswith('Wilderness_Area')]
    soil_cols = [c for c in train_df.columns if c.startswith('Soil_Type')]
    continuous_cols = [
        'Elevation', 'Aspect', 'Slope',
        'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
        'Horizontal_Distance_To_Roadways', 'Hillshade_9am',
        'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points'
    ]
    
    # One-hot integrity check
    train_df['wilderness_sum'] = train_df[wilderness_cols].sum(axis=1)
    train_df['soil_sum'] = train_df[soil_cols].sum(axis=1)
    test_df['wilderness_sum'] = test_df[wilderness_cols].sum(axis=1)
    test_df['soil_sum'] = test_df[soil_cols].sum(axis=1)
    
    train_wild_exact_one = (train_df['wilderness_sum'] == 1).all()
    train_soil_exact_one = (train_df['soil_sum'] == 1).all()
    test_wild_exact_one = (test_df['wilderness_sum'] == 1).all()
    test_soil_exact_one = (test_df['soil_sum'] == 1).all()
    
    # Constant and zero variance features
    constant_train = [c for c in train_df.columns if train_df[c].nunique() == 1]
    constant_test = [c for c in test_df.columns if test_df[c].nunique() == 1]
    
    print(f"Constant columns in train: {constant_train}")
    print(f"Constant columns in test: {constant_test}")
    
    # Ranges check
    range_summary = {}
    for col in continuous_cols:
        range_summary[col] = {
            'train_min': float(train_df[col].min()),
            'train_max': float(train_df[col].max()),
            'train_mean': float(train_df[col].mean()),
            'train_std': float(train_df[col].std()),
            'test_min': float(test_df[col].min()),
            'test_max': float(test_df[col].max()),
            'test_mean': float(test_df[col].mean()),
            'test_std': float(test_df[col].std())
        }
        
    # Class distribution in train
    class_counts = train_df['Cover_Type'].value_counts().sort_index().to_dict()
    
    quality_metrics = {
        'n_train_rows': n_train_rows,
        'n_train_cols': n_train_cols,
        'n_test_rows': n_test_rows,
        'n_test_cols': n_test_cols,
        'missing_train': int(missing_train),
        'missing_test': int(missing_test),
        'dup_train_ids': int(dup_train),
        'dup_test_ids': int(dup_test),
        'dup_train_feature_rows': int(dup_train_features),
        'train_id_unique': bool(train_id_unique),
        'test_id_unique': bool(test_id_unique),
        'id_overlap': int(id_overlap),
        'train_wild_exact_one': bool(train_wild_exact_one),
        'train_soil_exact_one': bool(train_soil_exact_one),
        'test_wild_exact_one': bool(test_wild_exact_one),
        'test_soil_exact_one': bool(test_soil_exact_one),
        'constant_train_cols': constant_train,
        'constant_test_cols': constant_test,
        'class_counts': class_counts,
        'continuous_ranges': range_summary
    }
    
    with open('output/quality_metrics.json', 'w') as f:
        json.dump(quality_metrics, f, indent=2)
    print("Saved output/quality_metrics.json")
    
    # ----------------- Visualizations -----------------
    print("Generating EDA figures...")
    
    # 1. Target Class Distribution
    plt.figure(figsize=(9, 5))
    palette = sns.color_palette("Set2", 7)
    counts = [class_counts[i] for i in range(1, 8)]
    labels = [f"Class {i}\n({COVER_TYPE_NAMES[i]})" for i in range(1, 8)]
    bars = plt.bar(labels, counts, color=palette, edgecolor='black', linewidth=1)
    plt.title('Distribution of Forest Cover Types in Training Data (N = 15,120)', pad=12, fontweight='bold')
    plt.ylabel('Observation Count')
    plt.xlabel('Forest Cover Type')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 30, f"{int(yval):,}\n(14.3%)", ha='center', va='bottom', fontsize=9)
    plt.ylim(0, 2600)
    plt.tight_layout()
    plt.savefig('figures/fig1_class_distribution.png', dpi=300)
    plt.close()
    
    # 2. Elevation by Cover Type
    plt.figure(figsize=(10, 6))
    train_df['Cover_Name'] = train_df['Cover_Type'].map(COVER_TYPE_NAMES)
    order = [COVER_TYPE_NAMES[i] for i in range(1, 8)]
    sns.boxplot(x='Cover_Name', y='Elevation', data=train_df, order=order, palette='Set2')
    plt.title('Elevation Distribution Across Forest Cover Types', pad=12, fontweight='bold')
    plt.xlabel('Forest Cover Type')
    plt.ylabel('Elevation (meters above sea level)')
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    plt.savefig('figures/fig2_elevation_by_class.png', dpi=300)
    plt.close()
    
    # 3. Slope and Aspect by Cover Type
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.boxplot(x='Cover_Name', y='Slope', data=train_df, order=order, palette='Set2', ax=axes[0])
    axes[0].set_title('Slope (degrees) by Cover Type', fontweight='bold')
    axes[0].set_xlabel('Cover Type')
    axes[0].set_ylabel('Slope (degrees)')
    axes[0].tick_params(axis='x', rotation=30)
    
    sns.boxplot(x='Cover_Name', y='Aspect', data=train_df, order=order, palette='Set2', ax=axes[1])
    axes[1].set_title('Aspect (azimuth degrees) by Cover Type', fontweight='bold')
    axes[1].set_xlabel('Cover Type')
    axes[1].set_ylabel('Aspect (azimuth 0-360°)')
    axes[1].tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig('figures/fig3_slope_aspect_by_class.png', dpi=300)
    plt.close()
    
    # 4. Distance Measurements by Cover Type
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.boxplot(x='Cover_Name', y='Horizontal_Distance_To_Hydrology', data=train_df, order=order, palette='Set2', ax=axes[0, 0])
    axes[0, 0].set_title('Horizontal Distance to Hydrology (m)', fontweight='bold')
    axes[0, 0].set_xlabel('')
    axes[0, 0].tick_params(axis='x', rotation=25)
    
    sns.boxplot(x='Cover_Name', y='Vertical_Distance_To_Hydrology', data=train_df, order=order, palette='Set2', ax=axes[0, 1])
    axes[0, 1].set_title('Vertical Distance to Hydrology (m)', fontweight='bold')
    axes[0, 1].set_xlabel('')
    axes[0, 1].tick_params(axis='x', rotation=25)
    
    sns.boxplot(x='Cover_Name', y='Horizontal_Distance_To_Roadways', data=train_df, order=order, palette='Set2', ax=axes[1, 0])
    axes[1, 0].set_title('Horizontal Distance to Roadways (m)', fontweight='bold')
    axes[1, 0].set_xlabel('')
    axes[1, 0].tick_params(axis='x', rotation=25)
    
    sns.boxplot(x='Cover_Name', y='Horizontal_Distance_To_Fire_Points', data=train_df, order=order, palette='Set2', ax=axes[1, 1])
    axes[1, 1].set_title('Horizontal Distance to Fire Points (m)', fontweight='bold')
    axes[1, 1].set_xlabel('')
    axes[1, 1].tick_params(axis='x', rotation=25)
    
    plt.tight_layout()
    plt.savefig('figures/fig4_distances_by_class.png', dpi=300)
    plt.close()
    
    # 5. Wilderness Area Distribution across Cover Types
    # Reconstruct categorical wilderness
    def get_wilderness(row):
        for i in range(1, 5):
            if row[f'Wilderness_Area{i}'] == 1:
                return f'Area {i}'
        return 'Unknown'
    
    train_df['Wilderness'] = train_df.apply(get_wilderness, axis=1)
    wild_ct = pd.crosstab(train_df['Cover_Name'], train_df['Wilderness'], normalize='index') * 100
    wild_ct = wild_ct.reindex(order)
    
    ax = wild_ct.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab10', edgecolor='black', linewidth=0.5)
    plt.title('Percentage of Wilderness Areas within Each Forest Cover Type', pad=12, fontweight='bold')
    plt.xlabel('Forest Cover Type')
    plt.ylabel('Percentage (%)')
    plt.legend(title='Wilderness Area', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    plt.savefig('figures/fig5_wilderness_by_class.png', dpi=300)
    plt.close()
    
    # 6. Soil Type Distribution across Cover Types (Top Soil Types per Class)
    def get_soil(row):
        for i in range(1, 41):
            if row[f'Soil_Type{i}'] == 1:
                return i
        return 0
    
    train_df['Soil_Code'] = train_df.apply(get_soil, axis=1)
    top_soils = train_df['Soil_Code'].value_counts().head(15).index.tolist()
    
    soil_ct = pd.crosstab(train_df['Soil_Code'], train_df['Cover_Name'])
    soil_ct_top = soil_ct.loc[top_soils][order]
    
    plt.figure(figsize=(11, 7))
    sns.heatmap(soil_ct_top, cmap='YlGnBu', annot=True, fmt='d', cbar_kws={'label': 'Observation Count'})
    plt.title('Top 15 Most Prevalent Soil Types by Cover Type', pad=12, fontweight='bold')
    plt.xlabel('Cover Type')
    plt.ylabel('Soil Type Index (1-40)')
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    plt.savefig('figures/fig6_top_soils_heatmap.png', dpi=300)
    plt.close()
    
    # 7. Correlation Matrix of Continuous Variables
    corr = train_df[continuous_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, square=True, linewidths=0.5)
    plt.title('Pearson Correlation Matrix of Continuous Terrain Predictors', pad=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/fig7_correlation_matrix.png', dpi=300)
    plt.close()
    
    # 8. 2D PCA Visualization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_df[continuous_cols])
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=train_df['Cover_Type'], cmap='tab10', alpha=0.6, s=15, edgecolors='none')
    plt.title(f'2D Principal Component Analysis (PCA) of Continuous Variables\nPC1 ({pca.explained_variance_ratio_[0]*100:.1f}%) & PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', pad=12, fontweight='bold')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)')
    cbar = plt.colorbar(scatter, ticks=range(1, 8))
    cbar.ax.set_yticklabels([f"{i}: {COVER_TYPE_NAMES[i]}" for i in range(1, 8)])
    cbar.set_label('Forest Cover Type')
    plt.tight_layout()
    plt.savefig('figures/fig8_pca_2d.png', dpi=300)
    plt.close()
    
    # 9. Train vs Test Continuous Predictor Distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    comp_cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points']
    for idx, col in enumerate(comp_cols):
        ax = axes[idx // 3, idx % 3]
        sns.kdeplot(train_df[col], label='Train (Balanced)', color='navy', lw=2, ax=ax)
        sns.kdeplot(test_df[col], label='Test (Unlabeled)', color='darkorange', lw=2, linestyle='--', ax=ax)
        ax.set_title(col, fontweight='bold')
        ax.legend()
    plt.suptitle('Comparison of Continuous Predictor Distributions: Train vs. Test', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('figures/fig9_train_test_distribution_shift.png', dpi=300)
    plt.close()

    print("EDA and Quality assessment completed successfully!")

if __name__ == '__main__':
    run_eda_and_quality()
