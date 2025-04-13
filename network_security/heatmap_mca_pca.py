import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('results/all_metrics_summary.csv') 

models = df['model'].unique()  
model_dfs = {model: df[df['model'] == model].reset_index(drop=True) for model in models}

for model_name, sub_df in model_dfs.items():
    if model_name == 'simpleANN' or model_name == 'simpleANN3':
        continue

    lr = 0.0032
    sub_df = sub_df[sub_df['lr'] == lr]

    pivot_df = sub_df.pivot_table(values='Precision', index='pca', columns='mca', aggfunc='mean').round(2)

    # Vẽ heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt=".2f")
    plt.gca().invert_yaxis()
    plt.title(f"{model_name} - Precision Heatmap (lr={lr})")
    plt.xlabel("MCA Dimension")
    plt.ylabel("PCA Dimension")
    plt.tight_layout()
    plt.savefig(f'results/heatmap_{model_name}_precision_lr_{lr}.png')
    plt.close()
    
    pivot_df = sub_df.pivot_table(values='Accuracy', index='pca', columns='mca', aggfunc='mean').round(2)

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt=".2f")
    plt.gca().invert_yaxis()
    plt.title(f"{model_name} - Accuracy Heatmap (lr={lr})")
    plt.xlabel("MCA Dimension")
    plt.ylabel("PCA Dimension")
    plt.tight_layout()
    plt.savefig(f'results/heatmap_{model_name}_accuracy_lr_{lr}.png')
    plt.close()
    
    pivot_df = sub_df.pivot_table(values='F1 Score', index='pca', columns='mca', aggfunc='mean').round(2)
    # Vẽ heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt=".2f")
    plt.gca().invert_yaxis()
    plt.title(f"{model_name} - F1 Score Heatmap (lr={lr})")
    plt.xlabel("MCA Dimension")
    plt.ylabel("PCA Dimension")
    plt.tight_layout()
    plt.savefig(f'results/heatmap_{model_name}_f1_lr_{lr}.png')
    plt.close()
    
    pivot_df = sub_df.pivot_table(values='Recall', index='pca', columns='mca', aggfunc='mean').round(2)
    # Vẽ heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt=".2f")
    plt.gca().invert_yaxis()
    plt.title(f"{model_name} - Recall Heatmap (lr={lr})")
    plt.xlabel("MCA Dimension")
    plt.ylabel("PCA Dimension")
    plt.tight_layout()
    plt.savefig(f'results/heatmap_{model_name}_recal_lr_{lr}.png')
    plt.close()
    
    pivot_df = sub_df.pivot_table(values='stop_epoch', index='pca', columns='mca', aggfunc='mean').round(2)
    # Vẽ heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt=".2f")
    plt.gca().invert_yaxis()
    plt.title(f"{model_name} - Stop epoch Heatmap (lr={lr})")
    plt.xlabel("MCA Dimension")
    plt.ylabel("PCA Dimension")
    plt.tight_layout()
    plt.savefig(f'results/heatmap_{model_name}_epoch_lr_{lr}.png')
    plt.close()