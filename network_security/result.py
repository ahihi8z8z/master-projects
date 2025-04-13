import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
    
parent_dir = "ns/results"  

data = []

pattern = re.compile(r'(?P<model>.+)_(?P<lr>[\d\.e-]+)_pca_(?P<pca>\d+)_mca_(?P<mca>\d+)')

for folder in os.listdir(parent_dir):
    folder_path = os.path.join(parent_dir, folder)
    if os.path.isdir(folder_path):
        match = pattern.match(folder)
        if match and (match.group('model') == 'simpleANN2' or match.group('model') == 'simpleANN'):
            try:
                info = match.groupdict()
                info['lr'] = float(info['lr'])
                info['pca'] = int(info['pca'])
                info['mca'] = int(info['mca'])

                metrics_path = os.path.join(folder_path, "metrics.txt")
                with open(metrics_path, "r") as f:
                    for line in f:
                        if ':' in line:
                            key, value = line.strip().split(":")
                            if key != "Model":
                                info[key.strip()] = float(value.strip())

                history_path = os.path.join(folder_path, "training_history.csv")
                history_df = pd.read_csv(history_path)
                stop_epoch = len(history_df)
                info['stop_epoch'] = stop_epoch

                data.append(info)

            except Exception as e:
                print(f"Lỗi khi xử lý folder {folder}: {e}")

df = pd.DataFrame(data)
df = df.round(4)
df.to_csv('results/all_metrics_summary.csv', index=False)

models = df['model'].unique()  
model_dfs = {model: df[df['model'] == model].reset_index(drop=True) for model in models}

for model_name, sub_df in model_dfs.items():
    print(f"\nModel: {model_name}")
    print(sub_df)

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "stop_epoch"]
    lr_values = df['lr'].unique().astype(float)

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=sub_df, x="lr", y=metric, hue="model", marker='o')
        plt.xlabel("Learning Rate")
        plt.ylabel(metric)
        plt.grid(True)
        
        plt.xticks(lr_values, rotation=45 )
        plt.tight_layout()
        plt.savefig(f"results/{model_name}_{metric}_lr.png")