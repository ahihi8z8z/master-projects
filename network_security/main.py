# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from prince import MCA
from sklearn.decomposition import PCA
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf


def data_handle():
    attack_dir = r"SWaT_Dataset_Attack_v0.csv"
    normal_dir = r"SWaT_Dataset_Normal_v0.csv"

    # Đọc 2 file excel
    df1 = pd.read_csv(normal_dir, skipinitialspace=True)
    df2 = pd.read_csv(attack_dir, skipinitialspace=True)

    # Gộp lại theo chiều dọc (nối thêm dòng)
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.drop('Timestamp', axis=1)

    # Dropna
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna()

    # Encode label
    df['label'] = df['Normal/Attack'].replace({'Normal': 0, 'Attack': 1, 'A ttack': 1})
    df.drop('Normal/Attack', axis=1, inplace=True)

    # Scale
    float_cols = df.drop(columns=['label']).select_dtypes(include='float').columns
    scaler = StandardScaler()
    df[float_cols] = scaler.fit_transform(df[float_cols])

    #OneHotEncoder
    int_cols = df.drop(columns=['label']).select_dtypes(include='int').columns
    encoder = OneHotEncoder(sparse_output=False, drop='first',dtype=int)
    encoded_array = encoder.fit_transform(df[int_cols])
    encoded_cols = encoder.get_feature_names_out(int_cols)
    df_encoded = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)
    df = pd.concat([df.drop(columns=int_cols), df_encoded], axis=1)


    # Cân bằng nhãn
    df_0 = df[df['label'] == 0]
    df_1 = df[df['label'] == 1]
    min_len = min(len(df_0), len(df_1))
    df_0_balanced = df_0.sample(min_len, random_state=42)
    df_1_balanced = df_1.sample(min_len, random_state=42)
    df_balanced = pd.concat([df_0_balanced, df_1_balanced]).sample(frac=1, random_state=42)
    x = df_balanced.drop('label', axis=1)
    y = df_balanced['label']

    # Chia dữ liệu thành train và test
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    # Thực hiện PCA
    numerical_cols = X_train.select_dtypes(include=['float']).columns
    df_train_num = X_train[numerical_cols].copy()
    df_test_num = X_test[numerical_cols].copy()
    pca_full = PCA()
    pca_full.fit(df_train_num)

    explained_variance = pca_full.explained_variance_ratio_ * 100
    cum_variance = explained_variance.cumsum()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_variance)+1), cum_variance, marker='o')
    plt.xlabel('Số lượng thành phần chính (Principal Components)')
    plt.ylabel('Tổng % thông tin được giữ lại')
    plt.title('Biểu đồ thông tin được giữ lại bởi PCA')
    plt.grid(True)
    plt.xticks(range(1, len(explained_variance)+1))
    plt.yticks(range(0, 101, 10))
    plt.axhline(y=95, color='r', linestyle='--', label='Ngưỡng 95%')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/explained_variance.png')

    # Áp dụng MCA
    categorical_cols = X_train.select_dtypes(include=['int']).columns

    df_train_cat = X_train[categorical_cols].astype('category')
    df_test_cat = X_test[categorical_cols].astype('category')

    mca = MCA(n_components=len(df_train_cat.columns))
    mca = mca.fit(df_train_cat)

    explained_inertia = np.array(mca.explained_inertia_) * 100
    cumulative_inertia = np.cumsum(explained_inertia)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_inertia)+1), cumulative_inertia, marker='o')
    plt.xlabel('Số lượng thành phần chính (MCA Components)')
    plt.ylabel('Tổng % thông tin được giữ lại')
    plt.title('Biểu đồ thông tin được giữ lại bởi MCA')
    plt.grid(True)
    plt.xticks(range(1, len(cumulative_inertia)+1))
    plt.yticks(range(0, 101, 10))
    plt.axhline(y=95, color='r', linestyle='--', label='Ngưỡng 95%')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/mca_explained_inertia.png')
    plt.show()

    X_train_pca = pca_full.transform(df_train_num)
    X_train_mca = mca.transform(df_train_cat).to_numpy()

    X_test_pca = pca_full.transform(df_test_num)
    X_test_mca = mca.transform(df_test_cat).to_numpy()
    
    return X_train, X_test, y_train, y_test, X_train_pca, X_test_pca, X_train_mca, X_test_mca, scaler, pca_full, mca
     

# Linear Model
def simpleANN(input_shape):
  model = Sequential()
  model.add(Dense(32, kernel_initializer='random_normal', input_dim=input_shape)) 
  model.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal')) 
  return model

# Thêm 01 hidden layer phi tuyến
def simpleANN2(input_shape):
  model = Sequential()
  model.add(Dense(32, kernel_initializer='random_normal', input_dim=input_shape)) 
  model.add(Dense(16, activation='relu', kernel_initializer='random_normal')) 
  model.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal')) 
  return model

# Thêm 01 hidden layer phi tuyến
def simpleANN3(input_shape):
    model = Sequential()
    model.add(Dense(64, kernel_initializer='random_normal', input_dim=input_shape)) 
    model.add(Dense(32, activation='relu', kernel_initializer='random_normal')) 
    model.add(Dense(16, activation='relu', kernel_initializer='random_normal')) 
    model.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal')) 
    return model

def do_run(model_name, learning_rate, tag, X_train, y_train, X_test, y_test):
    folder_name = f"results/{model_name}_{learning_rate}_{tag}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    if model_name == 'simpleANN':
        model = simpleANN(X_train.shape[1])
    elif model_name == 'simpleANN2':
        model = simpleANN2(X_train.shape[1])
    elif model_name == 'simpleANN3':
        model = simpleANN3(X_train.shape[1])
    else:
        raise ValueError("Model not supported")

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    
    with open(os.path.join(folder_name, "model_summary.txt"), "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    with open(os.path.join(folder_name, "model_config.json"), "w") as json_file:
        json_file.write(model.to_json())

    callback = tf.keras.callbacks.EarlyStopping(
        monitor='loss',      
        patience=5,              
        restore_best_weights=True, 
        verbose=1
    )
    # Train the model
    history = model.fit(X_train, y_train, epochs = 2000, batch_size = 1024, validation_split=0.2, callbacks=[callback])
    
    model.save(os.path.join(folder_name, "model.keras"))

    # Đánh giá mô hình
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    # In kết quả
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    # Lưu kết quả vào file text
    with open(os.path.join(folder_name, "metrics.txt"), "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")

    # Vẽ đồ thị về accuracy và loss
    plt.figure(figsize=(12, 6))
    
    # Đồ thị accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Đồ thị loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Lưu đồ thị vào thư mục
    plt.savefig(os.path.join(folder_name, "accuracy_loss.png"))
    plt.close()

    # Vẽ đồ thị ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    
    # Lưu đồ thị ROC vào thư mục
    plt.savefig(os.path.join(folder_name, "roc_curve.png"))
    plt.close()

    # Lưu lịch sử training (loss, accuracy)
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(folder_name, 'training_history.csv'), index=False)

    print(f"All results and plots saved in {folder_name}")
    return accuracy, precision, recall, f1, roc_auc

def plot_metrics(df, x_col, y_col, metric_name, filename):
    plt.figure(figsize=(8, 6))
    for model in df['model'].unique():
        sub_df = df[df['model'] == model]
        plt.plot(sub_df[x_col], sub_df[y_col], marker='o', label=model)
    plt.title(f'{metric_name} vs {x_col}')
    plt.xlabel(x_col)
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'results/{filename}')
    plt.close()

# === 1. Xử lý dữ liệu ===
X_train, X_test, y_train, y_test, X_train_pca, X_test_pca, X_train_mca, X_test_mca, scaler, pca_full, mca = data_handle()

# === 2. Chuẩn bị các list ===
pca_dim_range = list(range(5, 17)) # Lấy trong dải 90% - 99% 
mca_dim_range = list(range(8, 16)) # Lấy trong dải 90% - 99% 
lr_range = np.logspace(-4, -2, num=5).tolist() # Lấy trong dải 0.0001 - 0.01

model_list = ['simpleANN', 'simpleANN2', 'simpleANN3']
results = []

# === 3. Chạy train ===
for model_name in model_list:
    if model_name == 'simpleANN3':
        continue
    for lr in lr_range:
        # Gốc
        tag = "origin"
        accuracy, precision, recall, f1, roc_auc = do_run(model_name, lr, tag, X_train, y_train, X_test, y_test)
        results.append({
            "model": model_name, "lr": lr, "pca_dim": 0, "mca_dim": 0,
            "tag": tag, "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1, "roc_auc": roc_auc
        })

        # Các tổ hợp PCA/MCA
        for pca_dim in pca_dim_range:
            for mca_dim in mca_dim_range:
                tag = f"pca_{pca_dim}_mca_{mca_dim}"
                X_train_combined = np.hstack((X_train_pca[:, :pca_dim], X_train_mca[:, :mca_dim]))
                X_test_combined = np.hstack((X_test_pca[:, :pca_dim], X_test_mca[:, :mca_dim]))
                acc = do_run(model_name, lr, tag, X_train_combined, y_train, X_test_combined, y_test)
                results.append({
                    "model": model_name, "lr": lr, "pca_dim": 0, "mca_dim": 0,
                    "tag": tag, "accuracy": accuracy, "precision": precision,
                    "recall": recall, "f1": f1, "roc_auc": roc_auc
                })

