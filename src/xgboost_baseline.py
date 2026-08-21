import pandas as pd
import numpy as np
import networkx as nx
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


file_path = '/Users/sarthakmohindru/Downloads/HI-Small_Trans.csv'
df = pd.read_csv(file_path)

# Ensure datetime and sequential sorting for feature engineering
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values(['Account', 'Timestamp'])

# ==========================================
# 2. GRAPH / NETWORK FEATURES (GFs)
# ==========================================

G = nx.from_pandas_edgelist(
    df, source='Account', target='Account.1', create_using=nx.DiGraph()
)


pagerank_scores = nx.pagerank(G, alpha=0.85)
hubs, authorities = nx.hits(G, max_iter=100, normalized=True)
in_degree = nx.in_degree_centrality(G)
out_degree = nx.out_degree_centrality(G)


# Map to Sender
df['Sender_PageRank'] = df['Account'].map(pagerank_scores).fillna(0)
df['Sender_Hub_Score'] = df['Account'].map(hubs).fillna(0)
df['Sender_Out_Degree'] = df['Account'].map(out_degree).fillna(0)
# Map to Receiver
df['Receiver_PageRank'] = df['Account.1'].map(pagerank_scores).fillna(0)
df['Receiver_Authority_Score'] = df['Account.1'].map(authorities).fillna(0)
df['Receiver_In_Degree'] = df['Account.1'].map(in_degree).fillna(0)

# ==========================================
# 3. EATURE ENGINEERING

# --- A. Velocity Features ---
df['Time_Since_Last_Txn'] = df.groupby('Account')['Timestamp'].diff().dt.total_seconds().fillna(0)
r = df.set_index('Timestamp').groupby('Account')['Amount Received']
df['Count_1h'] = r.rolling('1h').count().values
df['Sum_1h']   = r.rolling('1h').sum().values
df['Count_24h'] = r.rolling('24h').count().values
df['Sum_24h']   = r.rolling('24h').sum().values

# --- B. Basic Topology Features ---
df['Unique_Receivers_Lifetime'] = df.groupby('Account')['Account.1'].transform('nunique')
df['Day_Date'] = df['Timestamp'].dt.date
df['Daily_Unique_Receivers'] = df.groupby(['Account', 'Day_Date'])['Account.1'].transform('nunique')
fan_in_map = df.groupby('Account.1')['Account'].nunique()
df['Receiver_Fan_In'] = df['Account.1'].map(fan_in_map).fillna(0)

# --- C. Deviation Features ---
df['User_Avg_Amount'] = df.groupby('Account')['Amount Received'].transform('mean')
df['Amount_vs_Avg_Ratio'] = df['Amount Received'] / (df['User_Avg_Amount'] + 1)

# ==========================================
# 4. PREPROCESSING & CLEANUP
# ==========================================
features_to_drop = ['Timestamp', 'Account', 'Account.1', 'Day_Date', 'Amount Paid', 'Payment Currency']
final_df = df.drop(columns=[c for c in features_to_drop if c in df.columns])

cat_cols = ['From Bank', 'To Bank', 'Receiving Currency', 'Payment Format']
le = LabelEncoder()
for col in cat_cols:
    if col in final_df.columns:
        final_df[col] = final_df[col].astype(str)
        final_df[col] = le.fit_transform(final_df[col])

# ==========================================
# 5. DATA SPLITTING & UNDERSAMPLING (TRAIN ONLY)
# ==========================================

target_col = 'Is Laundering'
X = final_df.drop(target_col, axis=1)
y = final_df[target_col]

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Undersample the training data to a 10:1 ratio
train_data = pd.concat([X_train, y_train], axis=1)
minority_train = train_data[train_data[target_col] == 1]
majority_train = train_data[train_data[target_col] == 0]

desired_normal_count = len(minority_train) * 10
majority_train_sampled = majority_train.sample(n=desired_normal_count, random_state=42)

train_data_balanced = pd.concat([minority_train, majority_train_sampled]).sample(frac=1, random_state=42)
X_train_balanced = train_data_balanced.drop(target_col, axis=1)
y_train_balanced = train_data_balanced[target_col]

print(f"   -> Training Class Ratio: 10 Normal to 1 Fraud")
print(f"   -> Test Set Shape (Real World): {X_test.shape}")

# ==========================================
# 6. TRAIN XGBOOST
# ==========================================

model = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=1,           
    n_estimators=150,              
    max_depth=12,                   
    learning_rate=0.04,            
    eval_metric='aucpr',           
    n_jobs=-1,                     
    random_state=42
)
model.fit(X_train_balanced, y_train_balanced)



y_prob = model.predict_proba(X_test)[:, 1] 

# --- Calculate AUC Scores ---
roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

# Calculate F1 across all thresholds to find the mathematical peak
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"\n--- Final AUC & Optimization Results ---")
print(f"ROC-AUC Score: {roc_auc:.4f} (Measures overall class separation)")
print(f"PR-AUC Score:  {pr_auc:.4f} (The most critical metric for imbalanced data!)")
print(f"Maximized F1 Score: {best_f1:.4f}")
print(f"Optimal Probability Threshold: {best_threshold:.4f}")

# --- Plot the ROC and PR Curves ---
fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: ROC Curve
axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[0].set_xlim([0.0, 1.0])
axes[0].set_ylim([0.0, 1.05])
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate (Recall)')
axes[0].set_title('Receiver Operating Characteristic (ROC)')
axes[0].legend(loc="lower right")
axes[0].grid(True)

# Plot 2: Precision-Recall Curve
axes[1].plot(recalls, precisions, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
axes[1].plot([0, 1], [0.001, 0.001], color='navy', lw=2, linestyle='--', label='Random Guess')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('Recall (True Positive Rate)')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve')
axes[1].legend(loc="upper right")
axes[1].grid(True)

plt.tight_layout()
plt.show()

# --- Print Final Report ---
y_pred_optimal = (y_prob >= best_threshold).astype(int)
print(f"\nClassification Report (Threshold = {best_threshold:.4f}):")
print(classification_report(y_test, y_pred_optimal, target_names=['Normal', 'Fraud']))

# Final Confusion Matrix
cm_optimal = confusion_matrix(y_test, y_pred_optimal)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Normal', 'Fraud'], 
            yticklabels=['Normal', 'Fraud'])
plt.title(f'Final Confusion Matrix (F1: {best_f1:.4f})')
plt.ylabel('Actual Truth')
plt.xlabel('Model Prediction')
plt.show()

# ==========================================
# 8. FEATURE IMPORTANCE PLOT
# ==========================================
importance_type = 'gain'
importances = model.get_booster().get_score(importance_type=importance_type)
sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15] # Top 15

plt.figure(figsize=(10, 6))
plt.barh([x[0] for x in sorted_importances][::-1], [x[1] for x in sorted_importances][::-1], color='teal')
plt.xlabel(f'F-Score ({importance_type})')
plt.title('Top 15 XGBoost Features')
plt.tight_layout()
plt.show()