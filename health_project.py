# ====================== 1. 导入工具包 ======================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE  # 解决数据不平衡

# 解决中文显示（可选）
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# ====================== 2. 生成更平衡的模拟糖尿病患者数据 ======================
np.random.seed(42)
n_patients = 1000

# 生成特征：年龄、BMI、血糖、血压、用药情况、并发症
age = np.random.randint(40, 90, size=n_patients)
bmi = np.random.normal(28, 5, size=n_patients)
glucose = np.random.normal(160, 40, size=n_patients)
bp = np.random.normal(130, 20, size=n_patients)
medication = np.random.randint(0, 2, size=n_patients)
complications = np.random.randint(0, 3, size=n_patients)

# 调整再入院概率，让正负样本更平衡
readmission_prob = 1 / (1 + np.exp(-(0.02*age + 0.05*bmi + 0.01*glucose + 0.01*bp + 0.5*complications - 8)))
readmission = np.random.binomial(1, readmission_prob, size=n_patients)

# 构建 DataFrame
df = pd.DataFrame({
    "age": age,
    "bmi": bmi,
    "glucose": glucose,
    "blood_pressure": bp,
    "on_medication": medication,
    "complications": complications,
    "readmission": readmission
})

print("===== 数据概况 =====")
print(f"患者数量: {len(df)}")
print(f"特征数量: {len(df.columns) - 1}")
print(f"再入院患者数量: {df['readmission'].sum()}")
print(f"再入院率: {df['readmission'].mean():.1%}")
print("\n===== 前5行数据 =====")
print(df.head())

# ====================== 3. 数据预处理 & 解决不平衡 ======================
# 处理缺失值（这里模拟数据无缺失，保留流程）
df = df.dropna()

# 特征与标签分离
X = df.drop("readmission", axis=1)
y = df["readmission"]

# 划分训练集/测试集（分层抽样，保证测试集也有足够的再入院样本）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=666, stratify=y
)

# 对训练集使用SMOTE过采样，解决类别不平衡
smote = SMOTE(random_state=666)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# ====================== 4. 构建预测模型 ======================
model = RandomForestClassifier(random_state=666)
model.fit(X_train_balanced, y_train_balanced)

# 预测 & 评分
y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)
auc = roc_auc_score(y_test, y_pred_prob)

print("\n==================================")
print(f" 模型最终 AUC 得分: {auc:.2f} ✅")
print("==================================")

# ====================== 5. 输出结果报告 ======================
print("\n===== 分类报告 =====")
print(classification_report(y_test, y_pred))

# ====================== 6. 关键因素可视化 ======================
importances = model.feature_importances_
features = X.columns
top_idx = np.argsort(importances)[-6:]  # 取全部特征

plt.figure(figsize=(10, 5))
plt.barh(features[top_idx], importances[top_idx], color="#549bd6")
plt.xlabel("特征重要性")
plt.title("影响糖尿病患者再入院的关键临床指标")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)  # 保存图片
plt.show()

# ====================== 7. 保存结果 ======================
results = pd.DataFrame({
    "真实是否再入院": y_test,
    "模型预测风险值": y_pred_prob
})
results.to_csv("糖尿病患者再入院预测结果.csv", index=False, encoding="utf-8-sig")
print("\n✅ 预测结果已保存 → 糖尿病患者再入院预测结果.csv")
print("✅ 特征重要性图已保存 → feature_importance.png")