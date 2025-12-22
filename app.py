import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ================== CẤU HÌNH TRANG ==================
st.set_page_config(
    page_title="Phân loại bệnh tiểu đường",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 PHÂN LOẠI BỆNH TIỂU ĐƯỜNG BẰNG CÂY QUYẾT ĐỊNH")
st.caption("Mô hình Decision Tree – dữ liệu giả lập Pima Diabetes")
st.markdown("---")

# ================== TẠO DỮ LIỆU ==================
st.header("1️⃣ Tạo dữ liệu giả lập")

np.random.seed(1)
n_samples = 768

pima = pd.DataFrame({
    "pregnant": np.random.randint(0, 15, n_samples),
    "glucose": np.random.randint(70, 200, n_samples),
    "bp": np.random.randint(40, 120, n_samples),
    "skin": np.random.randint(0, 100, n_samples),
    "insulin": np.random.randint(0, 300, n_samples),
    "bmi": np.round(np.random.uniform(18, 50, n_samples), 1),
    "pedigree": np.round(np.random.uniform(0.1, 2.5, n_samples), 3),
    "age": np.random.randint(21, 70, n_samples)
})

label = []
for i in range(n_samples):
    risk = 0
    if pima.loc[i, "glucose"] > 140: risk += 1
    if pima.loc[i, "bmi"] > 30: risk += 1
    if pima.loc[i, "age"] > 45: risk += 1
    label.append(
        1 if (risk >= 2 and np.random.rand() > 0.25)
        else 1 if (risk < 2 and np.random.rand() > 0.8)
        else 0
    )

pima["label"] = label

col1, col2 = st.columns(2)
col1.metric("Số mẫu", pima.shape[0])
col2.metric("Số thuộc tính", pima.shape[1] - 1)

with st.expander("📊 Xem 5 dòng dữ liệu đầu tiên"):
    st.dataframe(pima.head(), use_container_width=True)

# ================== PHÂN BỐ NHÃN ==================
st.header("2️⃣ Phân bố nhãn")

label_counts = pima["label"].value_counts()
st.dataframe(label_counts.to_frame("Số lượng"))

fig1, ax1 = plt.subplots()
label_counts.plot(
    kind="bar",
    xlabel="Nhãn (0: Không bệnh, 1: Bị bệnh)",
    ylabel="Số mẫu",
    legend=False,
    ax=ax1
)
st.pyplot(fig1)

# ================== CHỌN THUỘC TÍNH ==================
st.header("3️⃣ Thuộc tính & tập dữ liệu")

feature_cols = ["pregnant", "insulin", "bmi", "age", "glucose", "bp", "pedigree"]
X = pima[feature_cols]
y = pima["label"]

st.write("**Các thuộc tính sử dụng:**")
st.write(", ".join(feature_cols))

# ================== TRAIN / TEST ==================
st.header("4️⃣ Chia dữ liệu & huấn luyện mô hình")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

col1, col2 = st.columns(2)
col1.metric("Train", X_train.shape[0])
col2.metric("Test", X_test.shape[0])

clf = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=1
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# ================== ĐÁNH GIÁ ==================
st.header("5️⃣ Đánh giá mô hình")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Criterion", clf.criterion)
col2.metric("Max depth", clf.max_depth)
col3.metric("Độ sâu thực tế", clf.get_depth())
col4.metric("Số nút lá", clf.get_n_leaves())

st.success(f"🎯 Accuracy: {accuracy_score(y_test, y_pred):.2%}")

with st.expander("📌 Confusion Matrix"):
    cm_df = pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        columns=["Dự đoán 0", "Dự đoán 1"],
        index=["Thực tế 0", "Thực tế 1"]
    )
    st.dataframe(cm_df)

with st.expander("📌 Classification Report"):
    st.text(classification_report(y_test, y_pred, target_names=["Không bệnh", "Bị bệnh"]))

# ================== SO SÁNH ==================
st.header("6️⃣ So sánh nhãn thực tế & dự đoán")
compare_df = pd.DataFrame({
    "Thực tế": y_test.values[:10],
    "Dự đoán": y_pred[:10]
})
st.dataframe(compare_df)

# ================== CÂY QUYẾT ĐỊNH ==================
st.header("7️⃣ Trực quan hóa cây quyết định")

fig2, ax2 = plt.subplots(figsize=(22, 10))
plot_tree(
    clf,
    feature_names=feature_cols,
    class_names=["Không bệnh (0)", "Bị bệnh (1)"],
    filled=True,
    rounded=True,
    fontsize=9,
    ax=ax2
)
st.pyplot(fig2)

# ================== DỰ ĐOÁN MẪU ==================
st.header("8️⃣ Dự đoán cho bệnh nhân mẫu")

new_patient = pd.DataFrame({
    "pregnant": [2],
    "insulin": [120],
    "bmi": [32.5],
    "age": [45],
    "glucose": [150],
    "bp": [85],
    "pedigree": [0.6]
})

prediction = clf.predict(new_patient)
probability = clf.predict_proba(new_patient)

st.dataframe(new_patient)

if prediction[0] == 1:
    st.error("❌ KẾT LUẬN: BỊ BỆNH TIỂU ĐƯỜNG")
else:
    st.success("✅ KẾT LUẬN: KHÔNG BỊ BỆNH TIỂU ĐƯỜNG")

st.write("**Xác suất:**")
st.write(f"- Không bệnh: {probability[0][0]:.2%}")
st.write(f"- Bị bệnh  : {probability[0][1]:.2%}")
