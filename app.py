import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
import joblib
import io

st.set_page_config(page_title="Radio Source Localization", layout="wide")
st.title("📡 Ứng dụng Định vị Nguồn Phát Xạ Vô Tuyến")
st.markdown("""
<style>
    .main {
        background-color: #f9f9f9;
    }
    h1, h2, h3 {
        color: #003366;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Cài đặt")
    num_receivers = st.slider("Số trạm thu", 2, 10, 5)
    noise_level = st.slider("Mức nhiễu", 0.0, 1.0, 0.1)
    model_type = st.selectbox("Chọn mô hình", ["Random Forest", "MLP", "SVR"])

# Tạo dữ liệu tổng hợp
@st.cache_data
def generate_synthetic_data(n_samples=1000, n_receivers=5, noise=0.1):
    np.random.seed(42)
    sources = np.random.rand(n_samples, 2) * 100
    receivers = np.random.rand(n_receivers, 2) * 100
    data = []
    for src in sources:
        rssi = -20 * np.log10(np.linalg.norm(receivers - src, axis=1)) + np.random.normal(0, noise, n_receivers)
        data.append(np.concatenate([rssi, src]))
    columns = [f"RSSI_{i}" for i in range(n_receivers)] + ["Longitude", "Latitude"]
    return pd.DataFrame(data, columns=columns), receivers

data, receivers = generate_synthetic_data(n_receivers=num_receivers, noise=noise_level)

# Huấn luyện mô hình
X = data.drop(columns=["Longitude", "Latitude"])
y = data[["Longitude", "Latitude"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=min(10, X.shape[1]))
X_pca = pca.fit_transform(X_scaled)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

if model_type == "Random Forest":
    model = RandomForestRegressor()
elif model_type == "MLP":
    model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=500)
else:
    model = SVR()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.success(f"MSE của mô hình {model_type}: {mse:.2f}")

# Lưu mô hình
model_filename = f"model_{model_type.lower().replace(' ', '_')}.joblib"
if st.button("💾 Lưu mô hình"):
    joblib.dump({"model": model, "scaler": scaler, "pca": pca}, model_filename)
    st.success(f"Đã lưu mô hình vào {model_filename}")

# Tải file CSV để dự đoán
st.subheader("📁 Dự đoán từ file CSV thực tế")
uploaded_file = st.file_uploader("Tải lên file CSV chứa RSSI", type="csv")
if uploaded_file is not None:
    df_upload = pd.read_csv(uploaded_file)
    if len(df_upload.columns) != num_receivers:
        st.warning(f"File cần có đúng {num_receivers} cột RSSI")
    else:
        try:
            model_loaded = joblib.load(model_filename)
            X_real = model_loaded["scaler"].transform(df_upload)
            X_real_pca = model_loaded["pca"].transform(X_real)
            y_real_pred = model_loaded["model"].predict(X_real_pca)
            df_pred = pd.DataFrame(y_real_pred, columns=["Longitude", "Latitude"])
            st.dataframe(df_pred)

            # Hiển thị bản đồ
            st.subheader("🗺️ Bản đồ vị trí dự đoán")
            m = folium.Map(location=[50, 50], zoom_start=4)
            for _, row in df_pred.iterrows():
                folium.Marker([row["Latitude"], row["Longitude"],],
                              popup=f"Dự đoán: ({row['Latitude']:.2f}, {row['Longitude']:.2f})",
                              icon=folium.Icon(color='red')).add_to(m)
            for rx in receivers:
                folium.CircleMarker(location=[rx[1], rx[0]], radius=5, color='blue', fill=True, popup="Trạm thu").add_to(m)
            folium_static(m)

        except Exception as e:
            st.error(f"Lỗi khi tải mô hình hoặc dự đoán: {e}")

# Biểu đồ scatter
st.subheader("📊 So sánh vị trí thực và dự đoán")
fig, ax = plt.subplots()
ax.scatter(y_test["Longitude"], y_test["Latitude"], c='blue', label='Thực tế')
ax.scatter(y_pred[:, 0], y_pred[:, 1], c='red', label='Dự đoán')
ax.legend()
ax.set_xlabel("Kinh độ")
ax.set_ylabel("Vĩ độ")
st.pyplot(fig)
