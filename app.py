import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import io

# --- Hàm tạo dữ liệu tổng hợp ---
def generate_synthetic_data(num_samples, num_receivers, noise_level=0.1):
    st.write(f"Đang tạo {num_samples} mẫu dữ liệu với {num_receivers} trạm thu...")

    # Tọa độ nguồn phát xạ ngẫu nhiên (target)
    source_coords = np.random.rand(num_samples, 2) * 100  # Nguồn trong phạm vi 100x100

    # Tọa độ trạm thu ngẫu nhiên
    receiver_coords = np.random.rand(num_samples, num_receivers * 2) * 100

    # RSSI và AoA (mô phỏng đơn giản)
    rssi_data = np.zeros((num_samples, num_receivers))
    aoa_data = np.zeros((num_samples, num_receivers))

    for i in range(num_samples):
        for j in range(num_receivers):
            rx_x, rx_y = receiver_coords[i, j * 2], receiver_coords[i, j * 2 + 1]
            src_x, src_y = source_coords[i, 0], source_coords[i, 1]

            distance = np.sqrt((src_x - rx_x)**2 + (src_y - rx_y)**2)
            rssi_data[i, j] = 100 - 20 * np.log10(distance + 1e-6) + np.random.normal(0, noise_level * 10)
            angle = np.degrees(np.arctan2(src_y - rx_y, src_x - rx_x))
            aoa_data[i, j] = angle + np.random.normal(0, noise_level * 5)

    # Gộp dữ liệu đầu vào
    X = np.hstack([receiver_coords, rssi_data, aoa_data])
    y = source_coords

    # Tạo tên cột cho dataframe
    feature_names = []
    for j in range(num_receivers):
        feature_names.append(f'rx_{j+1}_x')
        feature_names.append(f'rx_{j+1}_y')
    for j in range(num_receivers):
        feature_names.append(f'rssi_{j+1}')
    for j in range(num_receivers):
        feature_names.append(f'aoa_{j+1}')

    df_X = pd.DataFrame(X, columns=feature_names)
    df_y = pd.DataFrame(y, columns=['source_x', 'source_y'])

    st.success("Tạo dữ liệu tổng hợp thành công!")
    return df_X, df_y

# --- Hàm tính khoảng cách Haversine (cho tọa độ địa lý, giả định cho demo) ---
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# --- Giao diện Streamlit ---
st.set_page_config(layout="wide", page_title="Ứng dụng Huấn luyện Mô hình Định vị RF")

st.title("Ứng dụng Huấn luyện Mô hình Định vị Nguồn Phát xạ RF")
st.markdown("""
Ứng dụng này cho phép bạn tạo dữ liệu tổng hợp, huấn luyện mô hình dự đoán tọa độ nguồn phát RF, dự đoán và xuất kết quả.
""")

# --- Sidebar ---
st.sidebar.header("Cấu hình Dữ liệu & Mô hình")

# Tạo dữ liệu
st.sidebar.subheader("1. Tạo Dữ liệu Tổng hợp")
num_samples = st.sidebar.slider("Số lượng mẫu dữ liệu", 100, 5000, 1000)
num_receivers = st.sidebar.slider("Số lượng trạm thu", 2, 10, 4)
noise_level = st.sidebar.slider("Mức độ nhiễu", 0.0, 1.0, 0.1)

if st.sidebar.button("Tạo Dữ liệu"):
    X, y = generate_synthetic_data(num_samples, num_receivers, noise_level)
    st.session_state['X'] = X
    st.session_state['y'] = y
    st.session_state['data_generated'] = True
    st.sidebar.success("Dữ liệu đã được tạo!")
else:
    if 'data_generated' not in st.session_state:
        st.session_state['data_generated'] = False

if st.session_state.get('data_generated', False):
    st.subheader("📊 Dữ liệu Tổng hợp (5 dòng đầu tiên)")
    st.dataframe(st.session_state['X'].head())
    st.dataframe(st.session_state['y'].head())

    # Tiền xử lý
    st.sidebar.subheader("2. Tiền xử lý Dữ liệu")
    use_scaler = st.sidebar.checkbox("Chuẩn hóa (StandardScaler)", True)
    use_pca = st.sidebar.checkbox("Giảm chiều (PCA)", False)

    X_processed = st.session_state['X'].copy()
    y_processed = st.session_state['y'].copy()

    if use_scaler:
        scaler = StandardScaler()
        X_processed = pd.DataFrame(scaler.fit_transform(X_processed), columns=X_processed.columns)

    if use_pca:
        max_pca_components = min(X_processed.shape[1], num_samples - 1)
        if max_pca_components > 0:
            pca_components = st.sidebar.slider("Số thành phần PCA", 1, max_pca_components, min(5, max_pca_components))
            pca = PCA(n_components=pca_components)
            X_processed = pd.DataFrame(pca.fit_transform(X_processed))
        else:
            st.sidebar.warning("Không thể áp dụng PCA.")

    test_size = st.sidebar.slider("Tỷ lệ kiểm tra", 0.1, 0.5, 0.2, 0.05)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=test_size, random_state=42)

    # Mô hình
    st.sidebar.subheader("3. Mô hình & Siêu tham số")
    model_choice = st.sidebar.selectbox("Chọn mô hình", [
        "Random Forest Regressor",
        "MLP Regressor (Neural Network)",
        "Support Vector Regressor"
    ])

    model = None
    if model_choice == "Random Forest Regressor":
        n_estimators = st.sidebar.slider("Số cây", 50, 500, 100, 50)
        max_depth = st.sidebar.slider("Độ sâu tối đa", 5, 50, 10, 5)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    elif model_choice == "MLP Regressor (Neural Network)":
        hidden_layer_sizes = st.sidebar.text_input("Kích thước lớp ẩn", "100,50")
        hidden_layer_sizes = tuple(map(int, hidden_layer_sizes.split(',')))
        max_iter = st.sidebar.slider("Số vòng lặp", 100, 1000, 200, 50)
        learning_rate_init = st.sidebar.slider("Tốc độ học", 0.0001, 0.1, 0.001, 0.0001, format="%f")
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            learning_rate_init=learning_rate_init,
            early_stopping=True,
            random_state=42
        )

    elif model_choice == "Support Vector Regressor":
        C = st.sidebar.slider("C", 0.1, 10.0, 1.0, 0.1)
        epsilon = st.sidebar.slider("Epsilon", 0.01, 1.0, 0.1, 0.01)
        kernel = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly"])
        model = SVR(C=C, epsilon=epsilon, kernel=kernel)

    # Huấn luyện mô hình
    st.sidebar.subheader("4. Huấn luyện Mô hình")
    if st.sidebar.button("Bắt đầu Huấn luyện"):
        with st.spinner("Đang huấn luyện mô hình..."):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        st.subheader("Kết quả Huấn luyện")
        st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")

        # Vẽ biểu đồ scatter giữa y_test và y_pred
        fig, ax = plt.subplots()
        ax.scatter(y_test['source_x'], y_test['source_y'], label="Thực tế", c='blue')
        ax.scatter(y_pred[:, 0], y_pred[:, 1], label="Dự đoán", c='red', alpha=0.6)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Vị trí nguồn phát xạ: Thực tế vs Dự đoán")
        ax.legend()
        st.pyplot(fig)

        # Lưu mô hình và dữ liệu dự đoán để dùng sau (nếu muốn)
        st.session_state['model'] = model
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
        st.session_state['y_pred'] = y_pred

    # Phần dự đoán với dữ liệu test
    if st.session_state.get('model', None):
        st.subheader("Dự đoán với tập Test")
        if st.button("Hiển thị kết quả dự đoán và bản đồ"):

            output_df = st.session_state['X_test'].copy()
            output_df = output_df.reset_index(drop=True)
            output_df['source_x_thucte'] = st.session_state['y_test'].reset_index(drop=True)['source_x']
            output_df['source_y_thucte'] = st.session_state['y_test'].reset_index(drop=True)['source_y']
            output_df['source_x_du_doan'] = st.session_state['y_pred'][:, 0]
            output_df['source_y_du_doan'] = st.session_state['y_pred'][:, 1]

            st.write("Kết quả Dự đoán (5 mẫu đầu tiên):")
            st.dataframe(output_df.head())

            # Nút xuất file CSV
            csv_buffer = io.StringIO()
            output_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="⬇️ Tải file kết quả CSV",
                data=csv_buffer.getvalue(),
                file_name="ket_qua_du_doan.csv",
                mime="text/csv"
            )

            # Vẽ bản đồ Folium
            st.markdown("### 🗺️ Bản đồ Vị trí Thực tế và Dự đoán")

            # Giả định tọa độ trung tâm cho hiển thị bản đồ
            lat_base, lon_base = 20.0, 105.0
            scale = 0.01

            m = folium.Map(location=[lat_base, lon_base], zoom_start=13)

            for i in range(len(output_df)):
                lat_true = lat_base + output_df.loc[i, 'source_y_thucte'] * scale
                lon_true = lon_base + output_df.loc[i, 'source_x_thucte'] * scale

                lat_pred = lat_base + output_df.loc[i, 'source_y_du_doan'] * scale
                lon_pred = lon_base + output_df.loc[i, 'source_x_du_doan'] * scale

                # Marker thực tế
                folium.CircleMarker(
                    location=[lat_true, lon_true],
                    radius=5,
                    color='blue',
                    fill=True,
                    fill_color='blue',
                    fill_opacity=0.6,
                    popup=f"Thực tế #{i+1}"
                ).add_to(m)

                # Marker dự đoán
                folium.CircleMarker(
                    location=[lat_pred, lon_pred],
                    radius=5,
                    color='red',
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.6,
                    popup=f"Dự đoán #{i+1}"
                ).add_to(m)

                # Đường nối thực tế và dự đoán
                folium.PolyLine(
                    locations=[[lat_true, lon_true], [lat_pred, lon_pred]],
                    color='gray',
                    weight=1.5,
                    opacity=0.5
                ).add_to(m)

            # Giữ bản đồ hiển thị ổn định với key
            st_data = st_folium(m, width=700, height=500, returned_objects=[], key="ban_do_dudoan")

else:
    st.info("Vui lòng tạo dữ liệu trước rồi huấn luyện mô hình để dự đoán.")

