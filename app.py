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
import matplotlib.pyplot as plt # Import matplotlib

# --- Hàm tạo dữ liệu tổng hợp ---
def generate_synthetic_data(num_samples, num_receivers, noise_level=0.1):
    """
    Tạo dữ liệu tổng hợp cho bài toán định vị nguồn RF.
    Đầu vào: tọa độ trạm thu (x, y), RSSI, AoA.
    Đầu ra: tọa độ nguồn phát xạ (x, y).
    """
    st.write(f"Đang tạo {num_samples} mẫu dữ liệu với {num_receivers} trạm thu...")

    # Tọa độ nguồn phát xạ ngẫu nhiên (target)
    source_coords = np.random.rand(num_samples, 2) * 100  # Nguồn trong phạm vi 100x100

    # Tọa độ trạm thu ngẫu nhiên
    receiver_coords = np.random.rand(num_samples, num_receivers * 2) * 100

    # RSSI và AoA (mô phỏng đơn giản)
    # Giả định RSSI giảm theo khoảng cách và AoA là góc từ trạm thu đến nguồn
    rssi_data = np.zeros((num_samples, num_receivers))
    aoa_data = np.zeros((num_samples, num_receivers))

    for i in range(num_samples):
        for j in range(num_receivers):
            rx_x, rx_y = receiver_coords[i, j*2], receiver_coords[i, j*2 + 1]
            src_x, src_y = source_coords[i, 0], source_coords[i, 1]

            distance = np.sqrt((src_x - rx_x)**2 + (src_y - rx_y)**2)
            
            # RSSI: Giảm theo log khoảng cách + nhiễu
            rssi_data[i, j] = 100 - 20 * np.log10(distance + 1e-6) + np.random.normal(0, noise_level * 10)

            # AoA: Góc từ trạm thu đến nguồn + nhiễu
            angle = np.degrees(np.arctan2(src_y - rx_y, src_x - rx_x))
            aoa_data[i, j] = angle + np.random.normal(0, noise_level * 5)

    # Kết hợp tất cả các đặc trưng đầu vào
    # receiver_coords (num_samples, num_receivers * 2)
    # rssi_data (num_samples, num_receivers)
    # aoa_data (num_samples, num_receivers)
    feature_names = # Khởi tạo danh sách rỗng
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
    R = 6371000  # Bán kính Trái đất bằng mét
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

# --- Giao diện Streamlit ---
st.set_page_config(layout="wide", page_title="Ứng dụng Huấn luyện Mô hình Định vị RF")

st.title("Ứng dụng Huấn luyện Mô hình Định vị Nguồn Phát xạ RF")
st.markdown("""
Ứng dụng này cho phép bạn khám phá các mô hình học máy để dự đoán tọa độ nguồn phát xạ RF.
Bạn có thể tạo dữ liệu tổng hợp, áp dụng tiền xử lý và huấn luyện các mô hình khác nhau.
""")

# --- Sidebar để điều khiển ---
st.sidebar.header("Cấu hình Dữ liệu & Mô hình")

# Cấu hình dữ liệu
st.sidebar.subheader("1. Tạo Dữ liệu Tổng hợp")
num_samples = st.sidebar.slider("Số lượng mẫu dữ liệu", 100, 5000, 1000)
num_receivers = st.sidebar.slider("Số lượng trạm thu", 2, 10, 4)
noise_level = st.sidebar.slider("Mức độ nhiễu (0.0 - 1.0)", 0.0, 1.0, 0.1)

if st.sidebar.button("Tạo Dữ liệu"):
    X, y = generate_synthetic_data(num_samples, num_receivers, noise_level)
    st.session_state['X'] = X
    st.session_state['y'] = y
    st.session_state['data_generated'] = True
    st.sidebar.success("Dữ liệu đã được tạo!")
else:
    if 'data_generated' not in st.session_state:
        st.session_state['data_generated'] = False

# Hiển thị dữ liệu mẫu nếu đã tạo
if st.session_state['data_generated']:
    st.subheader("Dữ liệu tổng hợp đã tạo (5 hàng đầu tiên)")
    st.dataframe(st.session_state['X'].head())
    st.dataframe(st.session_state['y'].head())

    # Tiền xử lý
    st.sidebar.subheader("2. Tiền xử lý Dữ liệu")
    use_scaler = st.sidebar.checkbox("Sử dụng Chuẩn hóa (StandardScaler)", True)
    use_pca = st.sidebar.checkbox("Sử dụng Giảm chiều (PCA)", False)
    pca_components = 0
    if use_pca:
        max_pca_components = min(st.session_state['X'].shape[1], num_samples - 1)
        if max_pca_components > 0:
            pca_components = st.sidebar.slider(
                "Số lượng thành phần PCA",
                1,
                max_pca_components,
                min(5, max_pca_components)
            )
        else:
            st.sidebar.warning("Không đủ đặc trưng hoặc mẫu để áp dụng PCA.")
            use_pca = False

    X_processed = st.session_state['X'].copy()
    y_processed = st.session_state['y'].copy()

    if use_scaler:
        scaler = StandardScaler()
        X_processed = pd.DataFrame(scaler.fit_transform(X_processed), columns=X_processed.columns)
        st.sidebar.info("Đã áp dụng Chuẩn hóa.")

    if use_pca and pca_components > 0:
        pca = PCA(n_components=pca_components)
        X_processed = pd.DataFrame(pca.fit_transform(X_processed))
        st.sidebar.info(f"Đã áp dụng PCA với {pca_components} thành phần.")

    # Chia tập dữ liệu
    test_size = st.sidebar.slider("Tỷ lệ tập kiểm tra", 0.1, 0.5, 0.2, 0.05)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, test_size=test_size, random_state=42
    )
    st.sidebar.write(f"Kích thước tập huấn luyện: {X_train.shape} mẫu")
    st.sidebar.write(f"Kích thước tập kiểm tra: {X_test.shape} mẫu")

    # Cấu hình mô hình
    st.sidebar.subheader("3. Chọn Mô hình & Siêu tham số")
    model_choice = st.sidebar.selectbox(
        "Chọn Mô hình Học máy",
        ("Random Forest Regressor", "MLP Regressor (Neural Network)", "Support Vector Regressor")
    )

    model = None
    if model_choice == "Random Forest Regressor":
        n_estimators = st.sidebar.slider("Số lượng cây (n_estimators)", 50, 500, 100, 50)
        max_depth = st.sidebar.slider("Độ sâu tối đa (max_depth)", 5, 50, 10, 5)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        st.sidebar.write("Mô hình Hồi quy Rừng ngẫu nhiên được chọn.")
        st.sidebar.markdown("*(Hiệu quả với nhiễu, giảm overfitting)* [2]")

    elif model_choice == "MLP Regressor (Neural Network)":
        hidden_layer_sizes = st.sidebar.text_input("Kích thước lớp ẩn (ví dụ: 100,50)", "100,50")
        hidden_layer_sizes = tuple(map(int, hidden_layer_sizes.split(',')))
        max_iter = st.sidebar.slider("Số lần lặp tối đa (max_iter)", 100, 1000, 200, 50)
        learning_rate_init = st.sidebar.slider("Tốc độ học (learning_rate_init)", 0.0001, 0.1, 0.001, 0.0001, format="%f")
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            learning_rate_init=learning_rate_init,
            random_state=42,
            early_stopping=True, # Giúp ngăn chặn overfitting và tăng tốc độ hội tụ
            n_iter_no_change=20 # Số epoch không cải thiện để dừng sớm
        )
        st.sidebar.write("Mô hình Mạng thần kinh truyền thẳng (MLP) được chọn.")
        st.sidebar.markdown("*(Xử lý phi tuyến tính, học ánh xạ phức tạp)* [2]")

    elif model_choice == "Support Vector Regressor":
        C = st.sidebar.slider("Tham số C (C)", 0.1, 10.0, 1.0, 0.1)
        epsilon = st.sidebar.slider("Epsilon (epsilon)", 0.01, 1.0, 0.1, 0.01)
        kernel = st.sidebar.selectbox("Kernel", ("rbf", "linear", "poly"))
        model = SVR(C=C, epsilon=epsilon, kernel=kernel)
        st.sidebar.write("Mô hình Hồi quy Vector Hỗ trợ (SVR) được chọn.")
        st.sidebar.markdown("*(Mạnh mẽ với nhiễu nhỏ)* [2]")

    # Huấn luyện mô hình
    st.sidebar.subheader("4. Huấn luyện Mô hình")
    if st.sidebar.button("Bắt đầu Huấn luyện"):
        if model:
            st.write("---")
            st.subheader(f"Đang huấn luyện mô hình: {model_choice}")
            with st.spinner("Đang huấn luyện..."):
                model.fit(X_train, y_train)
            st.success("Huấn luyện hoàn tất!")

            # Đánh giá mô hình
            st.subheader("5. Đánh giá Hiệu suất Mô hình")
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)

            st.write(f"**Sai số bình phương trung bình (MSE):** {mse:.4f} [3, 4, 5, 6]")
            st.write(f"**Sai số tuyệt đối trung bình (MAE):** {mae:.4f} [3, 4, 5, 6]")
            st.write(f"**Sai số căn bậc hai trung bình (RMSE):** {rmse:.4f} [5, 6]")

            # Tính khoảng cách Haversine nếu có thể (giả định tọa độ là vĩ độ/kinh độ)
            # Để minh họa Haversine, chúng ta sẽ giả định source_x và source_y là vĩ độ và kinh độ
            # trong một phạm vi nhỏ để khoảng cách Haversine không quá khác biệt so với Euclidean
            # cho mục đích demo.
            
            # Giả định: source_x là vĩ độ, source_y là kinh độ
            haversine_errors = # Khởi tạo danh sách rỗng
            for i in range(y_test.shape): # Sửa lỗi: dùng y_test.shape để lấy số hàng
                # Giả định y_test.iloc[i, 0] là vĩ độ, y_test.iloc[i, 1] là kinh độ
                # Giả định y_pred[i, 0] là vĩ độ, y_pred[i, 1] là kinh độ
                dist = haversine_distance(y_test.iloc[i, 0], y_test.iloc[i, 1], y_pred[i, 0], y_pred[i, 1])
                haversine_errors.append(dist)
            
            mean_haversine_error = np.mean(haversine_errors)
            st.write(f"**Sai số khoảng cách Haversine trung bình (mét):** {mean_haversine_error:.2f} [6]")
            st.markdown("*(Lưu ý: Khoảng cách Haversine được tính dựa trên giả định tọa độ là vĩ độ/kinh độ.)*")

            st.subheader("So sánh Dự đoán và Thực tế (10 mẫu đầu tiên)")
            comparison_df = pd.DataFrame({
                'Thực tế X': y_test.iloc[:10, 0],
                'Thực tế Y': y_test.iloc[:10, 1],
                'Dự đoán X': y_pred[:10, 0],
                'Dự đoán Y': y_pred[:10, 1]
            })
            st.dataframe(comparison_df)

            st.subheader("Biểu đồ phân tán: Dự đoán so với Thực tế")
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))

            ax.scatter(y_test.iloc[:, 0], y_pred[:, 0], alpha=0.5) # Sửa lỗi: ax.scatter
            ax.plot([min(y_test.iloc[:, 0]), max(y_test.iloc[:, 0])],
                       [min(y_test.iloc[:, 0]), max(y_test.iloc[:, 0])],
                       color='red', linestyle='--')
            ax.set_title('Dự đoán X so với Thực tế X')
            ax.set_xlabel('Thực tế X')
            ax.set_ylabel('Dự đoán X')
            ax.grid(True)

            ax.[1]scatter(y_test.iloc[:, 1], y_pred[:, 1], alpha=0.5) # Sửa lỗi: ax.[1]scatter
            ax.[1]plot([min(y_test.iloc[:, 1]), max(y_test.iloc[:, 1])],
                       [min(y_test.iloc[:, 1]), max(y_test.iloc[:, 1])],
                       color='red', linestyle='--')
            ax.[1]set_title('Dự đoán Y so với Thực tế Y')
            ax.[1]set_xlabel('Thực tế Y')
            ax.[1]set_ylabel('Dự đoán Y')
            ax.[1]grid(True)

            st.pyplot(fig)

        else:
            st.error("Vui lòng chọn một mô hình để huấn luyện.")
    else:
        st.info("Nhấn 'Bắt đầu Huấn luyện' để huấn luyện mô hình đã chọn.")

else:
    st.info("Vui lòng nhấn 'Tạo Dữ liệu' trong thanh bên để bắt đầu.")

st.markdown("---")
st.markdown("""
**Lưu ý về dữ liệu tổng hợp:**
Dữ liệu được tạo trong ứng dụng này là mô phỏng đơn giản. Trong thực tế, việc thu thập dữ liệu RF chất lượng cao, có nhãn là một thách thức lớn và thường tốn kém, mất thời gian.[7, 8, 9, 10] Các kỹ thuật phức tạp hơn như dò tia (ray tracing) và mô hình kênh không dây nâng cao được sử dụng để tạo dữ liệu tổng hợp thực tế hơn.[7, 11, 12, 13]
""")
