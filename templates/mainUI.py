import streamlit as st
import requests

# Tiêu đề ứng dụng
st.title("TRÍCH XUẤT THÔNG TIN TỪ CCCD")

# Tải file ảnh lên
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

# URL API Flask
api_url = "http://localhost:5000/predict"

# Thêm nút Submit và xử lý khi nhấn nút
if st.button("Submit"):
    if uploaded_file is not None:
        try:
            # Hiển thị thông tin file được tải lên
            st.write("Filename:", uploaded_file.name)
            
            # Chuẩn bị payload và gửi request đến API
            files = {"file": (uploaded_file.name, uploaded_file, "multipart/form-data")}
            response = requests.post(api_url, files=files)
            
            # Xử lý kết quả từ API
            if response.status_code == 200:
                result = response.json().get("result", "")
                
                # Giả sử kết quả trả về là dictionary với các khóa tương ứng
                st.session_state.extracted_data = {
                    "id": result.get("ID", ""),
                    "name": result.get("Name", ""),
                    "dob": result.get("Date of Birth", ""),
                    "sex": result.get("Sex", ""),
                    "nationality": result.get("Nationality", ""),
                    "origin": result.get("Place of Origin", ""),
                    "residence": result.get("Place of Residence", ""),
                    "expiry": result.get("Expiry", "")
                }
                st.success("Kết quả trích xuất đã được cập nhật!")
            else:
                st.error(f"Lỗi từ API: {response.status_code}")
                st.text(response.text)
        except Exception as e:
            st.error("Đã xảy ra lỗi!")
            st.text(str(e))
    else:
        st.warning("Vui lòng tải lên một file trước khi nhấn Submit!")

# Tạo các trường để hiển thị thông tin
st.subheader("Thông tin trích xuất:")
id_field = st.text_input("ID", value=st.session_state.get("extracted_data", {}).get("id", ""), key="id")
name_field = st.text_input("Name", value=st.session_state.get("extracted_data", {}).get("name", ""), key="name")
dob_field = st.text_input("Date of Birth", value=st.session_state.get("extracted_data", {}).get("dob", ""), key="dob")
sex_field = st.text_input("Sex", value=st.session_state.get("extracted_data", {}).get("sex", ""), key="sex")
nationality_field = st.text_input("Nationality", value=st.session_state.get("extracted_data", {}).get("nationality", ""), key="nationality")
origin_field = st.text_input("Place of Origin", value=st.session_state.get("extracted_data", {}).get("origin", ""), key="origin")
residence_field = st.text_input("Place of Residence", value=st.session_state.get("extracted_data", {}).get("residence", ""), key="residence")
expiry_field = st.text_input("Expiry", value=st.session_state.get("extracted_data", {}).get("expiry", ""), key="expiry")
