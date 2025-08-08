import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Yêu cầu đồ án")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
### **Project 1: Image Retrieval - Truy vấn hình ảnh**
**Cài đặt một hệ thống tìm kiếm ảnh, với các yêu cầu sau:**
- Có giao diện người dùng, ít nhất cần có module nhập query và danh sách ảnh kết quả.
- Sử dụng bộ dữ liệu các bộ dữ liệu có sẵn, ví dụ như https://paperswithcode.com/task/image-retrieval#datasets, hoặc bộ dữ liệu tự thu thập kích thước tối thiểu 5K ảnh, 50 query.
- Đánh giá kết quả và so sánh với các phương pháp khác đã công bố.
- Phân tích kết quả cho thấy ưu/nhược điểm của phương pháp lựa chọn.

**Điểm cộng:**
- Sử dụng các kĩ thuật tìm kiếm trên CSDL ảnh lớn.
- Sử dụng bất kì benchmark dataset nào khác > 20K ảnh.
- Sử dụng các loại hình query khác nhau (ảnh, text, âm thanh hoặc kết hợp).
"""
)