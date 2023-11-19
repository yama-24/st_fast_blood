import streamlit as st
import requests
from PIL import Image
# import io
from visualize import visualize_results

# Streamlitインターフェースの設定
st.sidebar.title('血液の顕微鏡画像から細胞検出')

# fast_api_url
# fast_api_url = "http://localhost:8000/detect/" # local
fast_api_url = "https://st-fast-blood.onrender.com/detect/" # Render

# ユーザーが画像をアップロード
uploaded_file = st.sidebar.file_uploader("画像をアップロードしてください", type=["jpg"])

if uploaded_file is not None:
    # FastAPIサーバーに画像を送信
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(fast_api_url, files=files)
    data = response.json()

    # 画像と検出結果の表示
    image = Image.open(uploaded_file)
    # st.image(image, caption='アップロードされた画像', use_column_width="auto")
    # st.write(data)

    st.image(visualize_results(image, data), caption='推論結果画像', use_column_width="auto")
