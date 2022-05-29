# 以下を「app.py」に書き込み
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import os
os.chdir("/content/")
from model import predict
from PIL import Image
import numpy as np


st.set_option("deprecation.showfileUploaderEncoding", False)

st.sidebar.title("CNNを用いた画像認識")
st.sidebar.write("画像認識モデルを使って与えられた画像が何の画像かを判定します!")

st.sidebar.write("")

#img_source = st.sidebar.radio("画像のソースを選択してください。",
                              #("既存画像で試す", "画像をアップロード", "カメラで撮影"))
img_source = st.selectbox("画像選択方法を選んでください。",
                          ["", "既存画像で試す", "画像をアップロード", "カメラで撮影"])

n_top = st.sidebar.slider("上位のいくつの判定結果を見る？",
                                  min_value=3,
                                  max_value=10)
if img_source == "既存画像で試す":
    #img_file =Image.open("shosai_dogday_main.jpg")
    #img_file = img_file.convert("RGB")
    image_file = np.load("image.npy")
    img_file = Image.fromarray(image_file)
elif img_source == "画像をアップロード":
    img_file = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg"])
elif img_source == "カメラで撮影":
    img_file = st.camera_input("カメラで撮影")
else:
  img_file = None
  st.warning("選択方法を選んでください")


if (img_file is not None):
    with st.spinner("只今推定しています..."):
        if img_source == "既存画像で試す":
          img = img_file
        else:
          img = Image.open(img_file)
        st.image(img, caption="対象の画像", width=480)
        st.write("")

        # 予測
        results = predict(img)

        # 結果の表示
        st.subheader("判定結果")
        
        #n_top = 5  # 確率が高い順に3位まで返す
        check_label = []
        cnt = 0        
        for result in results[:n_top]:
            if cnt == 0:
              st.info(str(round(result[2]*100, 2)) + "%の確率で" + result[0] + "です。")
            else:
              st.write(str(round(result[2]*100, 2)) + "%の確率で" + result[0] + "です。")
            check_label.append(result[0])
            cnt += 1



        st.sidebar.write("-------------------------------------")
        st.sidebar.warning("正解はどれですか？")
        img_label = st.sidebar.radio("",
                              ("", check_label[0], check_label[1], check_label[2], "いずれでもない"))
        if img_label == check_label[0]:
          st.success("予測を正しく行うことが出来ました！")
        elif img_label == "いずれでもない":
          st.error("申し訳ございません！今後の精度向上にご期待ください")
        elif img_label != "" :
          st.warning("ご回答ありがとうございました。ぜひ他の画像もお試しください")




        # 円グラフの表示
        st.subheader("画像判定予測値のグラフ")
        pie_labels = [result[1] for result in results[:n_top]]
        pie_labels.append("others")
        pie_probs = [result[2] for result in results[:n_top]]
        pie_probs.append(sum([result[2] for result in results[n_top:]]))
        fig, ax = plt.subplots()
        wedgeprops={"width":0.3, "edgecolor":"white"}
        textprops = {"fontsize":6}
        ax.pie(pie_probs, labels=pie_labels, counterclock=False, startangle=90,
               textprops=textprops, autopct="%.2f", wedgeprops=wedgeprops)  # 円グラフ
        st.pyplot(fig)




