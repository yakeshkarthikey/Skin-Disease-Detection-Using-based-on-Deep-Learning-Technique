import streamlit as st 
import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
import time

# def add_bg_from_url():
#     st.markdown(
#          f"""{'Melanoma': 0, 'Ringworm': 1, 'healthy': 2, 'psoriasis': 3}
#          <style>
#          .stApp {{
#              background-image: url("https://img.onmanorama.com/content/dam/mm/en/lifestyle/health/images/2022/7/1/stethoscope-c.jpg");
#              background-attachment: fixed;
#              background-size: cover
#          }}
#          </style>
#          """,
#          unsafe_allow_html=True
#      )
# # {'healthy': 0, 'melanoma': 1, 'eczema': 2, 'psoriasis': 3}
 
# add_bg_from_url() 

st.title("Skin Disease Detectior ")
st.subheader("Enter your Skin Image....")

with st.sidebar:
    file = st.file_uploader("enter the image")


    with st.spinner("Loading..."):
        time.sleep(1)
    st.success("Done!")


def start(file):
    img_file_buffer = file
    image = Image.open(img_file_buffer)
    st.write("input image")
    # st.image(image)
    img_array = np.array(image) # if you want to pass it to OpenCV
    img = 'D:/Py/aravindh_proj/color_img.jpg'
    cv2.imwrite(img, img_array)
    # st.image(image, caption="The caption", use_column_width=True)
    # array = np.reshape(img_array, (128, 128))

    if file:
        st.info("file entered")
        st.image(file)

    button = st.button('Enter')

    if button:
        model = load_model('D:/Py/aravindh_proj/model-1.h5')
        batch_size = 16
        image = cv2.imread(img)
        img = Image.fromarray(image)
        img = img.resize((128, 128))
        img = np.array(img)
        input_img = np.expand_dims(img, axis=0)
        # print(input_img)
        # print(input_img.shape)
        i = input_img.reshape(-1,1)
        # print("shape-i",i.shape)
            # result = model.predict_classes(input_img)
        result = model.predict(input_img)
        # st.write(result)
        st.subheader('The skin report is..')
        print(result)
        r = result[0]
        print(r)
        for i in range(len(r)):
            if r[i]==max(r):
                m = i
        i = m
        print(i)
        if i==0:
            st.subheader("skin-Disease State: Melanoma")
        elif i==1:
            st.subheader("skin-Disease state: Ringworm")
        elif i==2:
            st.subheader("skin-Disease state: healthy")
        elif i==3:
            st.subheader("skin-Disease state: psoriasis")
     

try:
    start(file)
except:
    pass
