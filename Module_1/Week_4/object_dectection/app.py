import numpy as np
import time
from PIL import Image
import streamlit as st 
from dectection import annotate_image, process_image

def main():
    
    st.title("Object Dectection for Images")
    file = st.file_uploader('Upload Image', type = ['jpg','png','jpeg'])
    
    if st.button("Dectect") and file is not None:
        with st.spinner('Uploading image...'):
            time.sleep(3)
        st.success('Image uploaded successfully!')
        st.image(file, caption = "Uploaded Image")
        
        st.text("Processing image...")
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        image = Image.open(file)
        image = np.array(image)
        detections = process_image(image)
        processed_image = annotate_image(image, detections)
        st.image(processed_image, caption="Processed Image")
        st.balloons()
        
if __name__ == "__main__":
    main()