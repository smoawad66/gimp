import streamlit as st
from algorithms import *

st.set_page_config(
    layout="wide", 
    page_title="hier serve",
    page_icon="images/logo2.png",
)

st.markdown("""
    <style>
    div[data-baseweb="select"] > div {
        cursor: default !important;
    }
    .thumbnail-container {
        position: relative;
        margin-top: 10px;
    }
    .fullscreen-btn {
        position: absolute;
        top: 5px;
        right: 5px;
        z-index: 1000;
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 4px;
        padding: 4px 8px;
    }
    .output-container {
        position: relative;
    }
    div[data-testid="column"]:first-child {
        width: 50% !important;
        min-width: 300px !important;
    }
    div[data-testid="column"]:nth-child(2) {
        width: 50% !important;
    }
    .stImage > img {
        max-height: 70vh !important;
        width: auto !important;
        object-fit: contain;
    }
    </style>
""", unsafe_allow_html=True)

OPERATION_CATEGORIES = {
    "Color Conversion": [
        "RGB to Gray",
        "Gray to Binary",
        "RGB to Binary"
    ],
    "Point Processing": [
        "Brightness Adjustment",
        "Log Transform",
        "Inverse Log Transform",
        "Negative",
        "Gamma Correction",
        "Histogram",
        "Contrast Stretching",
        "Histogram Equalization"
    ],
    "Spatial Filtering": [
        "Mean Filter Blurring",
        "Weighted Filter Blurring",
        "Edge Detection",
        "Directed Edge Detection",
        "Sharpening",
        "Directed Sharpening",
        "Rank Order Filter"
    ],
    "Transform Processing": [
        "Fourier Transform",
        "Inverse Fourier Transform",
        "Ideal Filter",
        "Butterworth Filter",
        "Gaussian Filter"
    ]
}

DIRECTION_MAPPING = {"Horizontal": "h", "Vertical": "v", "Left Diagonal": "ld", "Right Diagonal": "rd"}
OPERATION_MAPPING = {"Add": "+", "Subtract": "-", "Multiply": "*", "Divide": "/"}
FILTER_MODE_MAPPING = {"Low Pass": "l", "High Pass": "h"}

if "processed_image" not in st.session_state:
    st.session_state["processed_image"] = None

st.title("Image Processing Application")

left_column, right_column = st.columns([1, 1])

with left_column:
    with st.container():
        st.markdown("#### Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "tif"], label_visibility="collapsed")

    category = st.selectbox(
        "Select Operation Category",
        list(OPERATION_CATEGORIES.keys()),
        index=0
    )

    operation = st.selectbox(
        "Select Operation",
        OPERATION_CATEGORIES[category],
        index=0
    )

    with st.container():
        params = {}
        if operation == "Gray to Binary":
            params['threshold'] = st.slider("Threshold", 0, 255, 128)
        elif operation == "Brightness Adjustment":
            params['op'] = st.selectbox("Operation", list(OPERATION_MAPPING.keys()))
            params['value'] = st.number_input("Value", value=10)
        elif operation == "Log Transform":
            params['constant'] = st.number_input("Constant", value=1.0)
        elif operation == "Gamma Correction":
            params['gamma'] = st.slider("Gamma", 0.1, 5.0, 0.2)
            params['constant'] = st.number_input("Constant", value=1.0)
        elif operation == "Mean Filter Blurring":
            params['kernel_size'] = st.slider("Kernel Size", 3, 9, 3, step=2)
        elif operation == "Directed Edge Detection":
            params['direction'] = st.selectbox("Direction", list(DIRECTION_MAPPING.keys()))
        elif operation == "Directed Sharpening":
            params['direction'] = st.selectbox("Direction", list(DIRECTION_MAPPING.keys()))
        elif operation == "Rank Order Filter":
            params['mode'] = st.selectbox("Mode", ["Min", "Max", "Median", "Midpoint"])
            params['kernel_size'] = st.slider("Kernel Size", 3, 9, 3, step=2)
        elif operation in ["Ideal Filter", "Butterworth Filter", "Gaussian Filter"]:
            params['mode'] = st.selectbox("Mode", list(FILTER_MODE_MAPPING.keys()))
            params['d0'] = st.slider("Cut-off (D0)", 1, 256, 3)
            if operation == "Butterworth Filter":
                params['n'] = st.slider("Order (n)", 1, 10, 1)

        apply_button = st.button("Apply")

with right_column:
    
    if uploaded_file is not None:
        if "last_uploaded_file" not in st.session_state or st.session_state["last_uploaded_file"] != uploaded_file:
            st.session_state["processed_image"] = None
            st.session_state["last_uploaded_file"] = uploaded_file
            
        image = Image.open(uploaded_file)
        
        img_array = np.array(image)

        if len(img_array.shape) == 3:
            img_array_gray = rgb2Gray(img_array)
        else:
            img_array_gray = img_array

        if apply_button:
            match operation:
                case "RGB to Gray":
                    st.session_state["processed_image"] = rgb2Gray(img_array)
                case "Gray to Binary":
                    st.session_state["processed_image"] = gray2Binary(img_array_gray, params['threshold'])
                case "RGB to Binary":
                    st.session_state["processed_image"] = rgb2Binary(img_array)
                case "Brightness Adjustment":
                    st.session_state["processed_image"] = brightnessProcessing(img_array_gray, OPERATION_MAPPING[params['op']], params['value'])
                case "Histogram":
                    st.session_state["processed_image"] = histogram(img_array_gray)
                case "Log Transform":
                    st.session_state["processed_image"] = logTransform(img_array_gray, params['constant'])
                case "Inverse Log Transform":
                    st.session_state["processed_image"] = logInverseTransform(img_array_gray)
                case "Negative":
                    st.session_state["processed_image"] = negative(img_array_gray)
                case "Gamma Correction":
                    st.session_state["processed_image"] = gamma(img_array_gray, params['gamma'], params['constant'])
                case "Contrast Stretching":
                    st.session_state["processed_image"] = contrastStretching(img_array_gray)
                case "Histogram Equalization":
                    st.session_state["processed_image"] = histogramEqualization(img_array_gray)
                case "Mean Filter Blurring":
                    st.session_state["processed_image"] = meanFilterBlurring(img_array_gray, params['kernel_size'])
                case "Weighted Filter Blurring":
                    st.session_state["processed_image"] = weightFilterBlurring(img_array_gray)
                case "Edge Detection":
                    st.session_state["processed_image"] = edgeDetection(img_array_gray)
                case "Directed Edge Detection":
                    st.session_state["processed_image"] = directedEdgeDetection(img_array_gray, DIRECTION_MAPPING[params['direction']])
                case "Sharpening":
                    st.session_state["processed_image"] = sharpening(img_array_gray)
                case "Directed Sharpening":
                    st.session_state["processed_image"] = directedSharpening(img_array_gray, DIRECTION_MAPPING[params['direction']])
                case "Fourier Transform":
                    st.session_state["processed_image"], st.session_state["fourier"] = dft(img_array_gray)
                case "Inverse Fourier Transform":
                    st.session_state["processed_image"] = inverseDft(st.session_state["fourier"])
                case "Ideal Filter":
                    st.session_state["processed_image"] = idealFilter(img_array_gray, params['d0'], FILTER_MODE_MAPPING[params['mode']])
                case "Butterworth Filter":
                    st.session_state["processed_image"] = butterworthFilter(img_array_gray, params['n'], params['d0'], FILTER_MODE_MAPPING[params['mode']])
                case "Gaussian Filter":
                    st.session_state["processed_image"] = gaussianFilter(img_array_gray, params['d0'], FILTER_MODE_MAPPING[params['mode']])
                case "Rank Order Filter":
                    st.session_state["processed_image"] = rankOrderFilter(img_array_gray, params['kernel_size'], params['mode'].lower())
            old_uploaded_file = uploaded_file

        st.markdown("#### Processed Image" if st.session_state["processed_image"] is not None else "#### Uploaded Image")
        st.image(
            st.session_state["processed_image"].astype(np.uint8) if st.session_state["processed_image"] is not None else img_array,
            use_container_width=True
        )
        
        if st.session_state["processed_image"] is not None:
            buffer = BytesIO()
            processed_image_pil = Image.fromarray(st.session_state["processed_image"])
            processed_image_pil.save(buffer, format="PNG")
            st.download_button(
                label="Save Image",
                data=buffer.getvalue(),
                file_name="processed_image.png",
                mime="image/png",
            )
