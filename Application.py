import streamlit as st
import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import shap

# Define cancer classes and their corresponding labels
classes = {
    "Brain Cancer": {0: "Glioma", 1: "Meningioma", 2: "Pituitary Tumor"},
    "Breast Cancer": {0: "Benign", 1: "Malignant"},
    "Cervical Cancer": {
        0: "Dyskeratotic",
        1: "Koilocytotic",
        2: "Metaplastic",
        3: "Parabasal",
        4: "Superficial-Intermediat",
    },
    "Kidney Cancer": {0: "Normal", 1: "Tumor"},
    "Lung and Colon Cancer": {
        0: "Colon Adenocarcinoma",
        1: "Colon Benign Tissue",
        2: "Lung Adenocarcinoma",
        3: "Lung Benign Tissue",
        4: "Lung Squamous Cell Carcinoma",
    },
    "Lymphoma": {
        0: "Chronic Lymphocytic Leukemia",
        1: "Follicular Lymphoma",
        2: "Mantle Cell Lymphoma",
    },
    "Oral Cancer": {0: "Normal", 1: "Oral Squamous Cell Carcinoma"},
}

def load_background_batch():
    # Function to load background images for SHAP explanation
    test_dir = "./test"
    batch_data = []

    for dirc in os.listdir(test_dir):
        dir_path = os.path.join(test_dir, dirc)
        image_files = os.listdir(dir_path)

        background_data = []

        for img_file in image_files:
            img_path = os.path.join(dir_path, img_file)
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            background_data.append(img_array)

        background_batch = np.vstack(background_data)

        batch_data.append(background_batch)

    return batch_data

def load_all_models():
    # Function to load all models
    models_list = []

    for each_model in os.listdir("./models"):
        model = load_model(f"./models/{each_model}", compile=False)
        models_list.append(model)

    return models_list

def predict_class(img, model):
    # Function to predict the class of an image
    img = Image.open(img)

    img = img.resize((224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]  # Get the index of the max predicted class
    return predictions, img, predicted_class_idx


def home():
    st.markdown(
        """
        <style>
        /* Style for main headings */
        .main-heading {
            font-size: 32px;
            color: blue;
            margin-bottom: 20px;
            font-weight:bold;
        }

        /* Style for subheadings */
        .subheading {
            font-size: 24px;
            color: #1a73e8;
            margin-top: 20px;
            margin-bottom: 10px;
            font-weight:bold;
        }

        /* Style for paragraphs */
        .paragraph {
            font-size: 18px;
            color: black;
            line-height: 1.5;
            font-weight:bold;
            margin-bottom: 20px;
        }

        /* Style for images */
        .project-images-container {
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
        }

        .project-image {
            width: 30%;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("Welcome to Cancer Classification App")
    
    st.header("Understanding Cancer and its Classifications")

    st.markdown(
        """
        Cancer is a complex group of diseases characterized by the uncontrolled growth and spread of abnormal cells. 
        These cells can invade and destroy normal tissue, leading to serious health problems or even death if not 
        detected and treated early.

        ## What is Cancer?
        
        Cancer begins when normal cells undergo genetic mutations that cause them to grow and multiply uncontrollably. 
        These mutated cells can form a mass of tissue known as a tumor, which can be either benign (non-cancerous) 
        or malignant (cancerous). Malignant tumors have the potential to invade nearby tissues and spread to other 
        parts of the body, a process known as metastasis.
        
        ## Classifications of Cancer
        
        Cancer can be classified into various types based on the location of the tumor, the type of cells involved, 
        and other factors. Some common classifications of cancer include:
        """
    )

    st.subheader("1. By Organ or Tissue Type")
    st.markdown(
        """
        - **Breast Cancer:** Occurs in the breast tissue, most commonly in women but can also affect men.
        - **Lung Cancer:** Develops in the lungs and is often associated with smoking or exposure to carcinogens.
        - **Colon Cancer:** Affects the colon or rectum and usually starts as benign polyps that become cancerous over time.
        """
    )

    st.subheader("2. By Cell Type")
    st.markdown(
        """
        - **Adenocarcinoma:** Arises from glandular cells and can occur in various organs such as the breast, prostate, or lung.
        - **Squamous Cell Carcinoma:** Originates from squamous cells found in the skin, lungs, or other organs.
        - **Leukemia:** A type of blood cancer that affects the bone marrow and blood cells.
        """
    )

    st.subheader("3. By Stage of Development")
    st.markdown(
        """
        - **Stage 0:** Cancer in situ, meaning it has not spread beyond the original site.
        - **Stage I-IV:** Indicates the extent of cancer spread, with Stage IV being the most advanced.
        """
    )

    st.subheader("4. By Molecular Characteristics")
    st.markdown(
        """
        - **HER2-positive:** Cancer cells with high levels of human epidermal growth factor receptor 2 (HER2), 
          often found in breast cancer.
        - **Triple-negative:** Breast cancer cells that lack estrogen receptors, progesterone receptors, and HER2 
          receptors, making them less responsive to standard treatments.
        """
    )

    st.subheader("5. By Treatment Approach")
    st.markdown(
        """
        - **Surgery:** Removal of the tumor or affected tissue.
        - **Chemotherapy:** Use of drugs to kill cancer cells or slow their growth.
        - **Radiation Therapy:** Targeted radiation to destroy cancer cells.
        - **Immunotherapy:** Boosting the body's immune system to fight cancer.
        """
    )

    st.write(
        """
        Understanding the different classifications of cancer is crucial for accurate diagnosis, prognosis, 
        and treatment planning. Advances in medical research and technology continue to improve our ability 
        to detect and treat cancer effectively, leading to better outcomes for patients worldwide.
        """
    )

    st.header("Project Overview")

    st.markdown(
        """
        This Cancer Classification App is developed to aid in the diagnosis and classification of various types 
        of cancer using machine learning techniques. By analyzing medical images, such as X-rays, MRI scans, or 
        histopathology slides, the app can assist healthcare professionals in identifying cancerous lesions and 
        determining the type and stage of cancer.

        ## Role of Machine Learning
        
        Machine learning plays a crucial role in cancer detection and classification by leveraging algorithms 
        to analyze large volumes of medical data and extract meaningful patterns and features. These algorithms 
        can learn from labeled datasets of medical images, allowing them to accurately identify and classify 
        cancerous lesions based on visual characteristics.

        ## Key Features of the App

        - **Image Upload:** Users can upload medical images, such as X-rays or MRI scans, for classification.
        - **Multi-Class Classification:** The app supports the classification of various types of cancer, 
          including breast cancer, lung cancer, colon cancer, and more.
        - **Explanation with SHAP:** The app provides explanations for the model's predictions using SHAP (SHapley 
          Additive exPlanations), allowing users to interpret and understand the model's decision-making process.

        ## Future Enhancements

        As part of ongoing development, we aim to further improve the accuracy and interpretability of the 
        classification models, integrate additional features for enhanced user experience, and collaborate 
        with healthcare professionals to validate the app's performance in real-world clinical settings.
        """
    )

def main():
    
    st.markdown(
        """
        <style>
        .stApp {
            font-family: sans-serif;
            background-image: url('https://www.electropages.com/storage/app/media/2023/1.%20January/artificial-intelligence-in-healthcare-og.png'); /* Background image */
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
            background-color: black; /* Fallback background color */
        }
        .sidebar .sidebar-content {
            background-color:rgb(0,0,0,0.5);
        }
        .sidebar .sidebar-content .block-container {
            border-radius: 10px;
            margin: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 1);
        }
        .main .block-container {
            border-radius: 10px;
            margin: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            background-color:rgb(0,0,0,0.5);
            margin-top:5rem;
        }
        .main .stButton > button {
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            transition-duration: 0.4s;
        }
        .main .stButton > button:hover {
            background-color: #45a049;
        }
        .st-c9 {
            color: white;
        }
        .st-emotion-cache-6qob1r {
            background-color:rgb(0,0,0,0.5);
            border: 2px solid white;
            font: 20px; /* Adjust text size */
        }
        .st-emotion-cache-10trblm {
            color: white;
            font-family: serif;
            font-weight: 25px; /* Adjust text size */
        }
        .upload-section {
                background-color: rgb(255, 255, 0, 0.5);
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
            }
      
        </style>
        """,
        unsafe_allow_html=True
    )
    # Main function to run the Streamlit app
    models_list = load_all_models()
    background_batch_data = load_background_batch()

    cancer_classes = list(classes.keys())

    st.sidebar.title("CancerDetect Pro")
    page = st.sidebar.radio("Choose Your Option", ["Home", "Image Classification"])

    if page == "Home":
        home()
    elif page == "Image Classification":
        st.title("Image Classification App")
        


        uploaded_file = st.file_uploader("Choose an image...", type="jpg")
        choice = st.selectbox(
            "Select Cancer Type", options=list(classes.keys())
        )

        if uploaded_file is not None and st.button("Predict"):
            st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
            st.write("")
            st.write("Classifying...")

            model = models_list[cancer_classes.index(choice)]
            background_batch = background_batch_data[cancer_classes.index(choice)]

            predictions, img, predicted_class_idx = predict_class(uploaded_file, model)

            shap_values = shap_explanation(model, img, background_batch)

            show_shap(
                shap_values, img, predicted_class_idx, classes[choice]
            )

if __name__ == "__main__":
    main()
