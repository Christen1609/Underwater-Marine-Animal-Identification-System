# Underwater-Marine-Animal-Identification-System
This project, Underwater Marine Animal Classification using CNNs, leverages VGG16 to classify over 30 aquatic species from underwater images. With >70% accuracy, it enables automated identification, conservation monitoring, and ecological research, supported by a Streamlit app for real-time deployment


## Project Overview  
This project automates the classification of underwater marine species using a Convolutional Neural Network (CNN). Built on the VGG16 architecture, the system can identify and classify over **30 classes of aquatic life** from images with an accuracy of **70%+**.  

A **Streamlit app** provides an interactive interface, allowing users to upload underwater images and receive species predictions, along with additional information about the identified species.  

The work is part of the **Final Year Project** in Computer Science & Engineering and demonstrates the application of deep learning for ecological monitoring and conservation.  

---

## Repository Contents  
- **trained_model.h5** – Pre-trained VGG16 model used for classification.  
- **Image_recog_Project_Dataset_30Classes.csv** – Annotated dataset with 30 marine species classes.  
- **Validation_Set/** – Directory containing validation images for label mapping.  
- **app.py** – Streamlit application for species classification and information retrieval.  
- **Project Paper** – Full project documentation, including methodology, results, and references.  

---

## Project Workflow  
1. **Data Collection** – Underwater images of marine species compiled, annotated, and split into train/validation/test sets.  
2. **Model Training** – VGG16 CNN fine-tuned for marine classification; trained model saved as `trained_model.h5`.  
3. **Prediction Pipeline** –  
   - Preprocess images to 224×224 format  
   - Model inference for class prediction  
   - Map prediction to species label  
   - Retrieve species details from CSV  
4. **Deployment** – Streamlit app built for real-time usage.  

---

## Results  
- **Accuracy:** ~70% overall on validation set.  
- **Classes:** 30 species successfully recognised.  
- **Deployment:** Streamlit app delivers real-time predictions with contextual species info.  

---

## How to Use This Repository  
1. Clone the repository and ensure dependencies are installed (`TensorFlow`, `Keras`, `Streamlit`, `OpenCV`, `Pandas`).  
2. Run the application:  
   ```bash
   streamlit run app.py
3. Enter the file path of an underwater image in the app interface.

4. The model will display the predicted species name and related information from the dataset.   
   
