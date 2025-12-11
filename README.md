# Image Retrieval System

This project implements a multi-level k-means vocabulary tree for image retrieval using SIFT, ORB, and AKAZE descriptors. The system builds a hierarchical visual vocabulary, generates TF–IDF weighted histograms, and retrieves the closest matching images based on descriptor similarity.

## How to Run
1. Build the project in Visual Studio 2022.
2. Place your dataset in the appropriate folder (as specified inside the code).
3. Run the executable to build the vocabulary tree and perform retrieval.
4. Evaluation metrics such as Precision@10 and mAP are computed automatically.

## Project Features
- Multi-level k-means vocabulary tree  
- Support for SIFT, ORB, and AKAZE descriptors  
- TF–IDF weighted histograms  
- L2 normalization  
- Image retrieval and scoring  
- Evaluation on Caltech-101 dataset  

## Notes
This repo is for academic use and experimentation with classical computer-vision retrieval methods.

## Author
Athi K.
