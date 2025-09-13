#GeneCorrect.ai: An End-to-End DNA Mutation Analysis & Correction Pipeline
<!-- Replace with a URL to your screenshot -->

GeneCorrect AI is an integrated web application designed to automate and accelerate the process of DNA sequence analysis. The platform addresses the slow and fragmented workflow of traditional genetic research by providing a single, seamless pipeline from raw DNA sequence to a final, interpretable report with 3D protein visualization.

The system utilizes a multi-stage AI pipeline, served via a Python-based REST API, to classify gene types, detect mutations, and employ a generative AI model to propose corrected, healthy versions of mutated sequences. This project showcases a practical application of AI in bioinformatics, providing researchers with a powerful tool to accelerate the study of genetic disorders.

✨ Features
Multi-Stage AI Pipeline: Sequentially processes DNA through multiple models for classification and correction.

Gene Type Classification: Identifies if a sequence belongs to the CFTR or DSCAM gene using a Logistic Regression model.

Mutation Status Detection: Classifies a sequence as "mutated" or "non-mutated" with a second high-speed classifier.

Generative AI Correction: Employs a Sequence-to-Sequence (Seq2Seq) LSTM neural network to generate a corrected, healthy version of a mutated gene.

Interactive 3D Protein Visualization: Fetches and renders the corresponding protein structure from the Protein Data Bank (PDB) using 3Dmol.js, providing critical biological context.

Dynamic PDF Report Generation: Creates a comprehensive PDF report containing all analysis results, a snapshot of the 3D model, and an AI-generated explanation of the mutation's impact using the Google Gemini API.

🛠️ Tech Stack & Architecture
The application is built with a decoupled frontend and backend architecture, communicating via a REST API.

Backend: Python, Flask, TensorFlow/Keras, Scikit-learn, Requests, FPDF

Frontend: HTML, CSS, JavaScript, Tailwind CSS, 3Dmol.js

External APIs: Google Gemini API, Protein Data Bank (PDB)

System Architecture
The data flows through a pipeline of interconnected models and services, ensuring an efficient and accurate analysis from start to finish.

<!-- Replace with a URL to your data flow diagram -->

🚀 Getting Started
Follow these instructions to set up and run the project locally.

Prerequisites
Python 3.9+

pip for package management

Installation
Clone the repository:

git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name

Set up a Python virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required Python packages:

pip install -r requirements.txt

Note: The requirements.txt file should contain all necessary libraries like Flask, TensorFlow, joblib, etc.

🏃‍♂️ Usage
To run the application, you need to start the backend server and then open the frontend interface.

Start the Backend Server:
Run the Flask application from the project's root directory. This will start the API server, which listens for requests from the frontend.

python app.py

You should see a message indicating that the server is running on http://127.0.0.1:5000.

Open the Frontend:
Open the index.html file in your web browser. The interface will now be able to communicate with your running backend server.

📸 Screenshots
Landing Page

Analysis Dashboard





3D Protein Viewer

PDF Report





<!-- Instructions: To use the screenshots, upload your images to a hosting service like Imgur and replace the placeholder URLs. -->
