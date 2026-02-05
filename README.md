# Gene-Corrector AI 🧬

**Gene-Corrector AI** is a machine learning-powered web application designed to analyze gene sequences, classify them, detect mutations, and generate corrected sequences using Generative AI models.

## 🚀 Features

- **Gene Classification**: Automatically identifies if a sequence belongs to the **CFTR** or **DSCAM** gene family.
- **Mutation Detection**: Detects if the provided gene sequence contains mutations.
- **AI-Powered Correction**: Uses Encoder-Decoder (Seq2Seq) models to generate a corrected version of the gene sequence if a mutation is found.
- **Protein Visualization**: Fetches raw PDB data for the identified gene type to assist in protein visualization.
- **Interactive Web UI**: Simple and clean interface built with HTML/CSS and JavaScript.

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Machine Learning**: TensorFlow (Keras), Scikit-learn, Joblib
- **Frontend**: HTML5, CSS3, JavaScript
- **Deployment**: Configured for Render/Gunicorn

## 📂 Project Structure

```bash
📦 gene-corrector-ai
 ┣ 📂 ml                 # Machine learning scripts
 ┣ 📂 pipeline           # Application source code
 ┃ ┣ 📂 templates        # HTML templates
 ┃ ┣ 📜 app.py           # Flask application entry point
 ┃ ┣ 📜 main.py          # Core pipeline and model inference logic
 ┃ ┗ 📜 ...              # Saved models (.keras, .joblib, .pkl)
 ┣ 📜 Procfile           # Deployment configuration
 ┣ 📜 requirements.txt   # Python dependencies
 ┗ 📜 runtime.txt        # Python runtime version
```

## 🔧 Installation & Local Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/varaprasadkarna/Gene-corrector.ai.git
   cd Gene-corrector.ai
   ```

2. **Create a virtual environment (Optional but Recommended)**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python pipeline/app.py
   ```
   Access the app at `http://127.0.0.1:5000`

## 🌐 Deployment

This project is configured for deployment on **Render**.

1. Create a new **Web Service** on Render.
2. Link your GitHub repository.
3. Render will automatically detect the `Procfile` and `requirements.txt`.
4. Click **Deploy**.

## 🧠 Models Used

- **Logistic Regression**: For gene type and mutation status classification.
- **LSTM / Encoder-Decoder**: For sequence correction (GenAI).

## 📄 License

This project is open-source.