from flask import Flask, request, jsonify
from flask_cors import CORS
import main as pipeline # Import main.py and give it an alias 'pipeline'
import sys

# --- Create the Flask App ---
app = Flask(__name__)
CORS(app)

# --- Load All Models ONCE when the server starts ---
print("Initializing backend server...")
if not pipeline.load_all_models():
    sys.exit("Could not start server: Failed to load models.")

# --- API Endpoint for Analysis ---
@app.route('/analyze', methods=['POST'])
def analyze_sequence():
    """
    Receives a gene sequence, processes it through the full pipeline,
    and returns the analysis results.
    """
    data = request.get_json()
    if not data or 'sequence' not in data:
        return jsonify({'error': 'No sequence provided.'}), 400

    sequence = data.get('sequence', '').strip().upper()

    try:
        results = pipeline.run_full_pipeline(sequence)
        return jsonify(results)
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        return jsonify({'error': 'An internal server error occurred during analysis.'}), 500

# --- MODIFIED: API Endpoint for getting the raw PDB data ---
@app.route('/visualize-protein', methods=['POST'])
def visualize_protein():
    """
    Receives a gene type, fetches the raw PDB file data, and returns it.
    """
    data = request.get_json()
    if not data or 'geneType' not in data:
        return jsonify({'error': 'No gene type provided for visualization.'}), 400
    
    gene_type = data.get('geneType')

    try:
        # Call the function from main.py to get the raw PDB text data
        pdb_data = pipeline.get_protein_pdb_data(gene_type)
        return jsonify({'pdbData': pdb_data})

    except (ValueError, RuntimeError) as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"An error occurred during protein visualization: {e}")
        return jsonify({'error': 'An internal server error occurred during visualization.'}), 500

# --- Run the App ---
if __name__ == '__main__':
    print("Backend server is running on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
