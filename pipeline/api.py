import requests
import json
from rdkit import Chem
from rdkit.Chem import AllChem
import os


API_KEY = "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"  # Make sure this is accurate!

url = "https://health.api.nvidia.com/v1/biology/nvidia/molmim/generate"

payload = {
    "smi": "CCO",           # Example SMILES string; replace with your own
    "algorithm": "none",    # Choose algorithm: "none" for random sampling
    "num_molecules": 5,
    "particles": 8,
    "scaled_radius": 1.0
}

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

response = requests.post(url, headers=headers, json=payload)

print("Status Code:", response.status_code)
print("Raw Response Text:\n", response.text)

try:
    result = response.json()
    response_json = json.loads(json.dumps(result))  # ensures dict format
    print("✅ Parsed JSON:")
    print(json.dumps(response_json, indent=2))
except Exception as e:
    print("❌ JSON parse error:", e)

# Parse molecules list
molecules = json.loads(response_json["molecules"])

# Create output folder
os.makedirs("molecules_3d", exist_ok=True)

for idx, mol_data in enumerate(molecules, start=1):
    smiles = mol_data["sample"]
    score = mol_data["score"]

    try:
        # Convert SMILES to molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"❌ Could not parse SMILES: {smiles}")
            continue

        # Add hydrogens
        mol = Chem.AddHs(mol)

        # Generate 3D coordinates
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)

        # Save as PDB file
        filename = f"molecules_3d/molecule_{idx}.pdb"
        Chem.MolToPDBFile(mol, filename)

        print(f"✅ Saved {filename} (score={score})")

    except Exception as e:
        print(f"⚠️ Error with {smiles}: {e}")
    
    import py3Dmol

# Load one of your saved PDB files
pdb_file = "molecules_3d/molecule_1.pdb"
with open(pdb_file, "r") as f:
    pdb_data = f.read()

# Create 3Dmol viewer
view = py3Dmol.view(width=400, height=400)
view.addModel(pdb_data, "pdb")
view.setStyle({'stick': {}})
view.zoomTo()
view.show()
