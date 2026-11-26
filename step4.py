# ==============================================================================
# Step 4: Surrogate MLP Conductivity Model (SMILES -> RDKit Descriptors -> MLP)
# ==============================================================================
# NOTE:
# - This script showcases how polymer electrolyte SMILES strings from `data.csv`
#   are converted into 12 RDKit molecular descriptors and used to train the
#   multi-layer perceptron (`MLPRegressor`) conductivity predictor.
# - For environment setup, descriptor/reward integration, and debugging, refer to
#   the `ReLeaSE/` repository (`ReLeaSE/README.md`, `ReLeaSE/release/`, and the
#   demo notebooks `ReLeaSE/LogP_optimization_demo.ipynb` and
#   `ReLeaSE/JAK2_min_max_demo.ipynb`).
# ==============================================================================

import joblib
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors


# ------------------------------------------------------------------------------
# 1. SMILES -> 12 RDKit Molecular Descriptors
# ------------------------------------------------------------------------------
def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Failed to parse SMILES: {smiles}")
        return None
    try:
        descriptors = [
            Descriptors.MolWt(mol),  # Molecular Weight
            Descriptors.MolLogP(mol),  # LogP (Octanol-Water Partition Coefficient)
            Descriptors.NumRotatableBonds(mol),  # Number of Rotatable Bonds
            Descriptors.TPSA(mol),  # Topological Polar Surface Area
            Descriptors.NumHDonors(mol),  # Number of Hydrogen Bond Donors
            Descriptors.NumHAcceptors(mol),  # Number of Hydrogen Bond Acceptors
            Descriptors.BalabanJ(mol),  # Balaban's J Index (topological descriptor)
            Descriptors.BertzCT(mol),  # Bertz Complexity Index
            Descriptors.FractionCSP3(mol),  # Fraction of sp3 hybridized carbons
            Descriptors.PEOE_VSA1(mol),  # Partial charge descriptors (example)
            Descriptors.NumAromaticRings(mol),  # Number of Aromatic Rings
            Descriptors.MolMR(mol)  # Molar Refractivity
        ]
    except Exception as e:
        return None
    return descriptors


# ------------------------------------------------------------------------------
# 2. Train MLP Surrogate on Standardized Conductivity Dataset (`data.csv`)
#    Architecture: 4 hidden layers (200, 150, 100, 50), max_iter=10000
#    Adjust column names in `data.csv` if needed when running on your dataset.
# ------------------------------------------------------------------------------
file_path = 'data.csv'
data = pd.read_csv(file_path)
data = data.drop(columns=['Unnamed: 1'])
X = np.array([smiles_to_descriptors(smiles) for smiles in data['smiles'] if smiles_to_descriptors(smiles) is not None])
y = data['preds'][[smiles_to_descriptors(smiles) is not None for smiles in data['smiles']]]
mlp = MLPRegressor(hidden_layer_sizes=(200, 150, 100, 50), max_iter=10000, random_state=42)
mlp.fit(X, y)
model_filename = 'mlp_conductivity_model_complex.pkl'
joblib.dump(mlp, model_filename)


# ------------------------------------------------------------------------------
# 3. Load Trained MLP & Predict Conductivity / Reward from SMILES
#    Refer to `step5.py` and the `ReLeaSE` demo notebooks for usage inside RL.
# ------------------------------------------------------------------------------
model_filename = 'mlp_conductivity_model_complex.pkl'
loaded_mlp = joblib.load(model_filename)


def get_reward_mlp(smiles, predictor, invalid_reward=0.0):
    descriptors = smiles_to_descriptors(smiles)
    if descriptors is None:
        return invalid_reward
    X = np.array(descriptors).reshape(1, -1)
    predictions = predictor.predict(X)
    predicted_value = predictions[0]
    reward = 20.0 + predicted_value
    return reward
