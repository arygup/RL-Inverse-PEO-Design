# ==============================================================================
# Step 5: RL Fine-Tuning of ReLeaSE Generator Using the Trained MLP Surrogate
# ==============================================================================
# NOTE:
# - Run this script from within the `ReLeaSE/` directory (or update `./release/`,
#   `./data/...`, `./checkpoints/...`, and `mlp_conductivity_model_complex.pkl`
#   paths accordingly) and ensure output folders such as `plots3/` exist.
# - For generator architecture, `Reinforcement` policy-gradient internals,
#   hyperparameter changes, and debugging, refer directly to the `ReLeaSE`
#   source files (`ReLeaSE/release/stackRNN.py`, `ReLeaSE/release/data.py`,
#   `ReLeaSE/release/reinforcement.py`, `ReLeaSE/release/utils.py`) and the
#   official ReLeaSE demo notebooks (`ReLeaSE/LogP_optimization_demo.ipynb` and
#   `ReLeaSE/JAK2_min_max_demo.ipynb`).
# ==============================================================================

import sys
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm, trange
import pickle
sys.path.append('./release/')
from rdkit import Chem, DataStructs
from stackRNN import StackAugmentedRNN
from data import GeneratorData
from utils import canonical_smiles
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors
from reinforcement import Reinforcement

use_cuda = torch.cuda.is_available()

# ------------------------------------------------------------------------------
# 1. Load Pretrained ReLeaSE Stack-Augmented RNN Generator
#    Refer to `ReLeaSE/LogP_optimization_demo.ipynb` and `ReLeaSE/release/`
#    for token vocabulary, `GeneratorData`, and `StackAugmentedRNN` details.
# ------------------------------------------------------------------------------
tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7', '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']', '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']
gen_data_path = './data/chembl_22_clean_1576904_sorted_std_final.smi'
gen_data = GeneratorData(training_data_path=gen_data_path, delimiter='\t', cols_to_read=[0], keep_header=True, tokens=tokens)
hidden_size = 1500
stack_width = 1500
stack_depth = 200
layer_type = 'GRU'
lr = 0.001
optimizer_instance = torch.optim.Adadelta
my_generator = StackAugmentedRNN(input_size=gen_data.n_characters, hidden_size=hidden_size, output_size=gen_data.n_characters, layer_type=layer_type, n_layers=1, is_bidirectional=False, has_stack=True, stack_width=stack_width, stack_depth=stack_depth, use_cuda=use_cuda, optimizer_instance=optimizer_instance, lr=lr)
model_path = './checkpoints/generator/checkpoint_biggest_rnn'
map_location = torch.device('cuda' if use_cuda else 'cpu')
weights = torch.load(model_path, map_location=map_location)
my_generator.load_state_dict(weights)
model_path = './checkpoints/generator/checkpoint_biggest_rnn'
weights = torch.load(model_path, map_location=torch.device('cuda' if use_cuda else 'cpu'))
my_generator.load_state_dict(weights)


# ------------------------------------------------------------------------------
# 2. Load Trained Surrogate MLP & Descriptor Featurizer (from `step4.py`)
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


model_filename = 'mlp_conductivity_model_complex.pkl'
loaded_mlp = joblib.load(model_filename)


# ------------------------------------------------------------------------------
# 3. Evaluation, Plotting & Custom RL Reward Functions
#    Refer to `ReLeaSE/LogP_optimization_demo.ipynb` and
#    `ReLeaSE/JAK2_min_max_demo.ipynb` for reward-shaping conventions and
#    debugging invalid SMILES or distribution plots.
# ------------------------------------------------------------------------------
def plot_hist(prediction, n_to_generate, threshold_max=-5):
    prediction = np.array(prediction)
    percentage_in_threshold = np.sum((prediction >= threshold_max)) / len(prediction)
    print(f"Percentage of predictions within the threshold [{threshold_max}]: {percentage_in_threshold:.2%}")
    proportion_valid_smiles = len(prediction) / n_to_generate
    print(f"Proportion of valid SMILES: {proportion_valid_smiles:.2%}")
    ax = sns.kdeplot(prediction, shade=True, color='blue')
    plt.axvline(x=threshold_max, color='red', linestyle='--')
    ax.set(xlabel='Predicted Conductivity',
           ylabel='Density',
           title=f'Distribution of Predicted Conductivity for Generated Molecules')
    plt.show()


def estimate_and_update(generator, predictor, n_to_generate):
    generated = []
    pbar = tqdm(range(n_to_generate))
    for i in pbar:
        pbar.set_description("Generating molecules...")
        generated.append(generator.evaluate(gen_data, predict_len=120)[1:-1])
    sanitized = canonical_smiles(generated, sanitize=False, throw_warning=False)[:-1]
    unique_smiles = list(np.unique(sanitized))[1:]
    SALTS = [ "O=S([N-]S(=O)(C(F)(F)F)=O)(C(F)(F)F)=O", "O=C1C(O[B-]2(OC(C(O2)=O)=O)O1)=O", "O=S(C(F)(F)C(F)(F)F)([N-]S(=O)(C(F)(F)C(F)(F)F)=O)=O","FS([N-]S(=O)(F)=O)(=O)=O", "F[B-](F)(F)F"]
    max_conductivities = []

    for smiles in unique_smiles:
        max_conductivity = -float('inf')
        for salt in SALTS:
            new_smiles = f"{smiles}.{salt}"
            descriptors = smiles_to_descriptors(new_smiles)
            if descriptors is not None:
                X = np.array(descriptors).reshape(1, -1)
                predictions = predictor.predict(X)
                predicted_value = predictions[0]
                if predicted_value > max_conductivity:
                    max_conductivity = predicted_value
        max_conductivities.append(max_conductivity)

    max_conductivities = np.array(max_conductivities)
    plot_hist(max_conductivities, n_to_generate)
    return unique_smiles, max_conductivities


def estimate_and_update_CCO(generator, predictor, n_to_generate):
    generated = []
    pbar = tqdm(range(n_to_generate))
    for i in pbar:
        pbar.set_description("Generating molecules...")
        generated.append(generator.evaluate(gen_data, predict_len=120)[1:-1])
    sanitized = canonical_smiles(generated, sanitize=False, throw_warning=False)[:-1]
    unique_smiles = list(np.unique(sanitized))[1:]
    CCO = ['CCO' * 14]
    SALTS = ["O=S([N-]S(=O)(C(F)(F)F)=O)(C(F)(F)F)=O.[Li]"]
    max_conductivities = []
    for smiles in unique_smiles:
        max_conductivity = -float('inf')
        for cco in CCO:
            new_smiles = f"{cco}{smiles}"
            descriptors = smiles_to_descriptors(new_smiles)
            if descriptors is not None:
                for salt in SALTS:
                    new_smiles = f"{new_smiles}.{salt}"
                    descriptors = smiles_to_descriptors(new_smiles)
                    if descriptors is not None:
                        X = np.array(descriptors).reshape(1, -1)
                        predictions = predictor.predict(X)
                        predicted_value = predictions[0]
                        if predicted_value > max_conductivity:
                            max_conductivity = predicted_value
        if max_conductivity == -float('inf'):
            continue
        max_conductivities.append(max_conductivity)
    max_conductivities = np.array(max_conductivities)
    plot_hist(max_conductivities, n_to_generate)
    return unique_smiles, max_conductivities


# Alternative reward variant with explicit CCO-repeat bias across salts:
def get_reward_with_salts(smiles, predictor, invalid_reward=0.0):
    descriptors = smiles_to_descriptors(smiles)
    if descriptors is None:
        return invalid_reward
    bias = 0.0
    if i < 30:
        if "CCO" in smiles:
            bias += 10
        cco_count = smiles.count("CCO")
        bias += cco_count * 2
        return bias + 2
    max_conductivity = -float('inf')
    for salt in SALTS:
        new_smiles = f"{smiles}.{salt}"
        descriptors = smiles_to_descriptors(new_smiles)
        if descriptors is not None:
            X = np.array(descriptors).reshape(1, -1)
            predictions = predictor.predict(X)
            predicted_value = predictions[0]
            if predicted_value > max_conductivity:
                max_conductivity = predicted_value

    if max_conductivity ==  -float('inf'):
        return invalid_reward

    if max_conductivity >= - 5.0:
        to_check.append(smiles)

    bias = 0.0
    cco_count = smiles.count("CCO")
    bias += cco_count * 0.5

    return 20.0 + max_conductivity + bias


# PEO-oligomer (`CCO` repeat) + Li-salt reward function used for RL fine-tuning:
def get_reward_with_CCO(smiles, predictor, invalid_reward=0.0):
    descriptors = smiles_to_descriptors(smiles)
    if descriptors is None:
        return invalid_reward
    max_conductivity = -float('inf')
    for cco in CCO:
        new_smiles = f"{cco}{smiles}"
        descriptors = smiles_to_descriptors(new_smiles)
        if descriptors is not None:
            for salt in SALTS:
                new_smiles = f"{new_smiles}.{salt}"
                descriptors = smiles_to_descriptors(new_smiles)
                if descriptors is not None:
                    X = np.array(descriptors).reshape(1, -1)
                    predictions = predictor.predict(X)
                    predicted_value = predictions[0]
                    if predicted_value > max_conductivity:
                        max_conductivity = predicted_value
    if max_conductivity >= - 5.0:
        to_check.append(smiles)
    if max_conductivity ==  -float('inf'):
        return invalid_reward
    return 20.0 + max_conductivity


# ------------------------------------------------------------------------------
# 4. Reinforcement Learning Policy-Gradient Loop & Saving High-Cond Candidates
#    Refer to `ReLeaSE/release/reinforcement.py` and the ReLeaSE demo notebooks
#    for modifying `n_iterations`, `n_policy`, `n_to_generate`, or checkpoints.
# ------------------------------------------------------------------------------
# CCO = ['CCO' * i for i in range(14, 71, 7)]
CCO = ['CCO' * 14]
SALTS = ["O=S([N-]S(=O)(C(F)(F)F)=O)(C(F)(F)F)=O.[Li]"] #, "O=C1C(O[B-]2(OC(C(O2)=O)=O)O1)=O.[Li]", "O=S(C(F)(F)C(F)(F)F)([N-]S(=O)(C(F)(F)C(F)(F)F)=O)=O.[Li]", "FS([N-]S(=O)(F)=O)(=O)=O.[Li]", "F[B-](F)(F)F.[Li]"]


n_to_generate = 100
n_policy = 25
n_iterations = 151
RL_logp = Reinforcement(my_generator, loaded_mlp, get_reward_with_CCO)
to_check = []

for i in range(n_iterations):
    for j in trange(n_policy, desc='Policy gradient...'):
        RL_logp.policy_gradient(gen_data)
    sm, pred = estimate_and_update_CCO(RL_logp.generator, loaded_mlp, n_to_generate)
    plt.figure()
    plot_hist(pred, n_to_generate)
    plt.savefig(f'plots3/graph_iter_{i}.png')
    for sms in sm:
        if get_reward_with_CCO(sms, loaded_mlp) > - 5.0:
            to_check.append(sms)
    # if(i % 10 == 0):
    #     torch.save(RL_logp.generator.state_dict(), f'RL_logp_generator_{i}.pkl')

df = pd.DataFrame(to_check, columns=['smiles'])
df.to_csv('to_check_sample3.csv')
