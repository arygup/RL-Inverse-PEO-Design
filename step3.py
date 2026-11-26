# ==============================================================================
# Step 3: High-Level Outline for Building Standardized `data.csv`
# ==============================================================================
# Overview of how to construct `data.csv` standardized to:
#   - Temperature: 35 °C
#   - Salt concentration: 5 mol kg^-1
#
# Refer to `Chem-prop-pred/` and
# `Chem-prop-pred/README.md` for data formatting and `ChemArr` model usage.
# ==============================================================================

import pandas as pd
from chemprop.args import PredictArgs
from chemprop.train import make_predictions

RAW_DATA_PATH = "<ADD_PATH_TO_SOURCE_DATASET>"
MODEL_CHECKPOINT_DIR = "<ADD_PATH_TO_CHEMARR_MODEL_DIR>"

REF_TEMP = 35.0      # Reference temperature (°C)
REF_MOLALITY = 5.0   # Reference salt concentration (mol kg^-1)


# 1. Start by loading the source dataset and making a clean list of
#    polymer-salt SMILES formulations set to the reference state (35 °C, 5 mol kg^-1)
raw_df = pd.read_csv(RAW_DATA_PATH)
# Clean / filter SMILES entries here as needed (see `Chem-prop-pred`)
clean_df = pd.DataFrame({
    "smiles": raw_df["smiles"].drop_duplicates(),
    "temperature": REF_TEMP,
    "molality": REF_MOLALITY,
    "preds": None,
})


# 2. Fill / replace conductivity values directly where real experimental data
#    at the reference condition (35 °C, 5 mol kg^-1) is already available
# (Match formulations with available measurements at REF_TEMP and REF_MOLALITY)


# 3. Run the pretrained Chem-prop-pred (ChemArr) model on the remaining list
#    where experimental data at 35 °C and 5 mol kg^-1 is unavailable
#    (Prepare test SMILES & features files as expected by Chem-prop-pred)
pred_args = [
    "--test_path", "<ADD_PATH_TO_MISSING_INPUTS>",
    "--features_path", "<ADD_PATH_TO_MISSING_FEATURES>",
    "--checkpoint_dir", MODEL_CHECKPOINT_DIR,
    "--arr_vtf", "arr",
    "--preds_path", "<ADD_PATH_TO_PREDICTIONS_OUTPUT>",
]
make_predictions(args=PredictArgs().parse_args(pred_args))

# Fill the missing entries in `clean_df["preds"]` with the ChemArr estimates


# 4. Export final standardized dataset (`data.csv`) for Step 4 MLP training
clean_df[["smiles", "preds"]].to_csv("data.csv", index=False)
