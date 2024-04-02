# Load necessary libraries
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import xgboost as xgb

# set the data path
data_path = Path.cwd().parent/'datas'

# Load the data
df_all = pd.read_csv(data_path / 'seqs_anno.csv')

# Load the data to be removed
cols_to_read = ["Name", "Accession", "Release_Date"]
remove_data = pd.read_csv(data_path / 'arbovirus_cleaned.csv', usecols=cols_to_read)
remove_data["Release_Date"] = pd.to_datetime(remove_data["Release_Date"], errors='coerce')  # Convert date column to datetime format
after_2022 = remove_data[remove_data["Release_Date"].dt.year >= 2022]  # Filter data submitted after 2022
accessions_after_2022 = after_2022["Accession"].tolist()

df_all_22 = df_all[df_all['Accession'].isin(accessions_after_2022)] # Filter out 2022 data

vector_22 = df_all_22[df_all_22["anno"] == "vector"] # remove vector Accession numbers
vector_22_list = vector_22.Accession.tolist()  # Get vector Accession list

# Exclude vector accessions for training and testing dataset
df = df_all[~df_all['Accession'].isin(vector_22_list)]   # 71,395


# check the positive class ratio
positive_class_ratio = (df['homo_infected'] == 1).sum() / len(df)

# Calculate scale_pos_weight
scale_pos_weight = ((1 - positive_class_ratio) / positive_class_ratio)

# Select features
X, y = df.iloc[:, list(range(1, 34))], df['homo_infected']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2, stratify=y)

# Create DMatrix
dtrain = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest = xgb.DMatrix(X_test, y_test, enable_categorical=True)

# set paramters
params = {"objective": "binary:logistic", 
          "tree_method": "exact",
          "scale_pos_weight" : scale_pos_weight,
          "eta":0.15,
          }
    
# Number of boosting rounds
n_rounds = 10000
cv_results = xgb.cv(
    params, dtrain,
    num_boost_round=n_rounds,
    nfold=5,
    early_stopping_rounds=20,
    metrics="auc",
    as_pandas=True,
    seed=42
)

# Determine the best number of iterations
best_iteration = cv_results.shape[0]

# Train the final model
final_model = xgb.train(params, dtrain, num_boost_round=best_iteration)

# Correct approach to create a directory and save the model
model_dir = data_path.parent / "models"
model_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

model_file_path = model_dir / "XGB_cli.json"  # Full path for the model file

final_model.save_model(str(model_file_path))  # Save the model


