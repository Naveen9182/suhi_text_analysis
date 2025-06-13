import numpy as np
import pandas as pd
import nltk
import os
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, recall_score, roc_curve, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from xgboost import XGBClassifier
from transformers import BertModel, BertTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
import joblib
joblib.parallel_backend('loky', inner_max_num_threads=1)
import matplotlib
import torch
matplotlib.use('Agg')

# NLTK setup
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')



# Global parameters
patient_subset = 1   # 1 - All patients, 2 - Patients with Notes, 3 - Patients without Notes
feature_subset = 1   # 1 - Demo, 2 - Demo + Tabular, 3 - Demo + Tabular + Notes
feat_sel = 0         # 1 - Feature selection on (RandomForest), 0 = off
test_split_ratio = 0.2
summarized = 1       # 0 - No summarization, 1 - DeepSeek LLM based summarization
seed = 0
vectorize_text = 1   # 1 - TF-IDF, 2 - BERT
n_component = 50     # Number of components to retain from PCA

file_path = '../Data/Fewshot_new_final_summary.xlsx'
output_csv = 'training_log.csv'


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
min_max_scaler = MinMaxScaler()


###############################################################################
# Utility function: Append a single row (dict) to CSV, with an auto-incremented Run ID
###############################################################################
def append_results_to_csv(results_dict, csv_file):
    # Check if the file exists
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
        if 'Run ID' in df_existing.columns:
            max_run_id = df_existing['Run ID'].max()
        else:

            max_run_id = 0
        new_run_id = max_run_id + 1
        results_dict['Run ID'] = new_run_id
            
        # Append the new row
        df_new = pd.DataFrame([results_dict])
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(csv_file, index=False)
    else:
        # If no file, create a new DataFrame
        results_dict['Run ID'] = 1
        df_new = pd.DataFrame([results_dict])
        df_new.to_csv(csv_file, index=False)

###############################################################################
# Preprocessing function
###############################################################################
def preprocess(suhi_df):
    # Drop columns
    # suhi_df.drop(columns=['record_id', 'new_patient'], inplace=True, errors='ignore')
    # Remove rows with NaN in day_readmit
    suhi_df.dropna(subset=['day_readmit'], inplace=True)
    # Convert day_readmit == 2 to 0
    suhi_df.loc[suhi_df['day_readmit'] == 2, 'day_readmit'] = 0
    # Convert day_readmit to int
    suhi_df['day_readmit'] = suhi_df['day_readmit'].astype(int)                    
    return suhi_df


# # include words that satisfy token_pattern=r'[a-zA-Z]{2,}'
def filter_tokens_in_notes(notes):
    pattern = re.compile(r'[a-zA-Z]{2,}')
    filtered_notes = []
    for note in notes:
        # Find all tokens that match the pattern
        filtered_tokens = pattern.findall(note)
        # Join tokens back to form the filtered note
        filtered_notes.append(' '.join(filtered_tokens))
    return filtered_notes



for patient_subset in tqdm(range(1,3)):
    for feature_subset in tqdm(range(2, 4)):
        for engaged in tqdm(range(0,2)):
          for seed in tqdm(range(0,30,1)):
            for summarized in tqdm(range(0,2)):
              for vectorize_text in tqdm(range(1,3)):
                try:
                    # Load data
                    suhi_df = pd.read_excel(file_path)
                    suhi_df = suhi_df[suhi_df['engaged']==engaged]

                    # If feature_subset == 1 -> only Demographic (first 13 columns)
                    if (feature_subset == 1 or feature_subset == 2) and  summarized==1 and vectorize_text==2:
                        continue
                    # # If we only want patients with notes
                    if patient_subset == 2:
                        # suhi_df['COMBINED_NOTES'] = suhi_df['COMBINED_NOTES'].replace('', np.nan)
                        suhi_df.dropna(subset=['COMBINED_NOTES'], inplace=True)


                    if summarized == 1:
                        if feature_subset == 2:
                            continue
                        else: 
                            suhi_df['FEW_SHORT_LLM_SUMMARY'] = suhi_df['FEW_SHORT_LLM_SUMMARY'].replace('nan', '')
                            suhi_df['COMBINED_NOTES'] = suhi_df['FEW_SHORT_LLM_SUMMARY']
            
                    if patient_subset == 3 and feature_subset == 3:
                        continue

                    print('Process Begins')





                

                    ###############################################################################
                    # Main Script Execution
                    ###############################################################################
                    # Preprocess
                    suhi_df = preprocess(suhi_df)

                    # If we include text features and vectorize them using TF-IDF
                    if summarized==0:
                        min_df=20
                    else:
                        min_df=10
                    
                    if feature_subset == 3 and vectorize_text == 1:
                        tfidf_vectorizer = TfidfVectorizer(
                            min_df= min_df, 
                        )
                        suhi_df['COMBINED_NOTES'].fillna('', inplace=True)
                        suhi_df['COMBINED_NOTES'] = filter_tokens_in_notes(suhi_df['COMBINED_NOTES'])
                        text_embeddings = tfidf_vectorizer.fit_transform(suhi_df['COMBINED_NOTES'])
                
                    # If we include text features and vectorize them using BERT embeddings
                    if patient_subset == 2 and feature_subset == 3 and vectorize_text == 2:
                        # Load pre-trained BERT model
                        model = BertModel.from_pretrained('bert-base-uncased')
                        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                        model.eval()
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        model.to(device)

                        suhi_df['COMBINED_NOTES'] = filter_tokens_in_notes(suhi_df['COMBINED_NOTES'])

                        text_embeddings = []
                        # Tokenize and encode the text
                        for text in tqdm(suhi_df['COMBINED_NOTES'].tolist()):
                            tokens = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
                            tokens = {key: value.to(device) for key, value in tokens.items()}
                            with torch.no_grad():
                                text_embedding = model(**tokens)
                            text_embeddings.append(text_embedding.pooler_output.cpu().squeeze().numpy())
                        text_embeddings = np.array(text_embeddings)
                        pca = PCA().fit(text_embeddings)
                        pca = PCA(n_components=n_component) 
                        text_embeddings = pca.fit_transform(text_embeddings)

                    elif patient_subset == 1 and feature_subset == 3 and vectorize_text == 2:
                        continue

                

                    print('Dropping Text and Columns')
                    # Drop textual/object columns (except for combining them if we do text vectorizing)
                    text_columns = [col for col in suhi_df.columns if suhi_df[col].dtype == 'object']
                    suhi_df.drop(columns=text_columns, inplace=True, errors='ignore')

                    # Drop date columns
                    date_columns = [col for col in suhi_df.columns if suhi_df[col].dtype == 'datetime64[ns]']
                    suhi_df.drop(columns=date_columns, inplace=True, errors='ignore')

                    # Drop columns that contain 'nores'
                    nores_columns = [col for col in suhi_df.columns if 'nores' in col]
                    suhi_df.drop(columns=nores_columns, inplace=True, errors='ignore')

                    # If we have vectorized text, merge them in
                    if feature_subset == 3 and vectorize_text == 1:
                        COMBINED_NOTES_vectorized_df = pd.DataFrame(text_embeddings.toarray())
                        COMBINED_NOTES_vectorized_df.columns = tfidf_vectorizer.get_feature_names_out()
                        suhi_df.reset_index(drop=True, inplace=True)
                        suhi_w_vectors_df = pd.concat([suhi_df, COMBINED_NOTES_vectorized_df], axis=1)

                    elif feature_subset == 3 and vectorize_text == 2:
                        COMBINED_NOTES_vectorized_df = pd.DataFrame(text_embeddings)
                        suhi_df.reset_index(drop=True, inplace=True)
                        suhi_w_vectors_df = pd.concat([suhi_df, COMBINED_NOTES_vectorized_df], axis=1)
                    else:
                        suhi_w_vectors_df = suhi_df

                    suhi_w_vectors_df.columns = suhi_w_vectors_df.columns.astype(str)


                    # Fill NaN with 0
                    suhi_w_vectors_df.fillna(0, inplace=True)
                    print(100*'-')
                    print(patient_subset, feature_subset)
                    print(suhi_w_vectors_df.shape)
                    print(suhi_w_vectors_df.columns)
                    # Split data
                    X = suhi_w_vectors_df.drop('day_readmit', axis=1)
                    y = suhi_w_vectors_df['day_readmit']

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_split_ratio, random_state=seed
                    )

                    ###############################################################################
                    # We'll collect all our results in one dictionary, final_results,
                    # so we only write one row per script run.
                    ###############################################################################
                    final_results = {}
                    final_results["Patient Subset"] = patient_subset
                    final_results["Feature Subset"] = feature_subset
                    final_results["Feature Selection"] = bool(feat_sel)
                    final_results["Engaged"] = engaged
                    final_results["Random Seed"] = seed
                    final_results["Summarized"] = summarized
                    final_results["Vectorization"] = vectorize_text
                    final_results["File Name"] = file_path
                    final_results["Shape"] = suhi_w_vectors_df.shape
                    final_results["Columns"] = suhi_w_vectors_df.columns


                    ###############################################################################
                    # Best hyper-parameters from previous analysis
                    ###############################################################################
                    BEST_PARAMS = {
                        "RandomForestClassifier": {
                            "n_estimators": 200,
                            "max_depth": 5,
                            "min_samples_split": 20,
                            "random_state": seed,
                        },
                        "AdaBoostClassifier": {
                            "n_estimators": 100,
                            "algorithm": "SAMME",
                            "learning_rate": 0.10,
                            "random_state": seed,
                        },
                        "XGBClassifier": {
                            "n_estimators": 10,
                            "max_depth": 5,
                            "learning_rate": 0.10,
                            "random_state": seed,       
                        },
                    }

                    ###############################################################################
                    # Instantiate, train, and evaluate each model
                    ###############################################################################
                    models = {
                        "RandomForest": RandomForestClassifier(**BEST_PARAMS["RandomForestClassifier"]),
                        "AdaBoost":     AdaBoostClassifier(**BEST_PARAMS["AdaBoostClassifier"]),
                        "XGBoost":      XGBClassifier(**BEST_PARAMS["XGBClassifier"]),
                    }

                    results = {}

                    for clf_name, model in models.items():
                        # ---- train ----
                        model.fit(X_train, y_train)

                        # ---- predictions ----
                        y_pred  = model.predict(X_test)
                        y_proba = model.predict_proba(X_test)[:, 1]   


                        best_params = model.get_params()

                        feature_importances = model.feature_importances_
                        final_results["Feature Importances"] = feature_importances

                        # Predict on test set
                        y_pred = model.predict(X_test)
                        y_pred_proba = model.predict_proba(X_test)[:, 1]

                        test_accuracy = round(accuracy_score(y_test, y_pred), 4)
                        test_roc_auc = round(roc_auc_score(y_test, y_pred_proba), 4)
                        y_pred_thresholded = (y_pred_proba >= 0.35).astype(int)
                        test_sensitivity = round(recall_score(y_test, y_pred_thresholded), 4)
                        test_specificity = round(recall_score(y_test, y_pred_thresholded, pos_label=0), 4)
                        clf_report = classification_report(y_test, y_pred_thresholded)

                        # Store results for this classifier
                        final_results[f"{clf_name}_Test_Accuracy"] = test_accuracy
                        final_results[f"{clf_name}_Test"] = test_roc_auc
                        final_results[f"{clf_name}_Sensitivity"] = test_sensitivity
                        final_results[f"{clf_name}_Specificity"] = test_specificity
                        # final_results[f"{clf_name}_Classification_Report"] = clf_report

                    ###############################################################################
                    # Append exactly ONE row for this entire run
                    ###############################################################################
                    append_results_to_csv(final_results, output_csv)
                    print(f"\nDone! Logged results to {output_csv} as a single row.\n")
                except Exception as e:
                    print(f"Error occurred: {e}")
                    print(patient_subset, feature_subset, engaged, seed, feat_sel, summarized, vectorize_text)
                    pass