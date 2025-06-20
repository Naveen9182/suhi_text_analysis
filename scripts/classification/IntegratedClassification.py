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
feat_sel = 0         # 1 = Feature selection on (RandomForest), 0 = off
test_split_ratio = 0.2
summarized = 1
seed = 0
vectorize_text = 1   # 1 - TF-IDF, 2 - BERT
file_path = '../Data/Fewshot_new_final_summary.xlsx'
output_csv = 'training_log.csv'
demographic_features = [ 'day_readmit',	'age', 'sex_Male', 'Hispanic', 'AfricanAmerican', 'White',                                        # 'visit_number', 'referral_date', 'hypertension' 
                       'Additional_race', 'language_English', 'language_Spanish', 'language_Other']

sdoh_features = ["sdoh_PCP", "sdoh_INS", "sdoh_HOUSING", "sdoh_FOOD",
                 "sdoh_UTIL", "sdoh_TRANS", "sdoh_EMPLOY",	"sdoh_CLOTH_CHILD_PHONE",
                "sdoh_DV",	"sdoh_HIV",	"sdoh_COVID", "sdoh_DIABETES", "sdoh_ASTHMA",
                "sdoh_BILL_FU_RX_HEALTH", "sdoh_EMOTIONAL", "sdoh_SUBSTANCE_ABUSE",
                "sdoh_SAFETY", "sdoh_HOME_EQUIP", "sdoh_LEGAL", "sdoh_OTHER"]

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
min_max_scaler = MinMaxScaler()

for patient_subset in tqdm(range(1,3)):
    for feature_subset in tqdm(range(1, 4)):
        for engaged in tqdm(range(0,2)):
          for seed in tqdm(range(0,50,1)):
            # for feat_sel in tqdm(range(0,2)):
             # if feature_subset!=3:
             #     pass
             # else:
            for summarized in tqdm(range(0,2)):
             for vectorize_text in tqdm(range(1,3)):
              for n_component in [50, 100]:

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


                    if summarized == 0:
                        if feature_subset == 1:
                            # suhi_df = suhi_df[suhi_df.columns[:13]]
                            suhi_df = suhi_df[demographic_features]
                        elif feature_subset == 2:
                            suhi_df = suhi_df[demographic_features + sdoh_features]
                        else: 
                            suhi_df['COMBINED_NOTES'].fillna('', inplace=True)
                            suhi_df = suhi_df[demographic_features + sdoh_features +  ["COMBINED_NOTES"]]

                    else:
                        if feature_subset == 1 or feature_subset == 2:
                            continue
                        #     # suhi_df = suhi_df[suhi_df.columns[:13]]
                        #     suhi_df = suhi_df[demographic_features]
                        # elif feature_subset == 2:
                        #     suhi_df = suhi_df[demographic_features + sdoh_features]
                        else: 
                            suhi_df['FEW_SHORT_LLM_SUMMARY'] = suhi_df['FEW_SHORT_LLM_SUMMARY'].replace('nan', '')
                            suhi_df = suhi_df[demographic_features + sdoh_features +  ["FEW_SHORT_LLM_SUMMARY"]]
                            suhi_df['COMBINED_NOTES'] = suhi_df['FEW_SHORT_LLM_SUMMARY']

                # for feat_sel in range(0,2):
                # for seed in range(0,50,14):
            
                    if patient_subset == 3 and feature_subset == 3:
                        continue

                    print('Process Begins')
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
                        # # Scale age
                        # suhi_df['age'] = min_max_scaler.fit_transform(suhi_df[['age']])
        
                        # # Combine note columns
                        # note_cols = suhi_df.columns[suhi_df.columns.str.contains('notes_contact')]
                        # if summarized ==0:
                        #     # COMBINED_NOTES = suhi_df[note_cols].apply(
                        #     #     lambda x: '. '.join([str(note).lower() for note in x.dropna()]),
                        #     #     axis=1)
                        #     # suhi_df['COMBINED_NOTES'] = COMBINED_NOTES
                        #     suhi_df['COMBINED_NOTES'] = suhi_df['COMBINED_NOTES'].replace('nan', np.nan)

                        # else:
                        #     suhi_df['COMBINED_NOTES'] = suhi_df['FEW_SHORT_LLM_SUMMARY']
                        #     suhi_df['COMBINED_NOTES'] = suhi_df['COMBINED_NOTES'].replace('nan', np.nan)
                    
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
                

                    ###############################################################################
                    # Main Script Execution
                    ###############################################################################
                    print('Preprocess Begins')
                    # Preprocess
                    suhi_df = preprocess(suhi_df)


                    # # If we only want patients without notes
                    # if patient_subset == 3:
                    #     suhi_df['COMBINED_NOTES'] = suhi_df['COMBINED_NOTES'].replace('', np.nan)
                    #     suhi_df = suhi_df[suhi_df['COMBINED_NOTES'].isna()]
                    #     # suhi_df['COMBINED_NOTES'].fillna('', inplace=True)


                    # print('Vectorization Begins')

                    # If we include text features and vectorize them using TF-IDF
                    if summarized==0:
                        min_df=20
                    else:
                        min_df=10
                    
                    if feature_subset == 3 and vectorize_text == 1:
                        tfidf_vectorizer = TfidfVectorizer(
                            min_df= min_df, 
                            # max_df=0.7, 
                            # stop_words='english', 
                            # token_pattern=r'[a-zA-Z]{2,}'
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
                        # Tokenize and encode the text
                        # text_embeddings = tokenizer(suhi_df['COMBINED_NOTES'].tolist(), padding=True, truncation=True, return_tensors='pt')
                        # text_embeddings = model(**text_embeddings)
                        # text_embeddings = text_embeddings.pooler_output
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
                        # plt.plot(np.cumsum(pca.explained_variance_ratio_))
                        # plt.xlabel('Number of Components')
                        # plt.ylabel('Cumulative Explained Variance')
                        # plt.grid(True)
                        # plt.show()
                        pca = PCA(n_components=n_component)  # adjust based on performance/variance explained
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

                    top_25_features = []
                    top_25_percent_features_str = ""

                    ###############################################################################
                    # Feature Selection (RandomForest) step, if enabled
                    ###############################################################################
                    if feat_sel == 1:
                        fs_rf = RandomForestClassifier(random_state=seed)
                        param_grid_rf = {
                                "n_estimators": [10, 100, 200],
                                "max_depth": [5, 10, 15],
                                "min_samples_split": [20, 50]
                        }
                        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        
                        grid_search = GridSearchCV(
                            estimator=fs_rf,
                            param_grid=param_grid_rf,
                            cv=cv,
                            scoring='roc_auc',
                            refit=True,
                            return_train_score=True,
                            n_jobs=1,
                            verbose=1
                        )
        
                        grid_search.fit(X_train, y_train)

                        # Get mean train and test scores from CV results
                        train_scores = grid_search.cv_results_['mean_train_score']
                        test_scores = grid_search.cv_results_['mean_test_score']


                        # Compute overfitting gap (train score - test score)
                        gap = train_scores - test_scores
                        print(100*'-')
                        print(gap)

                        threshold = 0.1  # Adjust based on your metric (e.g., 0.1 for accuracy, 0.05 for F1)

                        # Criteria: (1) Gap < threshold, (2) Best test score among eligible models
                        eligible_mask = gap < threshold

                        if eligible_mask.any():
                            # Get indices of eligible models
                            eligible_indices = np.where(eligible_mask)[0]
                        
                            # Subset scores and gaps for eligible models
                            eligible_test_scores = test_scores[eligible_indices]
                            eligible_gaps = gap[eligible_indices]
                        
                            # Find the best test score among eligible models
                            best_test_score = eligible_test_scores.max()
                        
                            # Among models with best_test_score, find the one with SMALLEST GAP
                            best_score_mask = (eligible_test_scores == best_test_score)
                            best_gap = eligible_gaps[best_score_mask].min()
                        
                            # Get the index of the first model with best score AND smallest gap
                            best_model_idx = eligible_indices[
                                np.where((eligible_test_scores == best_test_score) & (eligible_gaps == best_gap))[0][0]
                            ]
                        else:
                            print("No model meets the criteria")
                            best_model_idx = grid_search.cv_results_['rank_test_score'].argmin()

                        # Best/Worst model indices in the results
                        # best_model_idx = grid_search.cv_results_['rank_test_score'].argmin()
                        # worst_model_idx = grid_search.cv_results_['rank_test_score'].argmax()

                        # Retrieve relevant metrics
                        fs_best_mean_train_score = round(grid_search.cv_results_['mean_train_score'][best_model_idx], 4)
                        # fs_best_std_train_score = round(grid_search.cv_results_['std_train_score'][best_model_idx], 4)
                        fs_best_mean_test_score = round(grid_search.cv_results_['mean_test_score'][best_model_idx], 4)
                        fs_best_std_test_score = round(grid_search.cv_results_['std_test_score'][best_model_idx], 4)

                        # fs_worst_mean_train_score = round(grid_search.cv_results_['mean_train_score'][worst_model_idx], 4)
                        # fs_worst_std_train_score = round(grid_search.cv_results_['std_train_score'][worst_model_idx], 4)
                        # fs_worst_mean_test_score = round(grid_search.cv_results_['mean_test_score'][worst_model_idx], 4)
                        # fs_worst_std_test_score = round(grid_search.cv_results_['std_test_score'][worst_model_idx], 4)
        
                        # Best estimator
                        best_rf = grid_search.best_estimator_
                        best_params_fs = best_rf.get_params()

                        # Evaluate on test set
                        y_pred = best_rf.predict(X_test)
                        y_pred_proba = best_rf.predict_proba(X_test)[:, 1]

                        fs_test_accuracy = round(accuracy_score(y_test, y_pred), 4)
                        fs_test_roc_auc = round(roc_auc_score(y_test, y_pred_proba), 4)
                        fs_clf_report = classification_report(y_test, y_pred)

                        # Feature Importances
                        feature_importances = best_rf.feature_importances_
                        sorted_idx = np.argsort(feature_importances)[::-1]
                        important_features = [X_train.columns[i] for i in sorted_idx]

                        top_25_percent = int(0.25 * len(feature_importances))
                        top_25_features = important_features[:top_25_percent]
                        top_25_percent_features_str = ", ".join(top_25_features)

                        if feature_subset != 1:
                            # Filter X_train, X_test to only top 25% features
                            X_train = X_train[top_25_features]
                            X_test = X_test[top_25_features]

                        # Store all in final_results
                        final_results["FS_CV_Train"] = fs_best_mean_train_score
                        # final_results["FS_Best_Std_Train_Score"] = fs_best_std_train_score
                        final_results["FS_CV_Test"] = fs_best_mean_test_score
                        # final_results["FS_Best_Std_Test_Score"] = fs_best_std_test_score
        
                        # final_results["FS_Worst_Mean_Train_Score"] = fs_worst_mean_train_score
                        # final_results["FS_Worst_Std_Train_Score"] = fs_worst_std_train_score
                        # final_results["FS_Worst_Mean_Test_Score"] = fs_worst_mean_test_score
                        # final_results["FS_Worst_Std_Test_Score"] = fs_worst_std_test_score
        
                        # final_results["FS_Best_Params"] = str(best_params_fs)
                        # final_results["FS_Test_Accuracy"] = fs_test_accuracy
                        final_results["FS_Test"] = fs_test_roc_auc
                        # final_results["FS_Classification_Report"] = fs_clf_report
                        # final_results["FS_Top_25_Features"] = top_25_percent_features_str
                    # else:
                        # If no feature selection, store placeholders or empty strings
                        # final_results["FS_Best_Mean_Train_Score"] = ""
                        # final_results["FS_Best_Std_Train_Score"] = ""
                        # final_results["FS_Best_Mean_Test_Score"] = ""
                        # final_results["FS_Best_Std_Test_Score"] = ""
        
                        # final_results["FS_Worst_Mean_Train_Score"] = ""
                        # final_results["FS_Worst_Std_Train_Score"] = ""
                        # final_results["FS_Worst_Mean_Test_Score"] = ""
                        # final_results["FS_Worst_Std_Test_Score"] = ""
        
                        # final_results["FS_Best_Params"] = ""
                        # final_results["FS_Test_Accuracy"] = ""
                        # final_results["FS_Test_ROC_AUC"] = ""
                        # final_results["FS_Classification_Report"] = ""
                        # final_results["FS_Top_25_Features"] = ""

                    ###############################################################################
                    # Define our classifiers and grids
                    ###############################################################################
                    classifiers_and_grids = {
                        "RandomForestClassifier": (
                            RandomForestClassifier(random_state=seed),
                            {
                                "n_estimators": [10, 100, 200],
                                "max_depth": [5, 10, 15],
                                "min_samples_split": [20, 50]

                        }
                        ),
                        "AdaBoostClassifier": (
                            AdaBoostClassifier(random_state=seed),
                            {
                                "n_estimators": [10, 100, 200],
                                "algorithm": ['SAMME'],
                                "learning_rate": [0.001, 0.01, 0.1]
                            }
                        ),
                        "XGBClassifier": (
                            XGBClassifier(random_state=seed, eval_metric='auc', use_label_encoder=False),
                            {
                                "n_estimators": [10, 100, 200],
                                "max_depth": [5, 10, 15],
                                "learning_rate": [0.001, 0.01, 0.1],
                            }
                        ),
                        # "VotingClassifier": (
                        #     VotingClassifier(
                        #         estimators=[('rf', RandomForestClassifier(random_state=seed)),
                        #                     ('xgb', XGBClassifier(random_state=seed, eval_metric='auc')),
                        #                     ('ada', AdaBoostClassifier(algorithm='SAMME', random_state=seed))],
                        #         voting='soft'
                        #     ),
                        #     {
                        #     'rf__n_estimators': [100],
                        #     'xgb__n_estimators': [100],
                        #     'ada__n_estimators': [100],
                        # }
                        # )
                    }

                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

                    ###############################################################################
                    # Train each classifier, store best & worst metrics in final_results
                    ###############################################################################
                    for clf_name, (clf, param_grid) in classifiers_and_grids.items():
                        grid_search = GridSearchCV(
                            estimator=clf,
                            param_grid=param_grid,
                            scoring="roc_auc",
                            cv=cv,
                            refit=True,
                            return_train_score=True,
                            n_jobs=1,
                            verbose=1
                        )

                        grid_search.fit(X_train, y_train)

                        # Get mean train and test scores from CV results
                        train_scores = grid_search.cv_results_['mean_train_score']
                        test_scores = grid_search.cv_results_['mean_test_score']

                        # Compute overfitting gap (train score - test score)
                        gap = train_scores - test_scores
                        print(100*'-')
                        print(gap)

                        threshold = 0.1  # Adjust based on your metric (e.g., 0.1 for accuracy, 0.05 for F1)

                        # Criteria: (1) Gap < threshold, (2) Best test score among eligible models
                        eligible_mask = gap < threshold

                        if eligible_mask.any():
                            # Get indices of eligible models
                            eligible_indices = np.where(eligible_mask)[0]
                        
                            # Subset scores and gaps for eligible models
                            eligible_test_scores = test_scores[eligible_indices]
                            eligible_gaps = gap[eligible_indices]
                        
                            # Find the best test score among eligible models
                            best_test_score = eligible_test_scores.max()
                        
                            # Among models with best_test_score, find the one with SMALLEST GAP
                            best_score_mask = (eligible_test_scores == best_test_score)
                            best_gap = eligible_gaps[best_score_mask].min()
                        
                            # Get the index of the first model with best score AND smallest gap
                            best_model_idx = eligible_indices[
                                np.where((eligible_test_scores == best_test_score) & (eligible_gaps == best_gap))[0][0]
                            ]
                        else:
                            print("No model meets the criteria")
                            best_model_idx = grid_search.cv_results_['rank_test_score'].argmin()

                        # Indices of best and worst models
                        # best_model_idx = grid_search.cv_results_['rank_test_score'].argmin()
                        # worst_model_idx = grid_search.cv_results_['rank_test_score'].argmax()

                        # Best model metrics
                        best_mean_train_score = round(grid_search.cv_results_['mean_train_score'][best_model_idx], 4)
                        best_std_train_score = round(grid_search.cv_results_['std_train_score'][best_model_idx], 4)
                        best_mean_test_score = round(grid_search.cv_results_['mean_test_score'][best_model_idx], 4)
                        best_std_test_score = round(grid_search.cv_results_['std_test_score'][best_model_idx], 4)

                        # Worst model metrics
                        # worst_mean_train_score = round(grid_search.cv_results_['mean_train_score'][worst_model_idx], 4)
                        # worst_std_train_score = round(grid_search.cv_results_['std_train_score'][worst_model_idx], 4)
                        # worst_mean_test_score = round(grid_search.cv_results_['mean_test_score'][worst_model_idx], 4)
                        # worst_std_test_score = round(grid_search.cv_results_['std_test_score'][worst_model_idx], 4)

                        # Best estimatorss
                        best_model = grid_search.best_estimator_
                        best_params = best_model.get_params()

                        feature_importances = best_model.feature_importances_
                        final_results["Feature Importances"] = feature_importances

                        # Predict on test set
                        y_pred = best_model.predict(X_test)
                        y_pred_proba = best_model.predict_proba(X_test)[:, 1]

                        # --- Plot and Save TPR, FPR and Threshold curve ---
                        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

                        plt.figure(figsize=(10, 6))

                        # Plot TPR and (1 - FPR) against Thresholds
                        plt.plot(thresholds, tpr, label='True Positive Rate (Sensitivity)')
                        plt.plot(thresholds, 1-fpr, label='True Negative Rate (Specificity)')

                        # Add labels and legend
                        plt.xlabel('Threshold')
                        plt.ylabel('Rate')
                        plt.title('TPR and (1-FPR) vs Threshold')
                        plt.legend()
                        plt.grid()
                        plt.savefig(f'../Visualizations/{patient_subset}_{feature_subset}_{engaged}_{seed}_{clf_name}_sensitivity_specificity_threshold.png')  # You can change the filename and format
                        plt.close() # Close the figure to free up memory


                    
                        # --- Plot and Save ROC-AUC Curve ---
                        fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
                        roc_auc = roc_auc_score(y_test, y_pred_proba)

                        plt.figure(figsize=(8, 6))
                        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                        plt.xlabel('False Positive Rate (FPR)')
                        plt.ylabel('True Positive Rate (TPR) or Recall')
                        plt.title('Receiver Operating Characteristic (ROC) Curve')
                        plt.legend(loc='lower right')

                        # Save the ROC-AUC curve figure
                        plt.savefig(f'../Visualizations/{patient_subset}_{feature_subset}_{engaged}_{seed}_{clf_name}_roc_auc_curve.png')  # You can change the filename and format
                        plt.close() # Close the figure to free up memory

                        # --- Plot and Save Precision-Recall Curve ---
                        precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
                        auc_pr = auc(recall, precision)

                        plt.figure(figsize=(8, 6))
                        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {auc_pr:.2f})')
                        plt.xlabel('Recall (TPR)')
                        plt.ylabel('Precision')
                        plt.title('Precision-Recall Curve')
                        plt.legend(loc='upper right')

                        # Save the Precision-Recall curve figure
                        plt.savefig(f'../Visualizations/{patient_subset}_{feature_subset}_{engaged}_{seed}_{clf_name}_precision_recall_curve.png')  # You can change the filename and format
                        plt.close() # Close the figure



                        test_accuracy = round(accuracy_score(y_test, y_pred), 4)
                        test_roc_auc = round(roc_auc_score(y_test, y_pred_proba), 4)
                        y_pred_thresholded = (y_pred_proba >= 0.35).astype(int)
                        test_sensitivity = round(recall_score(y_test, y_pred_thresholded), 4)
                        test_specificity = round(recall_score(y_test, y_pred_thresholded, pos_label=0), 4)
                        clf_report = classification_report(y_test, y_pred_thresholded)

                        # Store results for this classifier
                        # We'll prefix the columns with the classifier name to keep them unique
                        final_results[f"{clf_name}_CV_Train"] = best_mean_train_score
                        # final_results[f"{clf_name}_Best_Std_Train_Score"] = best_std_train_score
                        final_results[f"{clf_name}_CV_Test"] = best_mean_test_score
                        # final_results[f"{clf_name}_Best_Std_Test_Score"] = best_std_test_score

                        # final_results[f"{clf_name}_Worst_Mean_Train_Score"] = worst_mean_train_score
                        # final_results[f"{clf_name}_Worst_Std_Train_Score"] = worst_std_train_score
                        # final_results[f"{clf_name}_Worst_Mean_Test_Score"] = worst_mean_test_score
                        # final_results[f"{clf_name}_Worst_Std_Test_Score"] = worst_std_test_score

                        final_results[f"{clf_name}_Best_Model_Params"] = str(best_params)
                        # final_results[f"{clf_name}_Test_Accuracy"] = test_accuracy
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