import numpy as np
import pandas as pd
import nltk
import os

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

# NLTK setup
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Global parameters
patient_subset = 1   # 1 - All patients, 2 - Patients with Notes, 3 - Patients without Notes
feature_subset = 1   # 1 - Demo, 2 - Demo + Tabular, 3 - Demo + Tabular + Notes
feat_sel = 1         # 1 = Feature selection on (RandomForest), 0 = off
test_split_ratio = 0.2
seed = 23
file_path = '../Data/new_final_df.xlsx'
output_csv = 'outcome_contact_training_log.csv'
demographic_features = ['visit_number', 'referral_date', 'day_readmit',	'age', 
                        'sex_Male', 'Hispanic', 'AfricanAmerican', 'White',
                       'Additional_race', 'language_English', 'language_Spanish',
                      'language_Other', 'hypertension']

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
min_max_scaler = MinMaxScaler()

for patient_subset in range(1,4):
    for feature_subset in range(1,4):
        # for feat_sel in range(0,2):
        # for seed in range(0,50,14):
        
            if patient_subset == 3 and feature_subset == 3:
                continue



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
                suhi_df.drop(columns=['record_id', 'new_patient'], inplace=True, errors='ignore')
                # Remove rows with NaN in day_readmit
                suhi_df.dropna(subset=['day_readmit'], inplace=True)
                # Convert day_readmit == 2 to 0
                suhi_df.loc[suhi_df['day_readmit'] == 2, 'day_readmit'] = 0
                # Convert day_readmit to int
                suhi_df['day_readmit'] = suhi_df['day_readmit'].astype(int)
                # Scale age
                suhi_df['age'] = min_max_scaler.fit_transform(suhi_df[['age']])
    
                # Combine note columns
                note_cols = suhi_df.columns[suhi_df.columns.str.contains('notes_contact')]
                combined_notes = suhi_df[note_cols].apply(
                    lambda x: '. '.join([str(note).lower() for note in x.dropna()]),
                    axis=1
                )
                suhi_df['combined_notes'] = combined_notes
                return suhi_df

            ###############################################################################
            # Main Script Execution
            ###############################################################################

            # Load data
            suhi_df = pd.read_excel(file_path)

            # Preprocess
            suhi_df = preprocess(suhi_df)

            # If we only want patients with notes
            if patient_subset == 2:
                suhi_df['combined_notes'] = suhi_df['combined_notes'].replace('', np.nan)
                suhi_df.dropna(subset=['combined_notes'], inplace=True)

            # If we only want patients without notes
            if patient_subset == 3:
                suhi_df['combined_notes'] = suhi_df['combined_notes'].replace('', np.nan)
                suhi_df = suhi_df[suhi_df['combined_notes'].isna()]
                # suhi_df['combined_notes'].fillna('', inplace=True)

            # If feature_subset == 1 -> only Demographic (first 13 columns)
            if feature_subset == 1:
                suhi_df = suhi_df[demographic_features]

            # If we include text features
            if feature_subset == 3:
                tfidf_vectorizer = TfidfVectorizer(
                    min_df=20, 
                    max_df=0.7, 
                    stop_words='english', 
                    token_pattern=r'[a-zA-Z]{2,}')
                text_embeddings = tfidf_vectorizer.fit_transform(suhi_df['combined_notes'])

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
            if feature_subset == 3:
                combined_notes_vectorized_df = pd.DataFrame(text_embeddings.toarray())
                combined_notes_vectorized_df.columns = tfidf_vectorizer.get_feature_names_out()
    
                suhi_df.reset_index(drop=True, inplace=True)
                suhi_w_vectors_df = pd.concat([suhi_df, combined_notes_vectorized_df], axis=1)
            else:
                suhi_w_vectors_df = suhi_df

            # Fill NaN with 0
            suhi_w_vectors_df.fillna(0, inplace=True)
            print(100*'-')
            print(patient_subset, feature_subset)
            print(suhi_w_vectors_df.shape)
            print(suhi_w_vectors_df.head)
            print(suhi_w_vectors_df.columns)
            # Split data
            X = suhi_w_vectors_df.drop('day_readmit', axis=1)
            y = suhi_w_vectors_df['day_readmit']

            # X_train, X_test, y_train, y_test = train_test_split(
            #     X, y, test_size=test_split_ratio, random_state=seed
            # )

            # ###############################################################################
            # # We'll collect all our results in one dictionary, final_results,
            # # so we only write one row per script run.
            # ###############################################################################
            # final_results = {}
            # final_results["Patient Subset"] = patient_subset
            # final_results["Feature Subset"] = feature_subset
            # final_results["Feature Selection"] = bool(feat_sel)
            # final_results["Random Seed"] = seed
            # final_results["File Name"] = file_path

            # top_25_features = []
            # top_25_percent_features_str = ""

            # ###############################################################################
            # # Feature Selection (RandomForest) step, if enabled
            # ###############################################################################
            # if feat_sel == 1:
            #     fs_rf = RandomForestClassifier(random_state=seed)
            #     param_grid_rf = {
            #         'n_estimators': [50, 100, 200],
            #         'max_depth': [10, 20],
            #         'min_samples_split': [5, 20],
            #         'min_samples_leaf': [1, 5]
            #     }
            #     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
            #     grid_search = GridSearchCV(
            #         estimator=fs_rf,
            #         param_grid=param_grid_rf,
            #         cv=cv,
            #         scoring='roc_auc',
            #         refit=True,
            #         return_train_score=True,
            #         n_jobs=-1,
            #         verbose=1
            #     )
    
            #     grid_search.fit(X_train, y_train)

            #     # Best/Worst model indices in the results
            #     best_model_idx = grid_search.cv_results_['rank_test_score'].argmin()
            #     worst_model_idx = grid_search.cv_results_['rank_test_score'].argmax()

            #     # Retrieve relevant metrics
            #     fs_best_mean_train_score = round(grid_search.cv_results_['mean_train_score'][best_model_idx], 4)
            #     fs_best_std_train_score = round(grid_search.cv_results_['std_train_score'][best_model_idx], 4)
            #     fs_best_mean_test_score = round(grid_search.cv_results_['mean_test_score'][best_model_idx], 4)
            #     fs_best_std_test_score = round(grid_search.cv_results_['std_test_score'][best_model_idx], 4)

            #     fs_worst_mean_train_score = round(grid_search.cv_results_['mean_train_score'][worst_model_idx], 4)
            #     fs_worst_std_train_score = round(grid_search.cv_results_['std_train_score'][worst_model_idx], 4)
            #     fs_worst_mean_test_score = round(grid_search.cv_results_['mean_test_score'][worst_model_idx], 4)
            #     fs_worst_std_test_score = round(grid_search.cv_results_['std_test_score'][worst_model_idx], 4)
    
            #     # Best estimator
            #     best_rf = grid_search.best_estimator_
            #     best_params_fs = best_rf.get_params()

            #     # Evaluate on test set
            #     y_pred = best_rf.predict(X_test)
            #     y_pred_proba = best_rf.predict_proba(X_test)[:, 1]

            #     fs_test_accuracy = round(accuracy_score(y_test, y_pred), 4)
            #     fs_test_roc_auc = round(roc_auc_score(y_test, y_pred_proba), 4)
            #     fs_clf_report = classification_report(y_test, y_pred)

            #     # Feature Importances
            #     feature_importances = best_rf.feature_importances_
            #     sorted_idx = np.argsort(feature_importances)[::-1]
            #     important_features = [X_train.columns[i] for i in sorted_idx]

            #     top_25_percent = int(0.25 * len(feature_importances))
            #     top_25_features = important_features[:top_25_percent]
            #     top_25_percent_features_str = ", ".join(top_25_features)

            #     # Filter X_train, X_test to only top 25% features
            #     X_train = X_train[top_25_features]
            #     X_test = X_test[top_25_features]

            #     # Store all in final_results
            #     final_results["FS_Best_Mean_Train_Score"] = fs_best_mean_train_score
            #     final_results["FS_Best_Std_Train_Score"] = fs_best_std_train_score
            #     final_results["FS_Best_Mean_Test_Score"] = fs_best_mean_test_score
            #     final_results["FS_Best_Std_Test_Score"] = fs_best_std_test_score
    
            #     final_results["FS_Worst_Mean_Train_Score"] = fs_worst_mean_train_score
            #     final_results["FS_Worst_Std_Train_Score"] = fs_worst_std_train_score
            #     final_results["FS_Worst_Mean_Test_Score"] = fs_worst_mean_test_score
            #     final_results["FS_Worst_Std_Test_Score"] = fs_worst_std_test_score
    
            #     final_results["FS_Best_Params"] = str(best_params_fs)
            #     final_results["FS_Test_Accuracy"] = fs_test_accuracy
            #     final_results["FS_Test_ROC_AUC"] = fs_test_roc_auc
            #     final_results["FS_Classification_Report"] = fs_clf_report
            #     final_results["FS_Top_25_Features"] = top_25_percent_features_str
            # else:
            #     # If no feature selection, store placeholders or empty strings
            #     final_results["FS_Best_Mean_Train_Score"] = ""
            #     final_results["FS_Best_Std_Train_Score"] = ""
            #     final_results["FS_Best_Mean_Test_Score"] = ""
            #     final_results["FS_Best_Std_Test_Score"] = ""
    
            #     final_results["FS_Worst_Mean_Train_Score"] = ""
            #     final_results["FS_Worst_Std_Train_Score"] = ""
            #     final_results["FS_Worst_Mean_Test_Score"] = ""
            #     final_results["FS_Worst_Std_Test_Score"] = ""
    
            #     final_results["FS_Best_Params"] = ""
            #     final_results["FS_Test_Accuracy"] = ""
            #     final_results["FS_Test_ROC_AUC"] = ""
            #     final_results["FS_Classification_Report"] = ""
            #     final_results["FS_Top_25_Features"] = ""

            # ###############################################################################
            # # Define our classifiers and grids
            # ###############################################################################
            # classifiers_and_grids = {
            #     "RandomForestClassifier": (
            #         RandomForestClassifier(random_state=seed),
            #         {
            #             "n_estimators": [100, 200],
            #             "max_depth": [10, 20],
            #             "min_samples_split": [2, 5]
            #         }
            #     ),
            #     "AdaBoostClassifier": (
            #         AdaBoostClassifier(random_state=seed),
            #         {
            #             "n_estimators": [50, 100],
            #             "learning_rate": [0.1, 0.5, 1.0]
            #         }
            #     ),
            #     "XGBClassifier": (
            #         XGBClassifier(random_state=seed, eval_metric='auc'),
            #         {
            #             "n_estimators": [50, 100],
            #             "max_depth": [3, 5],
            #             "learning_rate": [0.01, 0.1]
            #         }
            #     )
            # }

            # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

            # ###############################################################################
            # # Train each classifier, store best & worst metrics in final_results
            # ###############################################################################
            # for clf_name, (clf, param_grid) in classifiers_and_grids.items():
            #     grid_search = GridSearchCV(
            #         estimator=clf,
            #         param_grid=param_grid,
            #         scoring="roc_auc",
            #         cv=cv,
            #         refit=True,
            #         return_train_score=True,
            #         n_jobs=-1,
            #         verbose=1
            #     )

            #     grid_search.fit(X_train, y_train)

            #     # Indices of best and worst models
            #     best_model_idx = grid_search.cv_results_['rank_test_score'].argmin()
            #     worst_model_idx = grid_search.cv_results_['rank_test_score'].argmax()

            #     # Best model metrics
            #     best_mean_train_score = round(grid_search.cv_results_['mean_train_score'][best_model_idx], 4)
            #     best_std_train_score = round(grid_search.cv_results_['std_train_score'][best_model_idx], 4)
            #     best_mean_test_score = round(grid_search.cv_results_['mean_test_score'][best_model_idx], 4)
            #     best_std_test_score = round(grid_search.cv_results_['std_test_score'][best_model_idx], 4)

            #     # Worst model metrics
            #     worst_mean_train_score = round(grid_search.cv_results_['mean_train_score'][worst_model_idx], 4)
            #     worst_std_train_score = round(grid_search.cv_results_['std_train_score'][worst_model_idx], 4)
            #     worst_mean_test_score = round(grid_search.cv_results_['mean_test_score'][worst_model_idx], 4)
            #     worst_std_test_score = round(grid_search.cv_results_['std_test_score'][worst_model_idx], 4)

            #     # Best estimator
            #     best_model = grid_search.best_estimator_
            #     best_params = best_model.get_params()

            #     # Predict on test set
            #     y_pred = best_model.predict(X_test)
            #     y_pred_proba = best_model.predict_proba(X_test)[:, 1]

            #     test_accuracy = round(accuracy_score(y_test, y_pred), 4)
            #     test_roc_auc = round(roc_auc_score(y_test, y_pred_proba), 4)
            #     clf_report = classification_report(y_test, y_pred)

            #     # Store results for this classifier
            #     # We'll prefix the columns with the classifier name to keep them unique
            #     final_results[f"{clf_name}_Best_Mean_Train_Score"] = best_mean_train_score
            #     final_results[f"{clf_name}_Best_Std_Train_Score"] = best_std_train_score
            #     final_results[f"{clf_name}_Best_Mean_Test_Score"] = best_mean_test_score
            #     final_results[f"{clf_name}_Best_Std_Test_Score"] = best_std_test_score

            #     final_results[f"{clf_name}_Worst_Mean_Train_Score"] = worst_mean_train_score
            #     final_results[f"{clf_name}_Worst_Std_Train_Score"] = worst_std_train_score
            #     final_results[f"{clf_name}_Worst_Mean_Test_Score"] = worst_mean_test_score
            #     final_results[f"{clf_name}_Worst_Std_Test_Score"] = worst_std_test_score

            #     final_results[f"{clf_name}_Best_Model_Params"] = str(best_params)
            #     final_results[f"{clf_name}_Test_Accuracy"] = test_accuracy
            #     final_results[f"{clf_name}_Test_ROC_AUC"] = test_roc_auc
            #     final_results[f"{clf_name}_Classification_Report"] = clf_report

            # ###############################################################################
            # # Append exactly ONE row for this entire run
            # ###############################################################################
            # append_results_to_csv(final_results, output_csv)
            # print(f"\nDone! Logged results to {output_csv} as a single row.\n")