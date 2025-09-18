import streamlit as st
import pandas as pd
from file_manage import list_files
import os
from sklearn.preprocessing import LabelEncoder
import warnings
import json

warnings.filterwarnings("ignore")

def preprocess_data():
    st.title('Data Preprocessing')

    csv_files = list_files()

    if not csv_files:
        st.warning("No CSV files available for preprocessing.")
    else:
        selected_file = st.selectbox('Select a CSV file for Preprocessing', csv_files)
        if selected_file:
            df = pd.read_csv(os.path.join('uploads', selected_file))
            st.write('**Data Preview:**')
            st.dataframe(df.head())

            # Step 1: Show unique values before encoding
            if st.button('Show Unique Values'):
                st.subheader("Unique Values in Each Column")
                for col in df.columns:
                    st.write(f"**{col}** ({df[col].dtype}) → {df[col].nunique()} unique values")
                    st.write(df[col].unique())

            # Step 2: Analyze data
            if st.button('Analyze Data'):
                null_counts = df.isnull().sum()
                null_info = pd.DataFrame({'Column': null_counts.index, 'Null Values': null_counts.values})
                st.subheader("Null Values Information")
                st.dataframe(null_info)

                categorical_data = df.select_dtypes(include=['object']).columns.tolist()
                st.subheader("Categorical Columns")
                st.write(categorical_data if categorical_data else "No categorical columns found.")

            # Step 3: Preprocess
            if st.button('Preprocess Data'):
                labelencoder_mappings = {}  # save the encoded mappings

                for col in df.columns:
                    # Fill missing values
                    if df[col].isnull().sum() > 0:
                        df[col].fillna(df[col].mode()[0], inplace=True)

                    # Encode categorical columns
                    if df[col].dtype == 'object':
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col])
                        mapping_dict = {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
                        labelencoder_mappings[col] = mapping_dict

                        # Show encoding mapping in Streamlit
                        st.write(f"**Encoding for column `{col}`:**")
                        st.json(mapping_dict)

                # Save preprocessed CSV
                preprocessed_folder = 'preprocessed_data'
                os.makedirs(preprocessed_folder, exist_ok=True)
                preprocessed_filename = selected_file
                df.to_csv(os.path.join(preprocessed_folder, preprocessed_filename), index=False)
                st.success(f"Preprocessed file saved as {preprocessed_filename}")

                # Save label encoder mappings
                mapping_folder = "mapping"
                os.makedirs(mapping_folder, exist_ok=True)
                mapping_file = os.path.join(mapping_folder, f"{selected_file.split('.')[0]}.json")
                with open(mapping_file, "w") as f:
                    json.dump(labelencoder_mappings, f, indent=4)
                st.success(f"Label encoding mappings saved in {mapping_file}")

                # Show final preprocessed data
                st.subheader("Preprocessed Data")
                st.dataframe(df)
                st.success("Data preprocessing completed successfully!")