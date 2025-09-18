import streamlit as st 
import os

SAVE_DIR = 'uploads'
os.makedirs(SAVE_DIR, exist_ok=True)

def upload_csv():
    uploaded_file = st.file_uploader("Choose a CSV file",type="csv")
    
    if uploaded_file is not None:
        # Save the uploaded file to a specific directory
        save_path = os.path.join('uploads', uploaded_file.name)
        with open(save_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File {uploaded_file.name} uploaded successfully!")
        
def list_files():
    st.subheader("List of Uploaded Files")
    files = os.listdir('uploads')
    if files:
        for file in files:
            col1, col2 = st.columns([10, 1])
            col1.write(file)
            if col2.button("Delete", key=file):
                os.remove(os.path.join('uploads',file))
                st.success(f'File {file} deleted successfully!')
                st.rerun()
    else:
        st.info("No files uploaded yet.")
    
    return files

   
