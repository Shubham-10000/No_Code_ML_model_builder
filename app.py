import streamlit as st

from file_manage import upload_csv, list_files

from preprocessing import preprocess_data

from model_training import model_training






st.set_page_config(page_title='Automation',
                   page_icon='ai_logo.png', 
                   layout='wide'
                   )

st.sidebar.image('ai_logo.png')
page = st.sidebar.selectbox('Select an Option', ['File Management', 'Preprocessing Data', 'Model Training'])

if page == 'File Management':
    st.title('File Management')
    upload_csv()
    list_files()

elif page == 'Preprocessing Data':
    st.title('Preprocessing Data')
    preprocess_data()


elif page == 'Model Training':
    st.title('Model Training')
    model_training()
