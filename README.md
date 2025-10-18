# Libraries Used
  - EasyOCR : For extracting text from image of medicine carton
  - ChromadDB : As the vector database
  - Vectorizing model : all-MiniLM-L6-v2
  - Streamlit : for the interface
  - LangGraph and LangChain : for designing the agentica AI workflow

# LLM Used
  - Meta's Llama3.1:8b : For chatting with the user

# AI Tools Used :
  - for generating synthetic data of the dosage and warnings of each medicine
  - for help taking image as input in the streamlit interface
  - for designing the layout of the streamlit interface

# How to Run The Code :
First install the following libraries :
- chromadb
- langchain-core
- langchain-ollama
- langgraph
- opencv-python
- easyocr
- sentence-transformers
- torch
- typing extensions
- streamlit

  Also, install Ollama to run the llm locally

Then go to the directory in which the Interface.py file is presnt and run the command 'streamlit run Interface.py'.
Click on the local host link.

  
