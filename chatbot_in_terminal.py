import chromadb
from chromadb.utils import embedding_functions
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
import cv2
import easyocr
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from playsound import playsound
import pyttsx3



class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]

@tool
def retrieve(search_query: str) -> str:
    """Retrieve relevant information about the medicine from ChromaDB based on a search query.
    
    Args:
        search_query: The name of the medicine
        
    Returns:
        Details about the top 5 medicines names that match the search_query.
    """
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path="./Vectorized_Medicine_Info")
    collection = client.get_or_create_collection(
                name="Medicine_Details",
                embedding_function=embedding_function, # type:ignore
                metadata={"hnsw:space": "cosine"}  
    )

    results = collection.query(
        query_texts=search_query,
        n_results=5,
    )

    context = '' 
    for document_index in range(len(results['documents'][0])):  # type: ignore
        context = context + '\n' + results['documents'][0][document_index] # type: ignore
    
    print("#################### INFORMATION RETRIEVED ###########################")
    print('search query : ',search_query, '\ncontext:\n',context)
    print("######################################################################")
    return context


tools = [retrieve]

llm = ChatOllama(
    model = "llama3.1:8b",
    temperature = 0.07,
    num_predict = -1,
)

llm_with_tools = llm.bind_tools(tools)

def model_call(state:AgentState) -> AgentState:
    response = llm_with_tools.invoke(state["messages"])
    return {'messages': [response]} # add_messages (reducer function) takes care of automatically appending instead of overwriting

def should_continue(state: AgentState):
    messages = state['messages']
    last_message = messages[-1]
    if not last_message.tool_calls: # type: ignore
        return "end"
    else:
        return "continue"


class App:
    def __init__(self) -> None:
        self.graph = StateGraph(state_schema=AgentState)
        self.graph.add_node("Agent", model_call)

        self.tool_node = ToolNode(tools=tools)
        self.graph.add_node("tools", self.tool_node)

        self.graph.set_entry_point("Agent")

        self.graph.add_conditional_edges(
            "Agent", 
            should_continue,
            {
                "continue": "tools",
                "end": END,
            }
        )

        self.graph.add_edge("tools", "Agent")
        self.app = self.graph.compile()

        self.system_prompt = SystemMessage(content="""You are a helpful assistant that has access to details about any medicine.
        If the user's query is related to medication then use the retrieve tool to get relevant information from the database before answering, if not do not use the tool and redirect the conversation towards medication by telling the user how you can help him there.
        Do not mention anything about using the tool to the user""")

        self.conversation_history = [self.system_prompt]

        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.labels = [
            "Acretin cream",
            "Adol caplets",
            "Aggrex tablets",
            "Airoplast Tape",
            "All-Vent syrup",
            "Alphintern tablets",
            "Ambezim-G tablets",
            "Amrizole tablets",
            "Antopral tablets",
            "Anuva tablets",
            "Apidone syrup",
            "Asmakast tablets",
            "Atoreza tablets",
            "Augmentin tablets",
            "Avosoya capsules",
            "B.B.C. spray solution",
            "Benogen sachets",
            "Betaderm cream",
            "Betadine solution",
            "Betadine ointment",
            "Betadine mouth wash",
            "Bivatracin spray",
            "Bronchicum Elixir",
            "Bronchopro syrup",
            "Brufen tablets",
            "Brufen syrup",
            "C Zinc capsules",
            "Candalkan Plus tablets",
            "Carbamide cream",
            "Catafast sachets",
            "Cataflam tablets",
            "Ceftriaxone Vial",
            "Celebrex capsules",
            "Cemicresto tablets",
            "Cetal oral suspension",
            "Cetal Sinus capsules",
            "Cholerose tablets",
            "Choletimb tablets",
            "Ciprodiazole tablets",
            "Ciprofar tablets",
            "Ciprofloxacin Tablets USP 39 tablets",
            "Clarinase repetabs",
            "Claritine tablets",
            "Clindasol gel",
            "Comfort Massage Gel gel",
            "Congestal tablets",
            "Conta-flu tablets",
            "Corasore oral drops",
            "C-Retard capsules",
            "C-vit oral drops",
            "Daflon tablets",
            "Daktarin gel oral",
            "Dalacin C capsules",
            "Davalindi tablets",
            "Dentinox colic drops",
            "Diclac gel",
            "Diclac 75 ID tablets",
            "Diflucan capsules",
            "Diflucan capsule",
            "Ectomethrin emulsion",
            "Enrich oral drops",
            "Enrich syrup",
            "Ezamol-C tablets",
            "Farcolin syrup",
            "Fenistil oral drops",
            "Feroglobin capsules",
            "Ferrofol spansules",
            "Flagyl tablets",
            "Flector EP gel",
            "Flix nasal spray",
            "Floxabact tablets",
            "Flumox capsules",
            "Flumox tablets",
            "Foradil capsules",
            "Free Nose nasal spray",
            "Frost spray",
            "Fucicort cream",
            "Fucidin cream",
            "Fucidin cream",
            "Fucidin cream",
            "Gabalepsy capsules",
            "Garamycin cream",
            "Gengigel hydrogel",
            "Glucophage tablets",
            "GTN cream",
            "Haemostop tablets",
            "Hemoclar cream",
            "Hibiotic tablets",
            "Histazine-1 tablets",
            "ImmuGuard sachets",
            "Ivypront syrup",
            "Janumet tablets",
            "Jogel oral gel",
            "Jusprin tablets",
            "Lactulose syrup",
            "Lamifen cream",
            "Lamisil cream",
            "La-vie gel",
            "Lovir tablets",
            "Mebo ointment",
            "Megamox tablets",
            "Micropore Tape",
            "Midodrine tablets",
            "Miflonide capsules",
            "Milga tablets",
            "Minalax tablets",
            "Motilium tablets",
            "Mucophylline syrup",
            "Mucosta tablets",
            "Multi-Relax tablets",
            "Mupirax ointment",
            "Neurovit tablets",
            "Night & Day tablets",
            "Oracure oral gel",
            "Orex solution",
            "Orovex Delicate mouth wash",
            "Osteocare liquid",
            "Pandermal cream",
            "Panthenol cream",
            "Paramol tablets",
            "Pedical syrup",
            "Pentamix syrup",
            "Phenadone syrup",
            "Picolax oral drops",
            "Predsol syrup",
            "Pridocaine cream",
            "Primrose Plus capsules",
            "Prinorelax capsules",
            "Remowax ear drops",
            "Rennie chewable tablets",
            "Reparil-Gel N gel",
            "Rheumatizen cream",
            "Rhinopro capsules",
            "Salivex-L paint",
            "Sediproct cream",
            "SelokenZOC tablets",
            "Solvadol spray",
            "Steronate tablets",
            "Streptoquin tablets",
            "Tareg tablets",
            "Tears Guard eye drops",
            "Tentavair oral inhalation",
            "Vidrop oral drops",
            "Visceralgine syrup",
            "Vitacid C tablets",
            "Vitamax capsules",
            "Zantac tablets",
            "Zenta cream",
            "Zyrtec oral drops",
            "Zyrtec tablets",
            "Sensodyne",
            "Levolin inhaler",
        ]

        self.label_embeddings = self.model.encode(self.labels, convert_to_tensor=True)

    def predict_using_text(self, extracted_text):
        self.query_embedding = self.model.encode(extracted_text, convert_to_tensor=True)
        cosine_scores = util.cos_sim(self.query_embedding, self.label_embeddings)

        best_match_index = torch.argmax(cosine_scores[0])
        best_score = cosine_scores[0][best_match_index]

        return self.labels[best_match_index]

    def quadrilateral_area(self, tl, tr, br, bl):
        points = [tl, tr, br, bl]

        # Shoelace formula
        area = 0
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            area += (x1 * y2) - (x2 * y1)

        return abs(area) / 2

    def extract_text_from_image(self, image_path):
        """
        Preprocesses an image and extracts text using EasyOCR.
        
        Args:
            image_path (str): The path to the image.

        Returns:
            list: The list of extracted text arranged in the decreasing order of size.
        """
        img = cv2.imread(image_path, 0)
        blur = cv2.GaussianBlur(img,(5,5),0) # type: ignore
        reader = easyocr.Reader(['en'])
        result = reader.readtext(blur)

        text_in_image = []

        for (bbox, text, prob) in result:
            (tl, tr, br, bl) = bbox
            text_in_image.append((text, self.quadrilateral_area(tl, tr, br, bl)/len(text)))
        
        text_in_image = sorted(text_in_image, key=lambda x: x[1], reverse=True)

        text_in_image_sorted = []

        for obj in text_in_image:
            text_in_image_sorted.append(obj[0])

        
        return text_in_image_sorted

    def predict(self, image_path):
        extracted_text = self.extract_text_from_image(image_path)
        return self.predict_using_text(' '.join(extracted_text[0 : min(7, len(extracted_text))]))

    def get_first_message(self):
        self.conversation_history.append(llm.invoke([SystemMessage("You are an expert at recognizing medicine and have complete knowledge about the use and dosage of any medicine. Respond by telling the user that he can uplaod a pic of the medicine and you would recognize it for him."), HumanMessage('Hi')])) #type:ignore
        return self.conversation_history[-1].content

    def invoke_app(self, user_input):
        self.conversation_history.append(HumanMessage(content=user_input)) # type: ignore
        ai_msg = self.app.invoke({"messages" : self.conversation_history})
        self.conversation_history = ai_msg['messages']
        return self.conversation_history[-1].content
    
    def invoke_app_for_img(self, user_input, medicine_name):
        self.conversation_history.append(SystemMessage(content=f"An image was provided which you recognized as {medicine_name}"))
        self.conversation_history.append(HumanMessage(content=user_input)) # type: ignore
        ai_msg = self.app.invoke({"messages" : self.conversation_history})
        self.conversation_history = ai_msg['messages']
        return self.conversation_history[-1].content
    
app = App()

def voice_over(text):
    # Initialize the TTS engine
    engine = pyttsx3.init()

    # Optional: Customize voice and speed
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)  # type: ignore
    engine.setProperty('rate', 175)  

    # Speak the given text
    engine.say(text)
    engine.runAndWait()

print('-------------------------------------------')
print("Enter 1 for uploading an image and 2 for chatting about a medicine :\n")
print('AI : ',app.get_first_message())
voice_over("Enter 1 for uploading an image and 2 for chatting about a medicine.")
user_input = input("Enter 1 or 2 : ")


while user_input != 'exit':
    if user_input == '1':
        cam = cv2.VideoCapture(0)
        ret, frame = cam.read()
        if ret:
            cv2.imwrite("captured_image.jpg", frame)
            print("Image captured and saved as captured_image.jpg")
        else:
            print("Error: Could not read frame from webcam.")
        cam.release()

        result = app.invoke_app_for_img("What is this image?", app.predict('captured_image.jpg'))
        print("AI : ", result)
    else:
        prompt = input("User : ")
        result = app.invoke_app(prompt)
        print("AI : ", result)
    print('-------------------------------------------')
    voice_over(result)
    user_input = input("Enter 1 or 2 : ")
    

#app = App()
#print(app.predict("Dataset/All-Vent 125 ml syrup/huawei p30 415.jpg"))

# app = App()
# print('-------------------------------------------')
# print('AI : ',app.get_first_message())
# user_input = input("User : ")
# while user_input != 'exit':
#     print(app.invoke_app(user_input))
#     print('-------------------------------------------')
#     user_input = input("User : ")