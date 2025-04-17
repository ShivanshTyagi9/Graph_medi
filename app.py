import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from neo4j import GraphDatabase, basic_auth
import requests
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j.exceptions import ServiceUnavailable
from langchain_google_genai import ChatGoogleGenerativeAI
from pyvis.network import Network
from langchain_neo4j import Neo4jGraph
from PIL import Image
import streamlit.components.v1 as components

# Load environment variables
load_dotenv()

# Gemini & Neo4j configuration


NEO4J_URI = st.secrets["NEO4J_URI"]
NEO4J_USERNAME = st.secrets["NEO4J_USERNAME"]
NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]
NEO4J_URI2=st.secrets["NEO4J_URI3"]
NEO4J_USERNAME2=st.secrets["NEO4J_USERNAME3"]
NEO4J_PASSWORD2=st.secrets["NEO4J_PASSWORD3"]
NEO4J_URI3=st.secrets["NEO4J_URI3"]
NEO4J_USERNAME3 =st.secrets["NEO4J_USERNAME3"]
NEO4J_PASSWORD3 =st.secrets["NEO4J_PASSWORD3"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
EMBEDDING_MODEL = "models/text-embedding-004"
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

conversation_history = {}

st.set_page_config(page_title="Medical Assistant", layout="wide")

st.sidebar.title("🩺 Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Medical Assistant", "Knowledge Graph","Conversation Graph", "Prescription Validation"])

# ========== Utility Functions ==========
llm = ChatGoogleGenerativeAI(
    google_api_key=GOOGLE_API_KEY,
    model="gemini-2.0-flash",
    temperature=0.5,
)

graph_transformer = LLMGraphTransformer(llm=llm)

# Clear existing graph in Neo4j
def clear_graph():
    query = "MATCH (n) DETACH DELETE n"
    try:
        with GraphDatabase.driver(NEO4J_URI3, auth=(NEO4J_USERNAME3, NEO4J_PASSWORD3)) as driver:
            with driver.session(database="neo4j") as session:
                session.run(query)
                print("Graph data cleared.")
    except ServiceUnavailable as e:
        st.error(f"Error clearing graph: {e}")
        return False
    return True

def get_embedding(text: str):
    try:
        result = genai.embed_content(model=EMBEDDING_MODEL, content=text, task_type="RETRIEVAL_DOCUMENT")
        return result['embedding']
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None

def find_disease_and_treatments(symptom_query_embedding, top_k=3):
    cypher_query = """
    CALL db.index.vector.queryNodes('symptom_embeddings', $topK, $embedding)
    YIELD node AS symptom, score
    WHERE "Symptom" IN labels(symptom)
    MATCH (symptom)-[:SYMPTOM_OF]->(d:Disease)
    OPTIONAL MATCH (d)-[:TREATED_BY]->(t:Treatment)
    RETURN DISTINCT d.name AS disease, collect(DISTINCT t.name) AS treatments, score
    ORDER BY score DESC
    LIMIT $topK
    """
    try:
        with GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
            with driver.session(database="neo4j") as session:
                result = session.run(cypher_query, embedding=symptom_query_embedding, topK=top_k)
                return [record.data() for record in result]
    except Exception as e:
        st.error(f"Neo4j query error: {e}")
        return []

def get_weather(city: str):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()
        if data.get("main"):
            return {
                "temp": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "description": data["weather"][0]["description"],
                "wind_speed": data["wind"]["speed"]
            }
        else:
            return {"error": data.get("message", "Could not fetch weather.")}
    except Exception as e:
        return {"error": str(e)}

def validate_prescription(disease, drug):
        driver = GraphDatabase.driver(NEO4J_URI2, auth=basic_auth(NEO4J_USERNAME2, NEO4J_PASSWORD2))
        with driver.session(database="neo4j2") as session:
            result = session.run("""
                MATCH (d:Disease {name: $disease})-[:TREATED_BY]->(dr:Drug {name: $drug})
                RETURN count(*) > 0 AS is_valid
            """, disease=disease.strip(), drug=drug.strip())
            record = result.single()
            is_valid = record["is_valid"] if record else False
        driver.close()
        return is_valid

def fetch_graph_data(limit=100):
    query = """
    MATCH (n)-[r]->(m)
    RETURN DISTINCT n,r,m
    LIMIT $limit
    """
    try:
        with GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
            with driver.session(database="neo4j") as session:
                result = session.run(query, limit=limit)
                records = result.data()
                return records
    except Exception as e:
        st.error(f"Error fetching graph data: {e}")
        return []


def render_knowledge_graph(records):

    g = Network(height="1000px", width="100%", directed=True, bgcolor="#FFFFFF", font_color="black")
    g.force_atlas_2based()

    added_nodes = set()
    edge_count = 0

    for i, record in enumerate(records):
        source = record.get("n")
        target = record.get("m")
        rel_tuple = record.get("r")

        if not source or not target or not rel_tuple or len(rel_tuple) < 2:
            st.warning(f"Skipping invalid record at index {i}: {record}")
            continue

        source_id = source.get("name")
        target_id = target.get("name")
        relation_type = rel_tuple[1]

        if not source_id or not target_id:
            st.warning(f"Missing name in source/target at index {i}")
            continue

        if source_id not in added_nodes:
            g.add_node(source_id, label=source_id, title=str(source), color="#1f77b4")
            added_nodes.add(source_id)

        if target_id not in added_nodes:
            g.add_node(target_id, label=target_id, title=str(target), color="#ff7f0e")
            added_nodes.add(target_id)

        g.add_edge(source_id, target_id, label=relation_type)
        edge_count += 1

    st.info(f"✅ Added {len(added_nodes)} nodes and {edge_count} edges to the graph.")

    if len(added_nodes) == 0:
        st.error("🚫 No nodes were added. Please check your data.")

    g.set_options("""
    var options = {
      "nodes": {
        "shape": "dot",
        "size": 20,
        "font": { "size": 14 }
      },
      "edges": {
        "smooth": true,
        "arrows": { "to": { "enabled": true } }
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      }
    }
    """)

    path = "graph.html"
    g.save_graph(path)
    return path

def render_knowledge_graph2(records):
    g = Network(height="1000px", width="100%", directed=True, bgcolor="#FFFFFF", font_color="black")
    g.force_atlas_2based()

    added_nodes = set()
    edge_count = 0

    for i, record in enumerate(records):
        source = record.get("n")
        target = record.get("m")
        rel_tuple = record.get("r")

        if not source or not target or not rel_tuple or len(rel_tuple) < 2:
            st.warning(f"Skipping invalid record at index {i}: {record}")
            continue

        source_id = source.get("name") or source.get("id") or str(source)
        target_id = target.get("name") or target.get("id") or str(target)
        relation_type = rel_tuple[1]

        if source_id not in added_nodes:
            g.add_node(str(source_id), label=str(source_id), title=str(source), color="#1f77b4")
            added_nodes.add(source_id)

        if target_id not in added_nodes:
            g.add_node(str(target_id), label=str(target_id), title=str(target), color="#ff7f0e")
            added_nodes.add(target_id)

        g.add_edge(str(source_id), str(target_id), label=relation_type)
        edge_count += 1

    st.info(f"✅ Added {len(added_nodes)} nodes and {edge_count} edges to the graph.")

    g.set_options("""
    var options = {
      "nodes": {
        "shape": "dot",
        "size": 20,
        "font": { "size": 14 }
      },
      "edges": {
        "smooth": true,
        "arrows": { "to": { "enabled": true } }
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      }
    }
    """)

    path = "graph2.html"
    g.save_graph(path)
    return path

def fetch_graph_data2(limit=1):
    query = """
    MATCH (n)-[r]->(m)
    RETURN DISTINCT n, r, m
    """
    try:
        with GraphDatabase.driver(NEO4J_URI3, auth=(NEO4J_USERNAME3, NEO4J_PASSWORD3)) as driver:
            with driver.session(database="neo4j") as session:
                result = session.run(query, limit=limit)
                records = result.data()
                return records
    except Exception as e:
        st.error(f"Error fetching graph data: {e}")
        return []

# Add the Graph Documents into Neo4j
def add_graph_to_neo4j(graph_documents):
    clear_graph()
    graph = Neo4jGraph(
    url="neo4j+s://7d61a5bf.databases.neo4j.io",
    username="neo4j",
    password="VFdyMMmUgcPev9qZwr7t_UfnONXYuEyHHXGm12Lm39c"
    )
    graph.add_graph_documents(
    graph_documents,
    baseEntityLabel=True,
    include_source=True
    )



# ========== Streamlit UI ==========
if page == "Overview":
    st.title("🩺 Overview of the Medical Assistant")

    # Add a subtle intro line
    st.markdown("Welcome to your intelligent healthcare companion. Here's a quick look at what this assistant can do!")

    # Add a nice divider
    st.markdown("---")

    # Two-column layout for features + image (optional)
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🔍 Key Features")
        st.markdown("""
        - ✅ **Medical Assistant**: Get personalized medical advice based on your symptoms and local weather conditions.
        - 🧠 **Knowledge Graph**: Explore disease-symptom-treatment relationships visually in an interactive graph.
        - 🗣️ **Conversation Graph**: Understand patient history via knowledge graphs built from doctor-patient interactions.
        - 💊 **Prescription Validation**: Ensure drugs prescribed align with the condition for better patient safety.
        """)

    # Add a nice divider
    st.markdown("---")

    # Dataset samples with subheaders
    st.markdown("### 📚 Datasets Used")


    
    df1 = pd.read_csv("dataset/Diseases_Symptoms.csv")
    st.subheader("🧬 Knowledge Graph Dataset")
    st.dataframe(df1.head(), use_container_width=True)

    
    df2 = pd.read_csv("dataset/disease-drug.csv")
    st.subheader("💊 Prescription Graph Dataset")
    st.dataframe(df2.head(), use_container_width=True)


if page == "Medical Assistant":

    st.title("🩺 Medical Assistant")
    user_id = st.text_input("User ID", value="demo_user")
    city = st.text_input("Enter your city for weather-aware suggestions")

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = {}

    user_message = st.text_area("Describe your symptoms")

    if st.button("Get Advice"):
        if not user_message:
            st.warning("Please enter your symptoms.")
        elif not city:
            st.warning("Please enter your city.")
        else:
            # Fetch weather
            weather = get_weather(city)
            weather_context = ""
            if "error" in weather:
                weather_context = f"⚠️ Weather data error: {weather['error']}"
            else:
                weather_context = (
                    f"🌡️ Temperature: {weather['temp']}°C\n"
                    f"💧 Humidity: {weather['humidity']}%\n"
                    f"🌬️ Wind Speed: {weather['wind_speed']} m/s\n"
                    f"🌥️ Condition: {weather['description']}\n"
                )

            # Embedding
            query_embedding = get_embedding(user_message)
            if not query_embedding:
                st.stop()

            # Retrieve from Neo4j
            retrieval_results = find_disease_and_treatments(query_embedding)
            context_text = "Based on your symptoms, here are some possible conditions and treatments:\n"
            for res in retrieval_results:
                context_text += f"\n🦠 Disease: {res['disease']}\n💊 Treatments: {', '.join(res['treatments']) if res['treatments'] else 'None'}\n"

            prompt = """
                You are a medical assistant who provides information about the symptoms of the patient based on the context and conversation history provided to you.
                Also take into account the weather information provided for the symptoms that can be based on weather and can be resolved easily.
                """

            # Prepare full prompt
            

            # Chat history tracking
            history = st.session_state.conversation_history.get(user_id, [])
            history.append({"role": "user", "parts": [f"{user_message}\n\n{context_text}"]})
            if len(history) > 6:
                history = history[-6:]

            full_prompt = f"""
            Role: {prompt} 
            Patient message: {user_message}
            Weather info: {weather_context}
            Retrieved context: {context_text}
            Conversation history : {history}
            """

            # Get Gemini response
            try:
                response = model.generate_content(full_prompt)
                assistant_reply = response.text
                history.append({"role": "model", "parts": [assistant_reply]})
                st.session_state.conversation_history[user_id] = history

                # Show response
                st.markdown("### 💬 Assistant Response")
                st.success(assistant_reply)

                # Show chat history
                with st.expander("🕘 Conversation History"):
                    for turn in history:
                        role = turn["role"]
                        content = turn["parts"][0]
                        st.markdown(f"**{role.capitalize()}**: {content}")

            except Exception as e:
                st.error(f"An error occurred: {e}")


if page == "Knowledge Graph":
        st.markdown("### Click the button below to load interactive sample Knowledge Graph from Neo4j AuraDb.")
        limit = st.number_input("Set Limit for Graph Data", min_value=1, max_value=1000, value=100, step=10)
        if st.button("🔍 Load Graph"):
            graph_data = fetch_graph_data()
            if graph_data:
                #print(f"Graph Data: {graph_data}")
                 graph_html = render_knowledge_graph(graph_data)
                 with open(graph_html, 'r', encoding='utf-8') as f:
                    components.html(f.read(), height=550, scrolling=True)
            else:
                st.warning("No graph data found.")

        st.markdown("### Overview of the Knowledge Graph from Neo4j AuraDb")
        image = Image.open("Images/Disease_graph1.png")
        st.image(image, caption='Disease-Symptom-Treatment Knowledge Graph', use_container_width=True)

        st.markdown("### Zoomed-in view of the Knowledge Graph from Neo4j AuraDb")
        image = Image.open("Images/Disease_graph2.png")
        st.image(image, caption='Zoomed-in View', use_container_width=True)
if page == "Prescription Validation":
    st.markdown("Check if a prescribed drug is valid for a given condition using your medical knowledge graph.")

    with st.form("validation_form"):
        disease_input = st.text_input("Enter Disease Name", placeholder="e.g. Alkylating Agent Cystitis")
        drug_input = st.text_input("Enter Prescribed Drug", placeholder="e.g. citric acid / sodium citrate")
        submitted = st.form_submit_button("Validate")

        if submitted:
            if not disease_input or not drug_input:
                st.warning("Please enter both the disease and drug names.")
            else:
                with st.spinner("Validating..."):
                    is_valid = validate_prescription(disease_input, drug_input)

                if is_valid:
                    st.success(f"✅ '{drug_input}' is a valid treatment for '{disease_input}'.")
                else:
                    st.error(f"❌ '{drug_input}' is NOT listed as a treatment for '{disease_input}' in the graph.")

    st.markdown("### Prescription Knowledge Graph in Neo4j AuraDb")
    image = Image.open("Images/Prescription.png")
    st.image(image, caption='Overview of Prescription Graph', use_container_width=True)

    st.markdown("### Zoomed-in view of Prescription Knowledge Graph in Neo4j AuraDb")
    image = Image.open("Images/Prescription2.png")
    st.image(image, caption='Overview of Prescription Graph', use_container_width=True)

if page == "Conversation Graph":
    st.title("Medical Knowledge Graph Builder")
    st.subheader("Provide the conversation transcription to easily create a knowledge graph about the health of the patient")

    conversation = """
    Doctor:  
    Good afternoon! How can I help you today?

    Patient:  
    Hi, Doctor. I’ve been having this strange pressure in my chest over the last week. I'm a 58-year-old male. I don’t smoke, but I drink occasionally — maybe 2-3 times a week. I used to be more active, but lately I’ve been quite sedentary due to work.

    Doctor:  
    Thanks for that context. When you say pressure in your chest, can you describe it a bit more? Is it sharp, dull, does it spread anywhere?

    Patient:  
    It’s more like a tightness or squeezing sensation. It usually happens when I walk upstairs or after a heavy meal. Sometimes it radiates to my left shoulder and jaw.

    Doctor:  
    That sounds concerning. Any family history of heart disease or high cholesterol?

    Patient:  
    Yes, my father had a heart attack at 60, and my older brother is on medication for high cholesterol and blood pressure.

    Doctor:  
    Okay, given your symptoms and family history, we should definitely run some cardiac tests — an ECG, blood work, and possibly a stress test. Have you had any episodes of sweating, nausea, or shortness of breath during these episodes?

    Patient:  
    Yes, I’ve noticed I get a bit sweaty and light-headed when it happens. It usually lasts for a few minutes and goes away with rest.

    Doctor:  
    Alright. These symptoms suggest it could be angina, which is a sign of reduced blood flow to the heart. We’ll need to rule out any risk of coronary artery disease. I’ll get the necessary tests ordered today.

    Patient:  
    That sounds serious… should I be worried?

    Doctor:  
    It’s good that you came in when you noticed the symptoms. If it is angina, we can manage it and prevent it from progressing. The key is early detection and lifestyle changes. We’ll also talk about modifying your diet, adding mild activity, and possibly starting you on medication depending on the results.

    Patient:  
    Thank you, Doctor. I’m glad I decided to check in.

    Doctor:  
    Absolutely — your health is worth staying ahead of. Let’s get started with the tests and take it from there.
    """


    if st.button("📄 Load Sample Conversation"):
        st.session_state.transcription_text = conversation

    transcription_input = st.text_area("Enter Doctor-Patient Conversation Transcription", height=300, value=st.session_state.get("transcription_text", ""))

    if st.button("Generate Knowledge Graph"):
        if transcription_input:
            document = Document(page_content=transcription_input)
            graph_documents = graph_transformer.convert_to_graph_documents([document])

            # Create triplets
            triplets = []
            for doc in graph_documents:
                for rel in doc.relationships:
                    triplets.append((
                        rel.source.id,     # subject
                        rel.type,          # predicate
                        rel.target.id      # object
                    ))

            # Add graph to Neo4j
            add_graph_to_neo4j(graph_documents)

            # Fetch and render the graph
            graph_data = fetch_graph_data2()
            if graph_data:
                graph_html = render_knowledge_graph2(graph_data)
                with open(graph_html, 'r', encoding='utf-8') as f:
                    st.components.v1.html(f.read(), height=600, scrolling=True)
            else:
                st.warning("No graph data found.")
        else:
            st.warning("Please enter a transcription.")

    st.markdown("### This feature can be easily integrated with a Voice-to-text model to provide a seemless pipeline for making the documentation easy.")