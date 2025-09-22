# app.py
import os
import base64
import tempfile
import time
import re
import streamlit as st
from PyPDF2 import PdfReader
from PIL import Image
from moviepy import editor as mp
import google.generativeai as genai
from gtts import gTTS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# =======================
# 1. API Keys
# =======================
GOOGLE_API_KEY = "AIzaSyAqJpwzrd9MWTD3kgnKwhRb3l_dxFweCH8"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# =======================
# 2. LLM + Embeddings
# =======================
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}   # ✅ force CPU
)
# =======================
# 3. Vector Store
# =======================
vectorstore = None

def add_to_vectorstore(text, source=""):
    """Add text into global vectorstore"""
    global vectorstore
    if not text.strip():
        return
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = [Document(page_content=chunk, metadata={"source": source}) for chunk in splitter.split_text(text)]
    if vectorstore:
        vectorstore.add_documents(docs)
    else:
        vectorstore = Chroma.from_documents(docs, embeddings)

def retrieve_context(query):
    if not vectorstore:
        return ""
    results = vectorstore.similarity_search(query, k=3)
    return "\n".join([r.page_content for r in results])

# =======================
# 4. Helpers
# =======================
def pil_to_base64(img):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    img.save(tmp.name, format="PNG")
    with open(tmp.name, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def process_video(path):
    clip = mp.VideoFileClip(path)
    frame = clip.get_frame(clip.duration / 2)
    return Image.fromarray(frame)

def transcribe_with_gemini(file_path):
    """Send audio/video file to Gemini for transcription"""
    model = genai.GenerativeModel("gemini-1.5-flash")
    uploaded = genai.upload_file(file_path)

    while uploaded.state.name == "PROCESSING":
        time.sleep(2)
        uploaded = genai.get_file(uploaded.name)

    if uploaded.state.name != "ACTIVE":
        return None

    response = model.generate_content([uploaded, "Transcribe the speech from this file."])
    return response.text

def translate_text(text, target_lang):
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(f"Translate this text to {target_lang}: {text}")
    return resp.text

def clean_text_for_speech(text):
    """Fix punctuation for better TTS output"""
    text = re.sub(r"\s+", " ", text)        # collapse spaces
    text = text.replace("?", " ?")
    text = text.replace(".", ". ")
    text = text.replace(",", ", ")
    return text.strip()

def speak_text(text, lang="en", filename="output.mp3"):
    text = clean_text_for_speech(text)
    tts = gTTS(text=text, lang=lang)
    tts.save(filename)
    return filename

# =======================
# 5. Streamlit UI
# =======================
st.set_page_config(page_title="📚 Multi-Modal RAG + AV Assistant", layout="centered")

st.title("📚 Multi-Modal RAG + 🎙️ Audio/Video Assistant")

uploaded_files = st.file_uploader(
    "Upload Files (PDF, TXT, IMG, AUDIO, VIDEO)",
    type=["pdf", "txt", "png", "jpg", "jpeg",
          "mp3", "wav", "m4a", "flac", "ogg", "opus", "aac",
          "mp4", "mov", "avi", "mkv", "wmv", "webm"],
    accept_multiple_files=True
)

query = st.text_input("🔎 Enter your question")

LANG_CODE_MAP = {
    "English": "en",
    "Telugu": "te",
    "Hindi": "hi",
    "French": "fr",
    "German": "de",
    "Spanish": "es"
}

# =======================
# 6. Process Files
# =======================
if uploaded_files:
    for uploaded_file in uploaded_files:
        path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        ext = os.path.splitext(uploaded_file.name)[1].lower()

        if ext == ".pdf":
            pdf_reader = PdfReader(path)
            text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
            add_to_vectorstore(text, source=uploaded_file.name)

        elif ext == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            add_to_vectorstore(text, source=uploaded_file.name)

        elif ext in [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac",
                     ".mp4", ".mov", ".avi", ".mkv", ".wmv", ".webm"]:
            transcription = transcribe_with_gemini(path)
            if transcription:
                st.subheader(f"📜 Transcription from {uploaded_file.name}")
                st.write(transcription)
                add_to_vectorstore(transcription, source=uploaded_file.name)

        elif ext in [".png", ".jpg", ".jpeg"]:
            img = Image.open(path)
            st.image(img, caption=f"🖼️ {uploaded_file.name}")
            vision = llm.invoke([{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail."},
                    {"type": "image_url", "image_url": f"data:image/png;base64,{pil_to_base64(img)}"}
                ]
            }])
            if vision.content:
                st.subheader(f"📝 Image Description for {uploaded_file.name}")
                st.write(vision.content)
                add_to_vectorstore(vision.content, source=uploaded_file.name)

# =======================
# 7. Ask Questions
# =======================
if query:
    context = retrieve_context(query)
    if context.strip():
        response = llm.invoke(query + "\n\nContext:\n" + context)
        st.success(response.content)

        # ---- Speak Answer ----
        target_lang = st.selectbox("🌍 Speak Answer in:", list(LANG_CODE_MAP.keys()), key="speak_lang")
        if st.button("🔊 Speak Answer"):
            lang_code = LANG_CODE_MAP.get(target_lang, "en")
            mp3 = speak_text(response.content, lang=lang_code, filename="final_answer.mp3")
            st.audio(mp3)
    else:
        st.warning("❌ I don’t know (no relevant context found).")
