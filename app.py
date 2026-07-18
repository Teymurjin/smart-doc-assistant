"""
Smart Document Assistant — RAG с Groq API
==========================================
ИСПРАВЛЕННАЯ ВЕРСИЯ: обновлены модели Groq (июль 2026)

Устаревшие модели (УДАЛЕНЫ из Groq):
  ❌ llama-3.1-70b-versatile — decommissioned
  ❌ mixtral-8x7b-32768 — deprecated

Актуальные модели (РАБОТАЮТ):
  ✅ llama-3.3-70b-versatile — замена старой 70B
  ✅ llama-3.1-8b-instant — быстрая, высокие лимиты
  ✅ openai/gpt-oss-120b — новая, лучшее качество
  ✅ openai/gpt-oss-20b — новая, быстрая

Установка:
    pip install -r requirements.txt

Запуск:
    streamlit run app.py
"""

import os
import tempfile
from typing import List

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# ИМПОРТЫ (LangChain 0.2.x)
# =============================================================================

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

# =============================================================================
# НАСТРОЙКИ
# =============================================================================

st.set_page_config(
    page_title="Smart Doc Assistant — FREE",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# =============================================================================
# ФУНКЦИИ
# =============================================================================

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_document(file_path: str, file_type: str):
    file_type = file_type.lower().strip()
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
        return loader.load()
    elif file_type == "txt":
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()
    elif file_type in ("docx", "doc"):
        loader = Docx2txtLoader(file_path)
        return loader.load()
    else:
        raise ValueError(f"Неподдерживаемый формат: '{file_type}'")


def split_documents(documents: List):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    return splitter.split_documents(documents)


def create_vectorstore(chunks, embeddings):
    return FAISS.from_documents(chunks, embeddings)


def get_llm(groq_api_key: str, model_name: str, temperature: float):
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name,
        temperature=temperature,
    )


def create_qa_chain(vectorstore, llm, memory):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        chain_type="stuff",
    )
    return chain


def format_sources(source_docs) -> str:
    if not source_docs:
        return ""
    lines = []
    for i, doc in enumerate(source_docs, 1):
        source = doc.metadata.get("source", "Неизвестно")
        page = doc.metadata.get("page", "—")
        preview = doc.page_content[:300].replace("\n", " ")
        if len(doc.page_content) > 300:
            preview += "..."
        lines.append(f"**[{i}]** 📄 `{source}` (стр. {page})\n\n> {preview}\n")
    return "\n---\n".join(lines)


# =============================================================================
# SESSION STATE
# =============================================================================

def init_state():
    defaults = {
        "qa_chain": None,
        "memory": None,
        "processed": [],
        "messages": [],
        "docs_loaded": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_state()

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("⚙️ Настройки")

    st.subheader("🔑 Groq API Key")
    groq_key = st.text_input(
        "Введите ключ",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Бесплатно: https://console.groq.com/keys",
    )
    if not groq_key:
        st.warning("⚠️ Ключ не введён")

    st.divider()

    # ========== ИСПРАВЛЕННЫЙ СПИСОК МОДЕЛЕЙ ==========
    st.subheader("🧠 Модель")
    model_choice = st.selectbox(
        "Выберите LLM",
        [
            "llama-3.3-70b-versatile",      # ЗАМЕНА старой llama-3.1-70b
            "llama-3.1-8b-instant",          # Быстрая, высокие лимиты
            "openai/gpt-oss-120b",           # Новая, лучшее качество
            "openai/gpt-oss-20b",            # Новая, быстрая
        ],
        index=0,
        help=(
            "llama-3.3-70b — замена удалённой llama-3.1-70b.\n"
            "gpt-oss-120b — новейшая модель от OpenAI на Groq.\n"
            "gpt-oss-20b — быстрая и умная."
        ),
    )

    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)

    st.divider()

    st.subheader("📁 Документы")
    uploaded_files = st.file_uploader(
        "PDF, TXT или DOCX",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("🚀 Обработать", type="primary", use_container_width=True):
        if not groq_key:
            st.error("❌ Введите Groq API ключ!")
        else:
            with st.spinner("Индексация..."):
                all_chunks = []
                processed_names = []

                for uploaded_file in uploaded_files:
                    ext = uploaded_file.name.split(".")[-1].lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    try:
                        docs = load_document(tmp_path, ext)
                        chunks = split_documents(docs)
                        all_chunks.extend(chunks)
                        processed_names.append(uploaded_file.name)
                    except Exception as e:
                        st.error(f"Ошибка: {uploaded_file.name}: {e}")
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass

                if all_chunks:
                    try:
                        embeddings = get_embeddings()
                        vectorstore = create_vectorstore(all_chunks, embeddings)
                        memory = ConversationBufferMemory(
                            memory_key="chat_history",
                            return_messages=True,
                            output_key="answer",
                        )
                        llm = get_llm(groq_key, model_choice, temperature)
                        qa_chain = create_qa_chain(vectorstore, llm, memory)

                        st.session_state.qa_chain = qa_chain
                        st.session_state.memory = memory
                        st.session_state.processed = processed_names
                        st.session_state.messages = []
                        st.session_state.docs_loaded = True

                        st.success(f"✅ Готово! Файлов: {len(processed_names)}, чанков: {len(all_chunks)}")
                    except Exception as e:
                        st.error(f"Ошибка цепочки: {e}")
                else:
                    st.error("Не удалось извлечь текст.")

    if st.session_state.processed:
        st.divider()
        st.caption("**Загружено:**")
        for name in st.session_state.processed:
            st.caption(f"• {name}")
        if st.button("🗑️ Очистить всё", use_container_width=True):
            for key in ["qa_chain", "memory", "processed", "messages"]:
                st.session_state[key] = None if key != "messages" else []
            st.session_state.docs_loaded = False
            st.rerun()

    st.divider()
    st.info("1. Получите ключ → console.groq.com\n2. Вставьте выше\n3. Загрузите файлы\n4. Задавайте вопросы")

# =============================================================================
# MAIN
# =============================================================================

st.title("📚 Smart Document Assistant")
st.caption("RAG + Groq | Бесплатно | Исправлено: актуальные модели 2026")

if not st.session_state.docs_loaded:
    st.info("""
    ### 👋 Добро пожаловать!

    **Бесплатный ассистент для работы с документами**

    - 📖 PDF, TXT, DOCX
    - 🔍 Поиск по содержимому (RAG)
    - 💬 Память диалога
    - 📌 Источники ответов
    - 💰 **$0** (Groq бесплатный тир)

    **Начните с боковой панели слева ↑**
    """)
    st.stop()

st.success("✅ Документы загружены! Задавайте вопросы.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📌 Источники"):
                st.markdown(msg["sources"])

if prompt := st.chat_input("Задайте вопрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Думаю..."):
            try:
                result = st.session_state.qa_chain({"question": prompt})
                answer = result["answer"]
                source_docs = result.get("source_documents", [])
                sources_md = format_sources(source_docs)

                st.markdown(answer)
                if sources_md:
                    with st.expander("📌 Источники ответа"):
                        st.markdown(sources_md)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources_md,
                })
            except Exception as e:
                err = f"❌ Ошибка: {str(e)}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err, "sources": ""})

if st.session_state.get("messages"):
    st.divider()
    if st.button("🗑️ Очистить чат"):
        st.session_state.messages = []
        if st.session_state.memory:
            st.session_state.memory.clear()
        st.rerun()

st.divider()
st.caption("🔬 Проект по курсу Data Science | RAG + Groq + Streamlit")