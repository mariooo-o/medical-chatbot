import re
import random
import warnings

import nltk
import numpy as np
import pandas as pd
import streamlit as st

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MedBot v3 — Smart Medical Assistant",
    page_icon="🏥",
    layout="centered",
)


# ============================================================
# NLTK RESOURCES
# ============================================================

@st.cache_resource
def setup_nltk():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
    ]

    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


setup_nltk()


# ============================================================
# SENTENCE-BERT
# ============================================================

@st.cache_resource
def load_sbert():
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        return model, True
    except Exception:
        return None, False


SBERT_MODEL, USE_SBERT = load_sbert()


# ============================================================
# DATASET
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("medicalqa_clean.csv")

    required_columns = {"category", "question", "answer"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Kolom dataset yang diperlukan tidak ditemukan: {', '.join(sorted(missing))}"
        )

    return df


df = load_data()


# ============================================================
# TEXT PREPROCESSING
# ============================================================

lemmatizer = WordNetLemmatizer()

indonesian_stopwords = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk",
    "pada", "adalah", "atau", "juga", "dalam", "tidak", "akan", "ada",
    "saya", "kamu", "anda", "ia", "mereka", "kami", "kita", "bisa",
    "sudah", "bila", "jika", "maka", "oleh", "karena", "apa",
    "bagaimana", "berapa", "kapan", "dimana", "siapa", "apakah", "cara",
    "lebih", "sangat", "dapat", "nya", "pun", "lagi", "belum",
    "telah", "namun", "tapi", "serta", "meski", "agar", "supaya", "hal",
    "the", "is", "are", "was", "what", "how", "why", "when", "where"
}

english_stopwords = set(stopwords.words("english"))
all_stopwords = indonesian_stopwords | english_stopwords


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [
        t for t in tokens
        if t not in all_stopwords and len(t) > 2
    ]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


df["processed_question"] = df["question"].apply(preprocess_text)


# ============================================================
# MEDICAL CHATBOT ENGINE
# ============================================================

class MedicalChatbotEngineV3:

    def __init__(self, dataframe, threshold=0.35, top_k=3):
        self.df = dataframe
        self.threshold = threshold if USE_SBERT else 0.15
        self.top_k = top_k
        self.conversation_history = []

        self._build_index()
        self._define_rules()

    def _build_index(self):
        if USE_SBERT:
            self.sbert_embeddings = SBERT_MODEL.encode(
                self.df["question"].tolist(),
                convert_to_tensor=True
            )

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df["processed_question"]
        )

    def _define_rules(self):
        self.rules = {
            "emergency": {
                "patterns": [
                    r"(sesak.*berat|nyeri dada.*berat|tidak.*bernapas|pingsan)",
                ],
                "responses": [
                    "🚨 DARURAT! Hubungi 119 atau segera ke IGD!"
                ]
            },
            "greeting": {
                "patterns": [
                    r"\b(halo|hai|hi|hello)\b"
                ],
                "responses": [
                    "👋 Halo! Ada yang bisa saya bantu?"
                ]
            }
        }

    def _check_rules(self, text):
        for _, data in self.rules.items():
            for pattern in data["patterns"]:
                if re.search(pattern, text.lower()):
                    return random.choice(data["responses"])

        return None

    def _search_sbert(self, query):
        from sentence_transformers import util

        emb = SBERT_MODEL.encode(
            query,
            convert_to_tensor=True
        )

        scores = util.cos_sim(
            emb,
            self.sbert_embeddings
        )[0]

        scores = scores.cpu().numpy()
        top_results = np.argsort(-scores)[:self.top_k]

        return [
            (idx, float(scores[idx]))
            for idx in top_results
        ]

    def _search_tfidf(self, query):
        processed = preprocess_text(query)
        vec = self.vectorizer.transform([processed])

        scores = cosine_similarity(
            vec,
            self.tfidf_matrix
        ).flatten()

        top_results = np.argsort(scores)[::-1][:self.top_k]

        return [
            (idx, scores[idx])
            for idx in top_results
        ]

    def _find_best_match(self, query):
        if USE_SBERT:
            results = self._search_sbert(query)
        else:
            results = self._search_tfidf(query)

        best_idx, best_score = results[0]
        return best_idx, best_score

    def _build_context_query(self, user_input):
        if len(self.conversation_history) > 0:
            last_input = self.conversation_history[-1]
            return last_input + " " + user_input

        return user_input

    def get_response(self, user_input):
        if not user_input.strip():
            return "Silakan ketik pertanyaan."

        rule = self._check_rules(user_input)

        if rule:
            return rule

        query = self._build_context_query(user_input)

        if USE_SBERT:
            results = self._search_sbert(query)
            method = "SBERT"
        else:
            results = self._search_tfidf(query)
            method = "TF-IDF"

        best_idx, best_score = results[0]

        if best_score < self.threshold:
            return "🤔 Tidak menemukan jawaban yang cukup relevan."

        boosted = []

        for idx, score in results:
            text = self.df.iloc[idx]["question"]

            bonus = sum(
                1
                for word in user_input.split()
                if word in text
            )

            boosted.append(
                (idx, score + 0.05 * bonus)
            )

        best_idx = sorted(
            boosted,
            key=lambda x: x[1],
            reverse=True
        )[0][0]

        row = self.df.iloc[best_idx]

        self.conversation_history.append(user_input)

        return f"""[Kategori: {row["category"]} | {method}]

{row["answer"]}

─────────────────
⚠️ Untuk kondisi serius, konsultasikan ke dokter.
"""


# ============================================================
# BOT INSTANCE
# ============================================================

@st.cache_resource
def create_bot():
    return MedicalChatbotEngineV3(df)


bot = create_bot()


# ============================================================
# STREAMLIT UI
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background: #f8faff;
    }

    .header {
        background: linear-gradient(
            135deg,
            #1a1f5e,
            #2d3a8c,
            #1565c0
        );
        padding: 22px 24px;
        border-radius: 16px;
        color: white;
        margin-bottom: 12px;
    }

    .header h1 {
        margin: 0;
        font-size: 28px;
    }

    .header p {
        margin: 6px 0 0 0;
        opacity: 0.85;
    }

    .disclaimer {
        background: #fff8e1;
        padding: 12px 16px;
        border-left: 4px solid #f59e0b;
        border-radius: 6px;
        color: #78350f;
        font-size: 13px;
        margin-bottom: 18px;
    }

    .status {
        font-size: 13px;
        color: #15803d;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

mode = "Sentence-BERT" if USE_SBERT else "TF-IDF"

st.markdown(
    f"""
    <div class="header">
        <h1>🏥 MedBot v3 — Smart Medical Assistant</h1>
        <p>🟢 Online · Context-Aware Semantic Engine · {mode}</p>
    </div>

    <div class="disclaimer">
        ⚠️ <b>Edukasi saja.</b> MedBot bukan pengganti dokter.
        Untuk keadaan darurat medis, segera hubungi 119 atau pergi ke IGD.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CHAT STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("🏥 MedBot")

    st.write(
        "Asisten kesehatan berbasis semantic search "
        "untuk membantu menemukan jawaban dari dataset medis."
    )

    st.divider()

    st.write(f"**Model:** {mode}")
    st.write(f"**Total Q&A:** {len(df)}")
    st.write(
        f"**Kategori:** {df['category'].nunique()}"
    )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):
        st.session_state.messages = []
        bot.conversation_history = []
        st.rerun()


# ============================================================
# QUICK QUESTIONS
# ============================================================

st.subheader("💡 Quick Questions")

quick_questions = [
    ("🦠 Rabies", "symptoms of rabies"),
    ("🩸 Diabetes", "diabetes"),
    ("🧬 Genetic Disease", "genetic disease inherited"),
    ("🪱 Parasites", "parasites treatment"),
    ("🛡️ Prevention", "how to prevent disease"),
]

cols = st.columns(len(quick_questions))

for col, (label, question) in zip(cols, quick_questions):
    if col.button(
        label,
        use_container_width=True
    ):
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.spinner("MedBot sedang mencari jawaban..."):
            response = bot.get_response(question)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

        st.rerun()


# ============================================================
# CHAT HISTORY
# ============================================================

st.subheader("💬 Chat")

if not st.session_state.messages:
    st.info(
        "Halo! 👋 Silakan tanyakan sesuatu tentang kesehatan, "
        "atau pilih salah satu Quick Question di atas."
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Ketik pertanyaan kesehatan..."
)

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("MedBot sedang mengetik..."):
            response = bot.get_response(user_input)

        st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
