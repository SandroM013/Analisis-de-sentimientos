import streamlit as st
import joblib, nltk, re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import ToktokTokenizer, word_tokenize
from nltk.tag import pos_tag

st.set_page_config(page_title="Análisis de Sentimientos de Reseñas de IMDb", layout="wide")

@st.cache_resource(show_spinner=False)
def ensure_nltk():
    pkgs = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
    ]
    for path, name in pkgs:
        try: nltk.data.find(path)
        except LookupError: nltk.download(name)

@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load('logistic_regression_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    return model, vectorizer

ensure_nltk()
try:
    loaded_model, loaded_vectorizer = load_artifacts()
except FileNotFoundError:
    st.error("Faltan 'logistic_regression_model.joblib' y/o 'tfidf_vectorizer.joblib' en la carpeta.")
    loaded_model = loaded_vectorizer = None

tokenizer = ToktokTokenizer()
STOPWORDS = set(stopwords.words('english'))
wnl = WordNetLemmatizer()
EMOJI_RE = re.compile("["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF" "]+", flags=re.UNICODE)

def strip_html(text): return BeautifulSoup(text, "html.parser").get_text()

def limpiar_texto(texto):
    if not isinstance(texto, str): texto = str(texto)
    texto = strip_html(texto)
    texto = re.sub(r"http\S+", "", texto)
    texto = EMOJI_RE.sub("", texto)
    texto = texto.encode('ascii', 'ignore').decode('ascii', errors='ignore')
    texto = re.sub('\[[^]]*\]', '', texto)
    texto = re.sub(r'[^a-zA-Z0-9\s]', '', texto)
    return texto.lower()

def remove_stopwords(text):
    toks = [t.strip() for t in tokenizer.tokenize(text)]
    return ' '.join([t for t in toks if t and t not in STOPWORDS])

def lemmatize_all(sentence):
    try:
        tagged = pos_tag(word_tokenize(sentence))
    except LookupError:
        return sentence
    out = []
    for w, tag in tagged:
        if   tag.startswith("NN"): out.append(wnl.lemmatize(w, pos='n'))
        elif tag.startswith("VB"): out.append(wnl.lemmatize(w, pos='v'))
        elif tag.startswith("JJ"): out.append(wnl.lemmatize(w, pos='a'))
        else: out.append(w)
    return ' '.join(out)

def pipeline_text(x):
    return lemmatize_all(remove_stopwords(limpiar_texto(x)))

def predict_sentiment(review_text):
    if loaded_model is None or loaded_vectorizer is None:
        return None, "Modelo no cargado."
    Xv = loaded_vectorizer.transform([pipeline_text(review_text)])
    y_pred = loaded_model.predict(Xv)[0]
    try:
        proba = float(loaded_model.predict_proba(Xv).max())
    except Exception:
        proba = None
    label_map = {1:'positive', 0:'negative', 'positive':'positive', 'negative':'negative'}
    label = label_map.get(y_pred, str(y_pred))
    return label, proba

st.title("Análisis de Sentimientos de Reseñas de IMDb")
st.markdown("Predice si una reseña es **positiva** o **negativa** (Regresión Logística + TF-IDF).")

review_input = st.text_area("Ingresa tu reseña aquí:", height=150, placeholder="Type your IMDB review...")

if st.button("Predecir Sentimiento"):
    if not review_input.strip():
        st.warning("Por favor, ingresa una reseña.")
    else:
        label, proba = predict_sentiment(review_input)
        if label is None:
            st.error(proba)
        else:
            if label == 'positive':
                st.markdown(f"**Sentimiento:** :green[Positivo] 😊  {f'· Confianza: {proba:.2%}' if proba else ''}")
            else:
                st.markdown(f"**Sentimiento:** :red[Negativo] 😞  {f'· Confianza: {proba:.2%}' if proba else ''}")

st.caption("Desarrollado con Streamlit · scikit-learn · NLTK")
