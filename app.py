import streamlit as st
import spacy
import requests
from bs4 import BeautifulSoup
import configparser
import random
import spacy_streamlit
from string import punctuation
from heapq import nlargest

nlp = spacy.load("en_core_web_sm")
stopwords = list(spacy.lang.en.stop_words.STOP_WORDS)
punctuation += "\n"

config = configparser.ConfigParser()
config.read("config.ini")
news_api_key = config["API"]["news_api"]

st.set_page_config(
    page_title="Article News Suggestion",
    page_icon="\U0001F9CA",
    layout="wide",
)

def spacy_render(summary):
    """Visualize Named Entity Recognition (NER) using Spacy."""
    summ = nlp(summary)
    spacy_streamlit.visualize_ner(summ, labels=nlp.get_pipe("ner").labels, title="Summary Visualization", show_table=False, key=random.randint(0, 100))

def word_frequency(doc):
    """Calculate word frequencies in the document."""
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            word_frequencies[word.text] = word_frequencies.get(word.text, 0) + 1
    return word_frequencies

def sentence_score(sentence_tokens, word_frequencies):
    """Calculate sentence scores based on word frequencies."""
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word.text.lower()]
    return sentence_scores

@st.cache_data
def fetch_news_links(query, source="cnn", num_articles=10):
    """Fetch news article links, titles, and thumbnails using the News API."""
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&apiKey={news_api_key}" if query else \
        f"https://newsapi.org/v2/top-headlines?sources={source}&language=en&apiKey={news_api_key}"

    response = requests.get(url).json()
    articles = response.get("articles", [])
    
    links, titles, thumbnails = [], [], []
    for article in articles[:num_articles]:
        links.append(article["url"])
        titles.append(article["title"])
        thumbnails.append(article.get("urlToImage", ""))
    
    return links, titles, thumbnails

@st.cache_data
def fetch_news(link_list):
    """Fetch news content from provided links."""
    news_list = []
    for link in link_list:
        try:
            response = requests.get(link)
            soup = BeautifulSoup(response.content, "html.parser")
            paragraphs = soup.find_all("p")
            news_list.append(" ".join([p.get_text() for p in paragraphs]))
        except Exception as e:
            news_list.append(f"Could not fetch content from {link}. Error: {e}")
    return news_list

def get_summary(text, summary_length):
    """Generate a summary of the text based on the specified length."""
    doc = nlp(text)
    word_frequencies = word_frequency(doc)
    max_freq = max(word_frequencies.values(), default=1)
    word_frequencies = {word: freq / max_freq for word, freq in word_frequencies.items()}

    sentence_tokens = [sent for sent in doc.sents]
    sentence_scores = sentence_score(sentence_tokens, word_frequencies)
    sorted_sentences = nlargest(summary_length, sentence_scores, key=sentence_scores.get)

    return " ".join([sent.text for sent in sorted_sentences]).strip()

# Main Application Logic
st.title("Article News Suggestion")
st.write("Get summaries and headlines for the latest news articles on your favorite topics.")
search_query = st.text_input("Search News", placeholder="Enter the topic you want to search")

if search_query:
    links, titles, thumbnails = fetch_news_links(search_query)
    articles = fetch_news(links)

    if links:
        col1, col2 = st.columns(2)
        for i, (link, title, thumbnail, article) in enumerate(zip(links, titles, thumbnails, articles)):
            with (col1 if i % 2 == 0 else col2):
                st.image(thumbnail, use_container_width=True)
                st.write(title)
                with st.expander("Read The Summary"):
                    st.write(get_summary(article, summary_length=50))
                st.markdown(f"[**Read Full Article**]({link})", unsafe_allow_html=True)
    else:
        st.info(f"No results found for '{search_query}'. Please try another keyword.")
else:
    st.info("Please enter a search query to get news summaries.")
