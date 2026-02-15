"""
This is the Streamlit app for Nutshell.io, an intelligence engine for AI/ML engineers. 
It connects to a Qdrant vector database to fetch and display deduplicated and synthesized news updates from various sources. 
The app features a sidebar for controls and a main feed that highlights trending news based on mention counts.
"""

import streamlit as st
from qdrant_client import QdrantClient
from dotenv import load_dotenv
load_dotenv() 
import os

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "test_collection")
print(f"Using collection: {COLLECTION_NAME}")

# Page Config
st.set_page_config(page_title="Nutshell.io | AI Intelligence", layout="wide")

# Styling
st.markdown("""
    <style>
    .news-card {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
        margin-bottom: 15px;
        background-color: #0e1117;
    }
    .mention-badge {
        background-color: #ff4b4b;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Connection to Qdrant
client = QdrantClient(host="localhost", port=6333)

def get_all_nutshells():
    # We use scroll to get points without needing a search vector
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        with_payload=True,
        limit=50
    )
    return [p.payload for p in points]

# --- UI Layout ---
st.title("🥜 Nutshell.io")
st.caption("Intelligence engine for AI/ML engineers. Deduplicated & Synthesized.")

with st.sidebar:
    st.header("Controls")
    if st.button("Refresh Feed"):
        st.rerun()
    st.info("Currently monitoring: TLDR, The Neuron, TAAFT, What's AI.")

nutshells = get_all_nutshells()

if not nutshells:
    st.warning("No intelligence processed yet. Run your ingestion script!")
else:
    # Sort by mention_count to show "Trending" news first
    sorted_news = sorted(nutshells, key=lambda x: x.get('mention_count', 1), reverse=True)

    for item in sorted_news:
        with st.container():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.subheader(item.get('headline', 'Untitled Update'))
                st.write(item.get('summary', 'No summary available.'))
                
                # Show all merged links
                st.markdown("**Sources:**")
                links = item.get('links', [])
                for i, link in enumerate(links):
                    st.markdown(f"- [Source {i+1}]({link})")
            
            with col2:
                # Mention Count Badge
                mentions = item.get('mention_count', 1)
                st.markdown(f"<span class='mention-badge'>{mentions} Sources Mentioned</span>", unsafe_allow_html=True)
                st.metric("Significance", f"{item.get('relevance_score', 0)}/10")
                st.write(f"🏷️ `{item.get('category', 'General')}`")
            
            st.divider()