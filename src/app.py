"""
This is the Streamlit app for Nutshell.io, an intelligence engine for AI/ML engineers. 
It connects to a Qdrant vector database to fetch and display deduplicated and synthesized news updates from various sources. 
The app features a sidebar for controls and a main feed that highlights trending news based on mention counts.
"""

import streamlit as st
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
load_dotenv() 
import os

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "nutshell")

# Page Config
st.set_page_config(page_title="Nutshell.io | AI Intelligence", layout="wide", page_icon="🥜")

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
    .tag-pill {
        background-color: #1f77b4;
        color: white;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 0.75rem;
        margin: 2px;
        display: inline-block;
    }
    .company-pill {
        background-color: #2ca02c;
        color: white;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 0.75rem;
        margin: 2px;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

# Connection to Qdrant
@st.cache_resource
def get_qdrant_client():
    return QdrantClient(host="localhost", port=6333)

client = get_qdrant_client()

def get_all_nutshells(category_filter=None, min_relevance=None, tag_filter=None, company_filter=None):
    """Fetch insights with optional filters."""
    filter_conditions = []
    
    if category_filter and category_filter != "All":
        filter_conditions.append(models.FieldCondition(key="category", match=models.MatchValue(value=category_filter)))
    if min_relevance:
        filter_conditions.append(models.FieldCondition(key="relevance_score", range=models.Range(gte=min_relevance)))
    if tag_filter and tag_filter != "All":
        filter_conditions.append(models.FieldCondition(key="tags", match=models.MatchAny(any=[tag_filter])))
    if company_filter and company_filter != "All":
        filter_conditions.append(models.FieldCondition(key="companies_mentioned", match=models.MatchAny(any=[company_filter])))
    
    query_filter = models.Filter(must=filter_conditions) if filter_conditions else None
    
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=query_filter,
        with_payload=True,
        limit=100
    )
    return [p.payload for p in points]

def get_unique_values(field: str):
    """Get unique values for a field across all insights."""
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        with_payload=True,
        limit=100
    )
    values = set()
    for p in points:
        if field in p.payload:
            if isinstance(p.payload[field], list):
                values.update(p.payload[field])
            else:
                values.add(p.payload[field])
    return sorted(list(values))

# --- UI Layout ---
st.title("🥜 Nutshell.io")
st.caption("Intelligence engine for AI/ML engineers. Deduplicated & Synthesized.")

with st.sidebar:
    st.header("🎛️ Filters")
    
    # Category filter
    categories = ["All"] + get_unique_values("category")
    category_filter = st.selectbox("Category", categories, index=0)
    
    # Relevance filter
    min_relevance = st.slider("Min Relevance Score", 1, 10, 1)
    
    # Tag filter
    tags = ["All"] + get_unique_values("tags")
    tag_filter = st.selectbox("Tag", tags, index=0)
    
    # Company filter
    companies = ["All"] + get_unique_values("companies_mentioned")
    company_filter = st.selectbox("Company", companies, index=0)
    
    st.divider()
    
    # Sort options
    sort_by = st.radio("Sort by", ["Trending (Mentions)", "Relevance Score", "Recent"])
    
    st.divider()
    
    if st.button("🔄 Refresh Feed", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.info("📡 Monitoring: TLDR, The Neuron, TAAFT, What's AI")

# Fetch insights with filters
nutshells = get_all_nutshells(
    category_filter=category_filter if category_filter != "All" else None,
    min_relevance=min_relevance if min_relevance > 1 else None,
    tag_filter=tag_filter if tag_filter != "All" else None,
    company_filter=company_filter if company_filter != "All" else None
)

if not nutshells:
    st.warning("⚠️ No intelligence matching your filters. Try adjusting the filters or run your ingestion script!")
else:
    # Sort insights
    if sort_by == "Trending (Mentions)":
        sorted_news = sorted(nutshells, key=lambda x: x.get('mention_count', 1), reverse=True)
    elif sort_by == "Relevance Score":
        sorted_news = sorted(nutshells, key=lambda x: x.get('relevance_score', 0), reverse=True)
    else:  # Recent
        sorted_news = sorted(nutshells, key=lambda x: x.get('processed_at', ''), reverse=True)
    
    st.subheader(f"📰 {len(sorted_news)} Insights Found")
    
    for item in sorted_news:
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader(item.get('headline', 'Untitled Update'))
                st.write(item.get('summary', 'No summary available.'))
                
                # Tags
                tags = item.get('tags', [])
                if tags:
                    st.markdown("**Tags:**")
                    tags_html = " ".join([f"<span class='tag-pill'>{tag}</span>" for tag in tags])
                    st.markdown(tags_html, unsafe_allow_html=True)
                
                # Companies mentioned
                companies = item.get('companies_mentioned', [])
                if companies:
                    st.markdown("**Companies:**")
                    companies_html = " ".join([f"<span class='company-pill'>{comp}</span>" for comp in companies])
                    st.markdown(companies_html, unsafe_allow_html=True)
                
                # Key people
                key_people = item.get('key_people', [])
                if key_people:
                    st.markdown(f"**Key People:** {', '.join(key_people)}")
                
                # Show all merged links
                st.markdown("**Sources:**")
                links = item.get('links', [])
                for i, link in enumerate(links, 1):
                    st.markdown(f"- [Source {i}]({link})")
            
            with col2:
                # Mention Count Badge
                mentions = item.get('mention_count', 1)
                st.markdown(f"<span class='mention-badge'>🔥 {mentions} Mentions</span>", unsafe_allow_html=True)
                st.metric("Relevance", f"{item.get('relevance_score', 0)}/10")
                st.write(f"🏷️ `{item.get('category', 'General')}`")
                
                # Newsletter source info
                newsletter_name = item.get('newsletter_name', 'Unknown')
                st.caption(f"📬 {newsletter_name}")
            
            st.divider()
    
    # Stats at the bottom
    with st.expander("📊 Quick Stats"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Insights", len(sorted_news))
        with col2:
            avg_relevance = sum(n.get('relevance_score', 0) for n in sorted_news) / len(sorted_news) if sorted_news else 0
            st.metric("Avg Relevance", f"{avg_relevance:.1f}/10")
        with col3:
            trending_count = len([n for n in sorted_news if n.get('mention_count', 1) > 1])
            st.metric("Trending Items", trending_count)
        with col4:
            unique_categories = len(set(n.get('category', 'General') for n in sorted_news))
            st.metric("Categories", unique_categories)