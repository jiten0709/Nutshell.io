"""
This is the Streamlit app for Nutshell.io, an intelligence engine for AI/ML engineers. 
It connects to a Qdrant vector database to fetch and display deduplicated and synthesized news updates from various sources. 
The app features a sidebar for controls and a main feed that highlights trending news based on mention counts.
"""

import streamlit as st
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
import asyncio
import os
import sys

# Add project root to path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

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

# ─────────────────────────────────────────────
# Qdrant Client
# ─────────────────────────────────────────────
@st.cache_resource
def get_qdrant_client():
    return QdrantClient(host="localhost", port=6333)

client = get_qdrant_client()

# ─────────────────────────────────────────────
# Data Fetching
# ─────────────────────────────────────────────
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
    try:
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
    except Exception:
        return []

# ─────────────────────────────────────────────
# Pipeline Step Functions
# ─────────────────────────────────────────────
def step1_check_qdrant():
    """Step 1: Check Qdrant connection and collection status."""
    try:
        from src.adapters.vector_store import VectorService
        vs = VectorService()
        info = vs.client.get_collection(vs.collection_name)
        return True, f"✅ Qdrant connected! Collection `{vs.collection_name}` has **{info.points_count} insights** stored."
    except Exception as e:
        return False, f"❌ Qdrant connection failed: {str(e)}\n\n💡 Run: `docker run -p 6333:6333 qdrant/qdrant`"

def step2_fetch_emails():
    """Step 2: Fetch latest newsletters from Nylas."""
    try:
        from src.adapters.mail import NylasAdapter
        from src.adapters.email_tracker import EmailTracker

        mail_adapter = NylasAdapter()
        tracker = EmailTracker()

        # Run async function in sync context
        newsletters = asyncio.run(mail_adapter.get_latest_newsletters())

        if not newsletters:
            return False, "📭 No new newsletters found in inbox."

        # Filter out already processed
        new_emails = [nl for nl in newsletters if not tracker.is_processed(nl['id'])]
        already_processed = len(newsletters) - len(new_emails)

        # Store in session state for Step 3
        st.session_state['pending_emails'] = new_emails

        return True, (
            f"📬 Fetched **{len(newsletters)}** total emails.\n\n"
            f"- 🆕 **{len(new_emails)}** new to process\n"
            f"- ⏭️ **{already_processed}** already processed (skipped)"
        )
    except Exception as e:
        return False, f"❌ Failed to fetch emails: {str(e)}"

def step3_process_and_store():
    """Step 3: Run LLM extraction + store in Qdrant."""
    try:
        from src.core.use_cases import process_new_email
        from src.adapters.email_tracker import EmailTracker

        tracker = EmailTracker()
        pending = st.session_state.get('pending_emails', [])

        if not pending:
            return False, "⚠️ No pending emails to process. Please run Step 2 first."

        success_count = 0
        failed_count = 0
        results = []

        for nl in pending:
            payload = {
                "TextBody": nl['body'],
                "From": nl['from'],
                "Subject": nl['subject'],
                "MessageID": nl['id'],
                "date": str(nl.get('date', ''))
            }
            try:
                asyncio.run(process_new_email(payload))
                tracker.mark_processed(nl['id'])
                success_count += 1
                results.append(f"  ✅ {nl['subject']}")
            except Exception as e:
                failed_count += 1
                results.append(f"  ❌ {nl['subject']}: {str(e)}")

        # Clear pending after processing
        st.session_state['pending_emails'] = []

        details = "\n".join(results)
        return True, (
            f"🎉 Processing complete!\n\n"
            f"- ✅ **{success_count}** emails processed successfully\n"
            f"- ❌ **{failed_count}** failed\n\n"
            f"**Details:**\n{details}"
        )
    except Exception as e:
        return False, f"❌ Processing failed: {str(e)}"

# ─────────────────────────────────────────────
# UI Layout
# ─────────────────────────────────────────────
st.title("🥜 Nutshell.io")
st.caption("Intelligence engine for AI/ML engineers. Deduplicated & Synthesized.")

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:

    # ── Pipeline Controls ──────────────────────
    st.header("⚙️ Pipeline Controls")
    st.caption("Run each step in order to ingest and process new newsletters.")

    # Step 1: Check Qdrant
    st.markdown("**Step 1: Check Qdrant**")
    if st.button("🔍 Check Qdrant Connection", use_container_width=True):
        with st.spinner("Checking Qdrant..."):
            ok, msg = step1_check_qdrant()
        if ok:
            st.success(msg)
        else:
            st.error(msg)

    st.divider()

    # Step 2: Fetch Emails
    st.markdown("**Step 2: Fetch New Emails**")
    if st.button("📬 Fetch Newsletters from Nylas", use_container_width=True):
        with st.spinner("Connecting to Nylas and fetching emails..."):
            ok, msg = step2_fetch_emails()
        if ok:
            st.success(msg)
            pending = st.session_state.get('pending_emails', [])
            if pending:
                st.info(f"👇 {len(pending)} emails ready. Run Step 3 to process them.")
        else:
            st.warning(msg)

    st.divider()

    # Step 3: Process & Store
    st.markdown("**Step 3: Extract & Store Insights**")
    pending_count = len(st.session_state.get('pending_emails', []))
    if pending_count > 0:
        st.caption(f"📌 {pending_count} emails pending processing")
    
    if st.button("🤖 Run LLM Extraction + Store", use_container_width=True, type="primary"):
        with st.spinner("Running LLM extraction and storing in Qdrant... (this may take a while)"):
            ok, msg = step3_process_and_store()
        if ok:
            st.success(msg)
            st.balloons()
            st.cache_data.clear()
        else:
            st.error(msg)

    st.divider()

    # ── Filters ───────────────────────────────
    st.header("🎛️ Filters")
    
    categories = ["All"] + get_unique_values("category")
    category_filter = st.selectbox("Category", categories, index=0)
    
    min_relevance = st.slider("Min Relevance Score", 1, 10, 1)
    
    all_tags = ["All"] + get_unique_values("tags")
    tag_filter = st.selectbox("Tag", all_tags, index=0)
    
    all_companies = ["All"] + get_unique_values("companies_mentioned")
    company_filter = st.selectbox("Company", all_companies, index=0)

    st.divider()

    sort_by = st.radio("Sort by", ["Trending (Mentions)", "Relevance Score", "Recent"])

    st.divider()

    if st.button("🔄 Refresh Feed", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.info("📡 Monitoring: TLDR, The Neuron, TAAFT, What's AI")

# ─────────────────────────────────────────────
# Main Feed
# ─────────────────────────────────────────────
nutshells = get_all_nutshells(
    category_filter=category_filter if category_filter != "All" else None,
    min_relevance=min_relevance if min_relevance > 1 else None,
    tag_filter=tag_filter if tag_filter != "All" else None,
    company_filter=company_filter if company_filter != "All" else None
)

if not nutshells:
    st.warning("⚠️ No intelligence matching your filters. Try adjusting filters or run the pipeline!")
    st.info("👈 Use the **Pipeline Controls** in the sidebar to fetch and process newsletters.")
else:
    if sort_by == "Trending (Mentions)":
        sorted_news = sorted(nutshells, key=lambda x: x.get('mention_count', 1), reverse=True)
    elif sort_by == "Relevance Score":
        sorted_news = sorted(nutshells, key=lambda x: x.get('relevance_score', 0), reverse=True)
    else:
        sorted_news = sorted(nutshells, key=lambda x: x.get('first_seen', ''), reverse=True)

    st.subheader(f"📰 {len(sorted_news)} Insights Found")

    for item in sorted_news:
        with st.container():
            col1, col2 = st.columns([3, 1])

            with col1:
                st.subheader(item.get('headline', 'Untitled Update'))
                st.write(item.get('summary', 'No summary available.'))

                # Tags
                item_tags = item.get('tags', [])
                if item_tags:
                    st.markdown("**Tags:**")
                    tags_html = " ".join([f"<span class='tag-pill'>{tag}</span>" for tag in item_tags])
                    st.markdown(tags_html, unsafe_allow_html=True)

                # Companies
                item_companies = item.get('companies_mentioned', [])
                if item_companies:
                    st.markdown("**Companies:**")
                    companies_html = " ".join([f"<span class='company-pill'>{comp}</span>" for comp in item_companies])
                    st.markdown(companies_html, unsafe_allow_html=True)

                # Key People
                key_people = item.get('key_people', [])
                if key_people:
                    st.markdown(f"**Key People:** {', '.join(key_people)}")

                # Sources/Links
                links = item.get('links', [])
                if links:
                    st.markdown("**Sources:**")
                    for i, link in enumerate(links, 1):
                        st.markdown(f"- [Source {i}]({link})")

                # First seen / Last seen
                first_seen = item.get('first_seen', '')
                last_seen = item.get('last_seen', '')
                if first_seen:
                    st.caption(f"🕐 First seen: `{first_seen}` | Last seen: `{last_seen}`")

            with col2:
                mentions = item.get('mention_count', 1)
                st.markdown(f"<span class='mention-badge'>🔥 {mentions} Mentions</span>", unsafe_allow_html=True)
                st.metric("Relevance", f"{item.get('relevance_score', 0)}/10")
                st.write(f"🏷️ `{item.get('category', 'General')}`")

                # Show all sources
                sources = item.get('sources', [])
                if sources:
                    st.caption("📬 Seen in:")
                    for s in sources:
                        if isinstance(s, dict):
                            st.caption(f"  - {s.get('subject', 'Unknown')}")

            st.divider()

    # Stats
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
            