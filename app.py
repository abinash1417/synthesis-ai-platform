import streamlit as st
from document_processor import KnowledgeBase
from tools import get_knowledge_base
from agents import ResearchPipeline
from config import Config

st.set_page_config(page_title=Config.PAGE_TITLE, layout="wide", page_icon="🔬")


def load_custom_css():
    with open("static/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline() -> ResearchPipeline:
    return ResearchPipeline()


def get_kb() -> KnowledgeBase:
    return get_knowledge_base()


def render_sidebar(kb: KnowledgeBase):
    with st.sidebar:
        st.markdown("### 📄 Knowledge Base")
        uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

        if uploaded_files and st.button("➕ Add to Knowledge Base"):
            with st.spinner("Processing documents..."):
                for file in uploaded_files:
                    text = kb.extract_pdf_text(file)
                    num_chunks = kb.add_document(file.name, text)
                    st.success(f"Added {file.name} ({num_chunks} chunks)")

        st.metric("Total chunks stored", kb.chunk_count())

        if kb.chunk_count() > 0:
            st.markdown("**📁 Documents:**")
            for name in kb.list_document_names():
                st.markdown(f"- {name}")

        st.markdown("---")
        if st.button("🗑️ Clear Knowledge Base"):
            kb.clear()
            st.rerun()


def render_main(kb: KnowledgeBase, pipeline: ResearchPipeline):
    st.title(Config.APP_TITLE)
    st.caption(Config.APP_CAPTION)

    st.markdown("---")
    st.markdown("### Ask a Research Question")
    question = st.text_input(
        "Your question",
        placeholder="e.g., How strong is my resume? or Compare findings across documents",
        label_visibility="collapsed"
    )

    if st.button("🔍 Research", type="primary") and question:
        if kb.chunk_count() == 0:
            st.warning("Please upload documents to the knowledge base first.")
            return

        with st.spinner("Agents are researching, writing, and reviewing..."):
            try:
                result = pipeline.run(question)

                st.markdown("---")
                st.markdown("### 📋 Answer")
                st.markdown(result["final"])

                with st.expander("🔍 See research findings"):
                    st.write(result["research"])

                st.caption(f"Reviewed after {result['attempts']} attempt(s)")

            except Exception as e:
                st.error(f"Something went wrong after multiple retries: {e}")


def main():
    load_custom_css()
    kb = get_kb()
    pipeline = get_pipeline()

    render_sidebar(kb)
    render_main(kb, pipeline)


if __name__ == "__main__":
    main()