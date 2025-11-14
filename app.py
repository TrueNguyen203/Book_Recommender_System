import pandas as pd
import numpy as np

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

import streamlit as st

# Initialize our model and database
books = pd.read_csv('books_with_emotions.csv')
books["large_thumbnail"] = books["thumbnail"] + "&file=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover_not_found.png",
    books["large_thumbnail"]
)


embeddings = OllamaEmbeddings(
    model="llama3.2",
)
persist_directory = './chroma_books_db'
collection_name = 'books_collection'


db_books = Chroma(persist_directory=persist_directory,
                  embedding_function=embeddings,
                  collection_name=collection_name)



#Retrie top_k sematic search by llama3.2
def retrieve_sematic_search(
        query: str,
        category: str = None,   # Book category filter
        tone: str = None,   # Emotional tone filter
        initial_top_k: int = 60, # The number of books to retrieve from the database before filtering
        final_top_k: int = 20 # The final number of books to return after filtering
) -> pd.DataFrame:
    

    recs = db_books.similarity_search_with_score(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec, score in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    # Category filtering
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category][:final_top_k]
    else:
        book_recs = book_recs.head(final_top_k)
    
    # Tone filtering
    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="angry", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books():
    st.title("Book Recommendation System 📚")
    st.write("Find books that match your mood and interests!")

    # User inputs
    user_query = st.text_input("Enter a brief description of the type of book you're looking for:")

    category_options = ["All"] + sorted(books["simple_categories"].dropna().unique().tolist())
    selected_category = st.selectbox("Select Book Category:", category_options)

    tone_options = ["None", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
    selected_tone = st.selectbox("Select Emotional Tone:", tone_options)

    if st.button("Recommend Books"):
        if user_query.strip() == "":
            st.warning("Please enter a book description to get recommendations.")
        else:
            recommendations = retrieve_sematic_search(
                query=user_query,
                category=selected_category,
                tone=selected_tone if selected_tone != "None" else None
            )

            if recommendations.empty:
                st.info("No books found matching your criteria. Please try different inputs.")
            else:
                st.subheader("Recommended Books:")
                
                for _, row in recommendations.iterrows():
                    with st.container():
                        col1, col2 = st.columns([2, 3])
                        with col1:
                            st.image(row['large_thumbnail'], width=250)
                        with col2:
                            st.markdown(f"### {row['title']} by {row['authors']}")
                            st.markdown(f"**Category:** {row['simple_categories']}")
                            st.markdown(f"**Description:** {row['description']}")
                        st.markdown("---")

if __name__ == "__main__":
    recommend_books()



