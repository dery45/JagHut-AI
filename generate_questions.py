from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from services.vectorCSV import vector_store
import os

# Initialize the model
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:1.5b")
model = OllamaLLM(model=LLM_MODEL)

def generate_seed_questions():
    print("Fetching sample data from vector store...")
    # Get a sample of documents (e.g., 20 documents)
    # We can use get() with a limit
    docs_data = vector_store.get(limit=20)
    
    if not docs_data["documents"]:
        print("No documents found in the vector store.")
        return

    context = "\n---\n".join(docs_data["documents"][:15]) # Use first 15 for context

    template = """
    Below is a sample of data from an ESG (Environmental, Social, Governance) dataset about forestry, deforestation, and sustainability.
    Based on this data, generate a list of 10 diverse and informative questions that a user might ask a RAG system using this dataset.
    The questions should be in Indonesian and should be answerable using the data provided or similar data in the dataset.

    Sample Data:
    {context}

    List of 10 Questions:
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    print("Generating questions...")
    result = chain.invoke({"context": context})

    output_file = "generated_questions.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result)
    
    print(f"\nDone! Questions saved to {output_file}")
    print("\nGenerated Questions Preview:")
    print(result)

if __name__ == "__main__":
    generate_seed_questions()
