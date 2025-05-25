from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from main import RAGSystem, ChromaDB

from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
import os

load_dotenv()

sample_queries = [
    'Where is Ikianm.CO headquartered?',
    'What is the website of Ikianm.CO?',
    "What were the primary regions contributing to Ikianm.CO's revenue growth in 2024, and what were their respective growth percentages?"
]

expected_response = [
    'Ikianm.CO is headquartered in Oslo, Norway.',
    'Website of Ikianm.CO is www.ikianmco.com',
    "The primary regions contributing to Ikianm.CO's revenue growth in 2024 were: North America: +34%, Europe: +18%, Asia: +22%"
]

chroma_db = ChromaDB()
rag_system = RAGSystem()

dataset = []

for query, reference in zip(sample_queries, expected_response):
    related_documents = chroma_db.get_related_documents(query)
    retrieved_texts = [doc.page_content for doc in related_documents]
    chat_prompt_template = rag_system.create_chat_prompt(query, related_documents)
    response = rag_system.generate_response(chat_prompt_template)
    dataset.append({
        'user_input': query,
        'retrieved_contexts': retrieved_texts,
        'response': response,
        'reference': reference
    })
    
for data in dataset:
    print(f'\ninput: {data['user_input']}')
    print(f'\nretrieved_contexts: {data['retrieved_contexts']}')
    print(f'\nresponse: {data['response']}')
    print(f'\nreference: {data['reference']}')
    
evaluation_dataset = EvaluationDataset.from_list(dataset)

llm = ChatOpenAI(
            api_key=os.getenv('OPENROUTER_API_KEY'),
            base_url=os.getenv('OPENROUTER_BASE_URL'),
            model='openai/gpt-3.5-turbo'
            )
evaluator_llm = LangchainLLMWrapper(llm)

result = evaluate(dataset=evaluation_dataset, llm=evaluator_llm, metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()])
print(result)