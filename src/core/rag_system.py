from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from .llm import LLM
from .chroma_db import ChromaDB
from .memory_handler import MemoryHandler
from .reranker import Reranker

class RAGSystem:
    
    def __init__(self, retriever: ChromaDB, memory_handler: MemoryHandler):
        self.llm = LLM(temperature=0.0).chat_model
        self.retriever = retriever
        self.reranker = Reranker(k_num=10)
        self.memory_handler = memory_handler
        self.relevant_documents_content = []
        self.reranked_relevant_documents_content = []

    def create_chat_prompt(
        self, 
        input_prompt: str, 
        relevant_documents: list[str],
        memory_messages: list[dict[str, str]] = []
        ) -> ChatPromptTemplate:
        
        context = '\n'.join(relevant_documents)
        
        conversation_history = '\n'.join(
            [
                f'\n{message['role']}: {message['content']}'
                for message in memory_messages
            ]
        )
        
        chat_prompt_template = ChatPromptTemplate.from_messages([
            ('user', """
                شما یک سامانه هوشمند پاسخ‌گو هستید که به پرسش‌های کاربران بر اساس اطلاعات موجود، زمینه گفتگو و گفت‌وگوهای قبلی پاسخ می‌دهید. وظیفه شما ارائه پاسخ‌هایی دقیق، مرتبط و قابل فهم است.
                برای تولید پاسخ مناسب، مراحل زیر را رعایت کنید:
                ۱. ابتدا پرسش کاربر را به‌دقت بررسی و مقصود او را درک کنید.
                ۲. سپس اطلاعات مرتبطی که از منابع گوناگون به دست آمده‌اند را تحلیل نمایید.
                ۳. اگر تاریخچه گفتگو میان شما و کاربر وجود دارد، آن را نیز در نظر بگیرید تا پاسخ دقیق‌تری بدهید.
                ۴. با ترکیب پرسش، اطلاعات زمینه‌ای و تاریخچه مکالمه، پاسخی روشن، خلاصه و مرتبط تولید کنید.
                نکات مهم:
                اگر اطلاعات کافی برای پاسخ وجود ندارد، از حدس زدن خودداری کرده و محترمانه اعلام کنید که پاسخ در دسترس نیست.
                از تکرار مستقیم متن منابع خودداری نمایید؛ اطلاعات را به زبان روان و ساده بازنویسی کنید.
                لحن شما باید محترمانه، شفاف و رسمی باشد.
                پاسخ‌ها باید کوتاه، مفید، و دقیق باشند.
                
                تاریخچه گفتگو:
                {conversation_history}
                
                اطلاعات مرتبط:
                {context}
                
                سوال کاربر:
                {question}
                
                **پاسخ**
            """)
        ])
        
        return chat_prompt_template.invoke({
            'conversation_history': conversation_history,
            'context': context,
            'question': input_prompt
            })

        
    def generate_response(self, prompt: str):
        try:
            
            relevant_documents = self.retriever.query(prompt)
            self.relevant_documents_content = [doc.page_content for doc in relevant_documents]
            #self.reranked_relevant_documents_content = self.reranker.rerank(prompt, self.relevant_documents_content)
            prompt_template = self.create_chat_prompt(
                prompt, 
                self.relevant_documents_content, 
                self.memory_handler.messages
            )
            response = self.llm.invoke(prompt_template)
            return response.content
        
        except Exception as e:
            print(f'Error generating a response {str(e)}')
            raise e  