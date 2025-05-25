from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from .llm import LLM

class RAGSystem:
    
    def __init__(self):
        self.llm = LLM(temperature=0.3).chat_model
        
    def create_chat_prompt(
        self, 
        input_prompt: str, 
        related_documents: list[Document],
        memory: list[str] = []
        ) -> ChatPromptTemplate:
        
        related_texts = [doc.page_content for doc in related_documents]
        context = ''.join(related_texts)
        conversation_history = ''.join(memory)
        chat_prompt_template = ChatPromptTemplate.from_messages([
            ('system', 'شما یک دستیار هوشمند و مفید هستید.'),
            ('user', """
                شما یک دستیار هوشمند هستید. با استفاده از اطلاعات ارائه‌شده در «متن مرتبط» و «تاریخچه گفتگو»، به سوال کاربر پاسخ دهید. اگر پاسخ در این منابع یافت نشد، از دانش خود برای ارائه پاسخ دقیق و مختصر استفاده کنید. لحن پاسخ باید حرفه‌ای و بیطرف باشد. از توضیحات غیرضروری یا زبان غیررسمی خودداری کنید. به خودتان یا منبع دانشتان اشاره نکنید.

                تاریخچه گفتگو:
                {conversation_history}

                متن مرتبط:
                {context}

                سوال کاربر:
                {question}

                **پاسخ:**
            """)
        ])
        
        return chat_prompt_template.invoke({
            'conversation_history': conversation_history,
            'context': context,
            'question': input_prompt
            })
    
    def generate_response(self, chat_prompt_template: ChatPromptTemplate):
        if not self.llm:
            raise ValueError('No initialized LLM found')
        try:
            response = self.llm.invoke(chat_prompt_template)
            return response.content
        except Exception as e:
            print(f'Error generating a response {str(e)}')
            raise e  