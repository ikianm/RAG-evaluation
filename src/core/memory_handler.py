from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .llm import LLM

class MemoryHandler:
    def __init__(self):
        self.messages = [
            {'role': 'system', 'content': 'You are an intelligent assistant that answers user questions based on the provided context.'}
        ]
        self.llm = LLM(temperature=0.0).chat_model
        
    def append_messages(self, new_messages: list[dict[str, str]]): 
        self.messages += new_messages
        
    def summarize_messages(self):
        summarization_prompt = PromptTemplate.from_template("""
            **وظیفه خلاصه‌سازی گفتگو**  
            شما یک دستیار هوشمند هستید که وظیفه خلاصه‌کردن گفتگوی بین کاربر و دستیار هوشمند را بر عهده دارید.  
            خلاصه‌ای مختصر اما آموزنده ایجاد کنید که موارد زیر را پوشش دهد:  

            1. **موضوعات کلیدی** مطرح‌شده  
            2. **سوالات مهم** کاربر  
            3. **اطلاعات یا پاسخ‌های کلیدی** ارائه‌شده توسط دستیار  
            4. **تصمیمات یا نتیجه‌گیری‌های** گرفته‌شده (در صورت وجود)  
            5. **سوالات بی‌پاسخ یا موضوعات حل‌نشده** (اگر وجود دارد)  

            **راهنما:**  
            - از دیدگاه **بیطرف و سوم‌شخص** بنویسید.  
            - جزئیات ضروری برای **حفظ زمینه آینده** را حفظ کنید.  
            - از **زبان ساده و واضح** استفاده کنید.  
            - **گفتگوهای غیرمرتبط** (مثل احوالپرسی) را نادیده بگیرید، مگر اینکه مهم باشند.  

            **تاریخچه گفتگو:**  
            {conversation_history}  

            **خلاصه:**  
            """)
                    
        concatinated_messages = ''.join(
            [
                f'\n{message['role']}: {message['content']}.'
                for message in self.messages[-6:]
            ]
        )
        
        summarization_chain = summarization_prompt | self.llm | StrOutputParser()
        summary = summarization_chain.invoke({'conversation_history': concatinated_messages})
        
        self.messages = self.messages[:-6]
        self.messages.append({'role': 'assistant', 'content': f'Summary: {summary}'})