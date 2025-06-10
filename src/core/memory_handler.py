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
            خلاصه‌سازی این گفت‌وگو را به‌صورت دقیق، بی‌طرفانه و در زبان سوم انجام بده. موارد زیر را در خلاصه بگنجان:
            موضوعات و پرسش‌های اصلی مطرح‌شده
            پاسخ‌ها و راهکارهای کلیدی ارائه‌شده
            تصمیم‌گیری‌ها یا مسائل باز و ناتمام
            از پرداختن به گفتگوهای غیرمهم یا احوال‌پرسی‌ها صرف‌نظر کن. لحن خلاصه باید رسمی، شفاف و دقیق باشد.
            تاریخچه گفت‌وگو:
            {conversation_history}
            **خلاصه**:
            """ )
                    
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