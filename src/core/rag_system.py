from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from .llm import LLM

class RAGSystem:
    
    def __init__(self):
        self.llm = LLM(temperature=0.0).chat_model

    def create_chat_prompt(
        self, 
        input_prompt: str, 
        relevant_documents: list[Document],
        memory_messages: list[dict[str, str]] = []
        ) -> ChatPromptTemplate:
        
        related_texts = [doc.page_content for doc in relevant_documents]
        context = ''.join(related_texts)
        
        conversation_history = ''.join(
            [
                f'\n{message['role']}: {message['content']}'
                for message in memory_messages
            ]
        )
        
        chat_prompt_template = ChatPromptTemplate.from_messages([
            ('user', """
                Follow these guidelines:
                If the question is related to the given context (retrieved from the knowledge base), respond strictly based on the provided context.
                If the question is general or not found in the context, answer using your general knowledge.
                Always consider the conversation history (previous chat interactions) to maintain coherence and context-awareness.
                If the question is unclear or insufficient, politely ask for clarification before answering.
                Keep responses clear, helpful, and friendly.

                Context for this query:
                {context}

                Conversation history:
                {conversation_history}

                User's question:
                {question}
                
                **Answer: **
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