from transformers import AutoModelForSequenceClassification   

class Reranker:
    
    def __init__(self, k_num: int = 10):
        self.k_num = k_num
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'jinaai/jina-reranker-v2-base-multilingual',
            torch_dtype='auto',
            trust_remote_code=True,
            use_flash_attn=False,
            max_length=512
        )
        self.model.to('cpu')
        self.model.eval()

        
    def rerank(self, query: str, relevant_documents_content: list[str]) -> list[str]:
        rankings = self.model.rerank(query, relevant_documents_content, return_documents=True)
        return [ranking['document'] for ranking in rankings[:self.k_num]] 