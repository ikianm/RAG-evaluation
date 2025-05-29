from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ContextPrecision, LLMContextRecall, NoiseSensitivity, ResponseRelevancy, Faithfulness, FactualCorrectness
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ..core.rag_system import RAGSystem
from ..core.chroma_db import ChromaDB

import logging
from dotenv import load_dotenv
import os

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f'**Evaluating {os.getenv('LLM_MODEL')}**')

sample_queries = [
    'آیا مدیران و نمایندگان ثبت و مسئولین دفاتر و صاحبان دفاتر اسناد رسمی می‌توانند خارج از محل ماموریت خود انجام وظیفه کنند؟',
    'بر اساس ماده ۷۰، اعتبار سند رسمی ثبتشده چیست و آیا انکار مندرجات آن پذیرفته میشود؟',
    "نسبت به املاکی که مجهول‌المالک اعلان شده‌اند، چه مدت زمان برای تقاضای ثبت وجود دارد؟",
    "در صورت عدم حضور متقاضی و مجاورین در جلسه تحدید حدود، چه اتفاقی می‌افتد؟",
    'دولت چه کسی را مالک می‌شناسد وقتی ملکی مطابق قانون در دفتر املاک به ثبت رسیده است؟',
    'پس از پایان مهلت اعتراض، چه دعاوی پذیرفته نخواهد شد؟',
    'قانون اساسی جمهوری اسلامی ایران در چند فصل تنظیم گردیده است؟'
]

expected_response = [
    'مديران و نمايندگان ثبت و مسئولين دفاتر و صاحبان دفاتر اسناد رسمي جز در محل ماموريت خود نمي ‌توانند انجام وظيفه نمايند و اقدامات ‌آن ها در خارج از آن محل اثر قانوني ندارد',
    'سندی که مطابق قوانین ثبت شده باشد، رسمی است و تمام محتوای آن معتبر خواهد است، مگر اینکه مجعولیت آن ثابت شود. انکار مندرجات اسناد رسمی راجع به اخذ تمام یا قسمتی از وجه یا مال یا تعهد به تأدیه وجه یا تسلیم مال، مسموع نیست',
    "نسبت باملاكي كه مجهول‌المالك اعلان شده اشخاصيكه حق تقاضاي ثبت دارند ميتوانند در ظرف مدت دو سال از تاريخ اجراي اين‌ قانون تقاضاي ثبت نمايند",
    'اگر براي مرتبه دوم نيز تقاضاكننده و مجاورين هيچيك حاضر نشده و تحديد حدود بعمل نيايد حق ‌الثبت ملك دو برابر اخذ خواهد شد',
    'دولت فقط كسي را كه ملك باسم او ثبت شده و يا كسي را كه ملك مزبور باو‌ منتقل گرديده و اين انتقال نيز در دفتر املاك به ثبت رسيده يا اينكه ملك مزبور از مالك رسمي ارثا باو رسيده باشد مالك خواهد شناخت',
    'پس از انقضاي مدت اعتراض دعوي اينكه در ضمن جريان ثبت تضييع حقي از كسي شده پذيرفته نخواهد شد نه بعنوان عين نه بعنوان قيمت نه بهيچ عنوان ديگر خواه حقوقي باشد خواه جزائي',
    'قانون اساسی جمهوری اسلامی ایران در دوازده فصل تنظیم گردیده'
]

chroma_db = ChromaDB()
rag_system = RAGSystem()

dataset = []

for query, reference in zip(sample_queries, expected_response):
    logging.info('Generating response for sample queries...')
    related_documents = chroma_db.get_relevant_documents(query)
    retrieved_texts = [doc.page_content for doc in related_documents]
    chat_prompt_template = rag_system.create_chat_prompt(query, related_documents)
    response = rag_system.generate_response(chat_prompt_template)
    dataset.append({
        'user_input': query,
        'retrieved_contexts': retrieved_texts,
        'response': response,
        'reference': reference
    })
    
evaluation_dataset = EvaluationDataset.from_list(dataset)

logger.info('Dataset created!')

embedding = HuggingFaceEmbeddings( 
    model_name=os.getenv('EMBEDDING_MODEL'),
    model_kwargs={'trust_remote_code': True}
    )

llm = ChatOpenAI(
    model='openai/gpt-3.5-turbo',
    base_url=os.getenv('LLM_BASE_URL'),
    api_key=os.getenv('LLM_API_KEY')
)

evaluator_llm = LangchainLLMWrapper(llm)

logger.info('Evaluation started...')
result = evaluate(
    dataset=evaluation_dataset, 
    llm=evaluator_llm, 
    embeddings=embedding, # needed for some metrics like Context Precision
    metrics=[
        ContextPrecision(),
        LLMContextRecall(), 
        NoiseSensitivity(),
        ResponseRelevancy(),
        Faithfulness(), 
        FactualCorrectness()
    ]
)
print(result)