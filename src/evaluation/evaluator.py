from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

from ..core.rag_system import RAGSystem
from ..core.chroma_db import ChromaDB

from dotenv import load_dotenv
import os

load_dotenv()

sample_queries = [
    'یا مدیران و نمایندگان ثبت و مسئولین دفاتر و صاحبان دفاتر اسناد رسمی می‌توانند خارج از محل ماموریت خود انجام وظیفه کنند؟',
    'بر اساس ماده ۷۰، اعتبار سند رسمی ثبتشده چیست و آیا انکار مندرجات آن پذیرفته میشود؟',
    "نسبت به املاکی که مجهول‌المالک اعلان شده‌اند، چه مدت زمان برای تقاضای ثبت وجود دارد؟",
    "در صورت عدم حضور متقاضی و مجاورین در جلسه تحدید حدود، چه اتفاقی می‌افتد؟",
    'دولت چه کسی را مالک می‌شناسد وقتی ملکی مطابق قانون در دفتر املاک به ثبت رسیده است؟',
    'پس از پایان مهلت اعتراض، چه دعاوی پذیرفته نخواهد شد؟'
]

expected_response = [
    '.مديران و نمايندگان ثبت و مسئولين دفاتر و صاحبان دفاتر اسناد رسمي جز در محل ماموريت خود نمي ‌توانند انجام وظيفه نمايند و اقدامات ‌آن ها در خارج از آن محل اثر قانوني ندارد',
    '.سندی که مطابق قوانین ثبت شده باشد، رسمی است و تمام محتوای آن معتبر خواهد است، مگر اینکه مجعولیت آن ثابت شود. انکار مندرجات اسناد رسمی راجع به اخذ تمام یا قسمتی از وجه یا مال یا تعهد به تأدیه وجه یا تسلیم مال، مسموع نیست',
    "نسبت باملاكي كه مجهول‌المالك اعلان شده اشخاصيكه حق تقاضاي ثبت دارند ميتوانند در ظرف مدت دو سال از تاريخ اجراي اين‌ قانون تقاضاي ثبت نمايند",
    'اگر براي مرتبه دوم نيز تقاضاكننده و مجاورين هيچيك حاضر نشده و تحديد حدود بعمل نيايد حق ‌الثبت ملك دو برابر اخذ خواهد شد',
    'دولت فقط كسي را كه ملك باسم او ثبت شده و يا كسي را كه ملك مزبور باو‌ منتقل گرديده و اين انتقال نيز در دفتر املاك به ثبت رسيده يا اينكه ملك مزبور از مالك رسمي ارثا باو رسيده باشد مالك خواهد شناخت',
    'پس از انقضاي مدت اعتراض دعوي اينكه در ضمن جريان ثبت تضييع حقي از كسي شده پذيرفته نخواهد شد نه بعنوان عين نه بعنوان قيمت نه بهيچ عنوان ديگر خواه حقوقي باشد خواه جزائي'
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
    
evaluation_dataset = EvaluationDataset.from_list(dataset)

llm = ChatOpenAI(
            api_key='ollama',
            base_url=os.getenv('OLLAMA_BASE_URL'),
            model='gemma3:4b'
            )
evaluator_llm = LangchainLLMWrapper(llm)

result = evaluate(dataset=evaluation_dataset, llm=evaluator_llm, metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()])
print(result)