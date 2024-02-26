# !pip install PyPDF2
# !pip install langchain
# !pip install faiss-cpu
# !pip install openai
# !pip install python-dotenv
# !pip install sentence-transformers

# install PyTorch with CUDA support to enable GPU
# !pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu{CUDA_VERSION}/torch_stable.html
# where {CUDA_VERSION} can be obtained by the command in console: nvidia-smi
# For my case, {CUDA_VERSION} = 122 (referring to 12.2)

import requests
import torch
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback

# enable when run on cpu (to disable gpu)
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
# load the openai_api_key
load_dotenv()


def get_file(output_name:str='file.pdf', url:str=''):
    """
    :param output_name: name that will take the downloaded pdf file
    :param url: web url where located the pdf file
    :return: file as object
    """
    file = open(output_name, "wb")
    file.write(requests.get(url).content)
    file = open(output_name, 'rb')
    file = PdfReader(file)
    return file


def convert_file_into_text(file):
    text = ""
    for page in file.pages:
        text += page.extract_text()
    return text


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,  # overlap among chunks
    length_function=len
)

# information about pre-trained model names: https://huggingface.co/sentence-transformers
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#model_name = "sentence-transformers/all-MiniLM-L6-v2" # more robust
model_kwargs = {'device': 'cuda'}
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, multi_process=False, show_progress=False)
llm = ChatOpenAI(model_name='gpt-3.5-turbo')
chain = load_qa_chain(llm, chain_type="stuff")


def query(question:str, print_cost=False):

    file = get_file(output_name="file.pdf",
                    url="https://preguntapdf.s3.eu-south-2.amazonaws.com/BOE-A-1978-31229-consolidado.pdf")

    text = convert_file_into_text(file)
    chunks = text_splitter.split_text(text)
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    docs = knowledge_base.similarity_search(question, 3) # top 3 similarity
    answer = chain.invoke({"input_documents": docs, "question": question})
    if print_cost:
        get_openai_callback()
    return answer['output_text']


if __name__ == '__main__':
    answer = query(question="¿Qué asociaciones quedan expresamente prohibidas por el artículo 22?", print_cost=True)
    print(f'answer: {answer}')


