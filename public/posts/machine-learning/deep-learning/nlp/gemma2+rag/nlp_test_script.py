import contextlib
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


from huggingface_hub import login
from kaggle_secrets import UserSecretsClient

# Get Secret HuggingFace token
user_secrets = UserSecretsClient()
secret_value = user_secrets.get_secret("huggingface_api")

login(token=secret_value)

##
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

from IPython.display import Markdown, display


# from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import set_global_tokenizer

from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)

from llama_index.vector_stores.faiss import FaissVectorStore
import faiss


#  Load the PDF
documents = SimpleDirectoryReader('/kaggle/input/superconductivity-lectures/').load_data()

# Create embeddings
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
set_global_tokenizer(tokenizer)

llm_model = HuggingFaceLLM(model_name="google/gemma-2-9b-it",
                           tokenizer_name="google/gemma-2-9b-it", 
                           max_new_tokens=1000,
                           generate_kwargs={"temperature": 1, "num_return_sequences":1, "do_sample": False},
                           model_kwargs={"quantization_config": quantization_config},
                           device_map='auto')



llm_responses = []
queries = ["Which scientists contributed the most to superconductivity?",
          "Which are the differences between Type-I and Type-II superconductors? Describe magnetical properties and show formulas.",
          "What are the London Equation? Why are they important?",
          "Solve this problem: Consider a bulk superconductor containing a cylindrical hole of 0.1 mm diameter. There are 7 magnetic flux quanta trapped in the hole. Find the magnetic field in the hole."]


for query in queries[:]:
    print(query)
    llm_responses.append(
        llm_model.complete(query)
    )


for i, resp in enumerate(llm_responses):
#     tokenizer.decode(resp.raw['model_output'][0])
    display(Markdown("## " + queries[i] + "\n" + resp.text))


# vector store
d = 384  # embedding dimension
faiss_index = faiss.IndexFlatL2(d)
vector_store = FaissVectorStore(faiss_index=faiss_index)

len(embed_model.get_text_embedding("Hello!"))

vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model, show_progress=True
)

index.storage_context.persist()

query_engine = index.as_query_engine(llm=llm_model, similarity_top_k=5)

rag_responses = []
# query = "Which scientists contributed the most to superconductivity?"
for query in queries:
    response = query_engine.query(query)
    rag_responses.append(response)


# response.source_nodes[1].metadata['file_name'], 
# Markdown(response.response)
for i, resp in enumerate(rag_responses):
#     tokenizer.decode(resp.raw['model_output'][0])
    sources = []
    for node in resp.source_nodes:
        sources.append(node.metadata['file_name'])
#         print(f"{node.get_score()} 👉 {node.metadata['file_name']}")
    display(Markdown("## " + queries[i] + "\n" + f"Sources: _{sources}_\n" + resp.response))
