
# pip install ollama -q

# pip install langchain -q

# pip install langchain-community -q

import ollama

from langchain_community.llms import Ollama

llm = Ollama(model="gemma:2b", temperature=0.1)

respuesta = llm.invoke("Necesito que me des una codigo de python para hacer una funcion que calcule el factorial de un numero")

print(respuesta);


# Ejemplo con PDFS.
# Hay que instalar la libreria de pdfs, luego con langchain hacemos el spliting de los datos.
# !pip install pypdf -q

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import ollama


loader = PyPDFLoader("alcance.pdf")
paginas = loader.load()


len(paginas)
print(paginas[3].page_content)


text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 500,
    chun
)

