from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.docstore.document import Document

import chainlit as cl
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter 
import io
import PyPDF2
from io import BytesIO
from huggingface_hub import hf_hub_download


custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


custom_prompt_openai = """
<s>[INST] <<SYS>>
Use the context provided to answer the question at the end. If you dont know the answer just say that you don't know, don't try to make up an answer.
<</SYS>>

Context:
_______________________
{context}
_______________________

Question: {question} [/INST]

"""


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt



#This method can be called if a different LLM needs to be downloaded
def downloadmodel(model_id,model_filename,model_path_cache):
    model_path = hf_hub_download(             
                                 repo_id=model_id,             
                                 filename=model_filename,             
                                 resume_download=True,             
                                 cache_dir=model_path_cache,         
                                 )
    return model_path

#Loading the model
def load_llm():

    #uncomment to allow the UI to download a different model during first execution
    #mdpath = downloadmodel("TheBloke/Mistral-7B-Instruct-v0.1-GGUF","mistral-7b-instruct-v0.1.Q8_0.gguf","./models")

    # Load the locally downloaded model here
    llm = CTransformers(
        model = "models/llama-2-7b-chat.Q8_0.gguf",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.4,
        repetition_penalty = 1.2,
        top_k = 50
    )
    return llm

def hugging_embedding():
    
    # you can use a different embedding by replacing with model_name = sentence-transformers/all-MiniLM-L6-v2. system will download it during first execution
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5",
                                       model_kwargs={'device': 'cpu'})
    return embeddings


def retrieval_qa_chain(llm, prompt, db):     
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt}                                        
                                           )     
    return qa_chain


@cl.on_chat_start
async def on_chat_start():
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a pdf file to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=280,
        ).send()

    file = files[0]
    msg = cl.Message(
        content=f"Processing `{file.name}`...", disable_human_feedback=True
    )
    await msg.send()

    # Read the PDF file
    pdf_stream = BytesIO(file.content)
    pdf = PyPDF2.PdfReader(pdf_stream)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400,
                                                   chunk_overlap=80)
    # Split the text into chunks
    texts = text_splitter.split_text(pdf_text)

    # Create metadata for each chunk. 
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create a Chroma vector store
    embeddings = hugging_embedding()
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )

    chain = retrieval_qa_chain(load_llm(), set_custom_prompt(), docsearch)

    # Let the user know that the system is ready
    msg.content = f"Embedding for `{file.name}` is done. You can now ask questions! "
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  
    
    cb = cl.AsyncLangchainCallbackHandler(         stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]     )     
    cb.answer_reached = True

    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    source_documents = res["source_documents"] 

    text_elements = [] 

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()
