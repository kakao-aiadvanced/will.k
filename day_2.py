import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = {}

splits = []
for paths in urls:
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/"+paths,),
        session=session,  # 여기에 session 전달
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs[paths] = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n\n\n"],chunk_size=2000, chunk_overlap=200, length_function=len,
                                                   is_separator_regex=False)
    doc_splits = text_splitter.split_documents(docs[paths])
    splits.extend(doc_splits)

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

def rag(question: str) -> str:
  retriever = vectorstore.as_retriever()
  prompt = hub.pull("rlm/rag-prompt")

  def format_docs(docs):
      return "\n\n".join(doc.page_content for doc in docs)

  rag_chain = (
      {"context": lambda _: formatted_context, "question": RunnablePassthrough()}
      | prompt
      | llm
      | StrOutputParser()
  )

  def evaluate_retrieval(question_to_evaluate, document_to_evaluate) -> bool:
    query = f"Evaluate the relevance of RAG retrieval. You will be provided with question and document. Reply with either 'yes' or 'no'. question: {question_to_evaluate}, document: {document_to_evaluate}"

    parser = JsonOutputParser()

    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    evaluate_chain = prompt | llm | parser

    evaulate_result = evaluate_chain.invoke({"query": query})

    evaluate_response = next(iter(evaulate_result.values()))
    print(evaluate_response)

    return evaluate_response == "yes"

  def detect_hallucination(question_to_evaluate, document_to_evaluate, response_to_evaluate) -> bool:
    query = f"Detect hallucination of RAG result. You will be provided with question, document and response. Reply with either 'yes' or 'no'. question: {question_to_evaluate}, document: {document_to_evaluate}, response: {response_to_evaluate}"

    parser = JsonOutputParser()

    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    evaluate_chain = prompt | llm | parser

    evaulate_result = evaluate_chain.invoke({"query": query})

    evaluate_response = next(iter(evaulate_result.values()))
    print(evaluate_response)

    return evaluate_response == "yes"

  # Retrieval
  retrieved_document = retriever.invoke(question)
  print(f"retrieved_document: {retrieved_document}")

  # Evaluate relevance of retrieval
  evaluate_result = evaluate_retrieval(question, retrieved_document)
  print(f"evaluate_result: {evaluate_result}")

  for _ in range(2):
    # Chain
    format_docs(retrieved_document)
    chain = (
      {"context": retriever | format_docs, "question": RunnablePassthrough()}
      | prompt
      | llm
      | StrOutputParser()
    )
    chain_response = chain.invoke(question)
    print(f"chain_response: {chain_response}")

    # Detect Hallucination
    is_hallucination = detect_hallucination(question, retrieved_document, chain_response)
    print(f"is_hallucination: {is_hallucination}")
    if is_hallucination:
      continue
    else:
      break
  return chain_response

rag("MRKL")