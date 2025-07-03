
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_community.tools import DuckDuckGoSearchRun
# from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
# from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import RetrievalQA
# from langchain.agents import AgentExecutor, Tool, initialize_agent, AgentType
# from langchain.schema import Document
# import os
# from dotenv import load_dotenv
# import logging
# from typing import List, Dict, Any, Optional

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# load_dotenv()
# GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# class AgenticRAGSystem:
#     def __init__(self):
#         self.llm = ChatGoogleGenerativeAI(
#             model="gemini-1.5-flash",
#             google_api_key=GOOGLE_API_KEY,
#             temperature=0.1
#         )
        
#         self.embeddings = GoogleGenerativeAIEmbeddings(
#             model="models/embedding-001",
#             google_api_key=GOOGLE_API_KEY
#         )
        
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200
#         )
        
#         self.vectorstore = None
#         self.retriever = None
#         self.qa_chain = None
#         self.agent = None
#         self.search_tool = DuckDuckGoSearchRun()
        
#         self.initialize_tools()
    
#     def initialize_tools(self):
#         tools = [
#             Tool(
#                 name="pdf_retrieval",
#                 func=self.query_pdf_knowledge,
#                 description="Useful for answering questions from PDF documents"
#             ),
#             Tool(
#                 name="web_search",
#                 func=self.search_web,
#                 description="Useful for answering questions that require current information"
#             )
#         ]
        
#         self.agent = initialize_agent(
#             tools=tools,
#             llm=self.llm,
#             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#             verbose=True
#         )
    
#     def load_pdf_documents(self, pdf_paths: List[str]) -> None:
#         documents = []
#         for pdf_path in pdf_paths:
#             try:
#                 loader = PyPDFLoader(pdf_path)
#                 docs = loader.load()
#                 documents.extend(docs)
#                 logger.info(f"Loaded PDF: {pdf_path}")
#             except Exception as e:
#                 logger.error(f"Error loading PDF {pdf_path}: {e}")
        
#         if documents:
#             splits = self.text_splitter.split_documents(documents)
#             self.vectorstore = FAISS.from_documents(splits, self.embeddings)
#             self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
#             self.qa_chain = RetrievalQA.from_chain_type(
#                 llm=self.llm,
#                 chain_type="stuff",
#                 retriever=self.retriever,
#                 return_source_documents=True
#             )
#             logger.info(f"Loaded {len(documents)} PDF documents")
    
#     def query_pdf_knowledge(self, query: str) -> str:
#         if not self.qa_chain:
#             return "No PDF documents loaded yet"
        
#         try:
#             result = self.qa_chain({"query": query})
#             return result["result"]
#         except Exception as e:
#             logger.error(f"Error querying documents: {e}")
#             return f"Error querying documents: {e}"
    
#     def search_web(self, query: str) -> str:
#         try:
#             results = self.search_tool.run(query)
#             return results
#         except Exception as e:
#             logger.error(f"Web search error: {e}")
#             return f"Error searching web: {e}"
    
#     def query(self, user_query: str) -> Dict[str, Any]:
#         try:
#             result = self.agent.run(user_query)
#             return {
#                 "answer": result,
#                 "sources": "Generated from agent tools"
#             }
#         except Exception as e:
#             logger.error(f"Error processing query: {e}")
#             return {
#                 "answer": f"Sorry, I encountered an error: {e}",
#                 "sources": []
#             }

# def main() -> None:
#     rag_system = AgenticRAGSystem()
#     pdf_files = ["./chapter3.pdf"]  
    
#     # Load PDF documents
#     rag_system.load_pdf_documents(pdf_files)

#     while True:
#         user_query = input("\nEnter your query (or 'quit' to exit): ")
#         if user_query.lower() == 'quit':
#             break
        
#         result = rag_system.query(user_query)
#         print(f"\nAnswer:\n{result['answer']}")
        
#         if result["sources"]:
#             print(f"\nSources: {result['sources']}")

# if __name__ == "__main__":
#     main()


# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_community.tools import DuckDuckGoSearchRun
# from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
# from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import RetrievalQA
# from langchain.agents import AgentExecutor, Tool, initialize_agent, AgentType
# from langchain.schema import Document
# import os
# from dotenv import load_dotenv
# import logging
# from typing import List, Dict, Any, Optional

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# load_dotenv()
# GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# class AgenticRAGSystem:
#     def __init__(self):
#         self.llm = ChatGoogleGenerativeAI(
#             model="gemini-1.5-flash",
#             google_api_key=GOOGLE_API_KEY,
#             temperature=0.1
#         )
        
#         self.embeddings = GoogleGenerativeAIEmbeddings(
#             model="models/embedding-001",
#             google_api_key=GOOGLE_API_KEY
#         )
        
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200
#         )
        
#         # Separate vector stores for PDF and web content
#         self.pdf_vectorstore = None
#         self.web_vectorstore = None
        
#         self.pdf_retriever = None
#         self.web_retriever = None
        
#         self.pdf_qa_chain = None
#         self.web_qa_chain = None
        
#         self.agent = None
#         self.search_tool = DuckDuckGoSearchRun()
        
#         self.initialize_tools()
    
#     def initialize_tools(self):
#         tools = [
#             Tool(
#                 name="pdf_knowledge",
#                 func=self.query_pdf_knowledge,
#                 description="Useful for answering questions from loaded PDF documents"
#             ),
#             Tool(
#                 name="web_search",
#                 func=self.search_web,
#                 description="Useful for answering questions that require current web information"
#             )
#         ]
        
#         self.agent = initialize_agent(
#             tools=tools,
#             llm=self.llm,
#             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#             verbose=True
#         )
    
#     def load_pdf_documents(self, pdf_paths: List[str]) -> None:
#         documents = []
#         for pdf_path in pdf_paths:
#             try:
#                 loader = PyPDFLoader(pdf_path)
#                 docs = loader.load()
#                 documents.extend(docs)
#                 logger.info(f"Loaded PDF: {pdf_path}")
#             except Exception as e:
#                 logger.error(f"Error loading PDF {pdf_path}: {e}")
        
#         if documents:
#             splits = self.text_splitter.split_documents(documents)
#             self.pdf_vectorstore = FAISS.from_documents(splits, self.embeddings)
#             self.pdf_retriever = self.pdf_vectorstore.as_retriever(search_kwargs={"k": 3})
#             self.pdf_qa_chain = RetrievalQA.from_chain_type(
#                 llm=self.llm,
#                 chain_type="stuff",
#                 retriever=self.pdf_retriever,
#                 return_source_documents=True
#             )
#             logger.info(f"Loaded {len(documents)} PDF documents into PDF knowledge base")
    
#     def load_web_content(self, urls: List[str]) -> None:
#         documents = []
#         for url in urls:
#             try:
#                 loader = WebBaseLoader(url)
#                 docs = loader.load()
#                 documents.extend(docs)
#                 logger.info(f"Loaded web content: {url}")
#             except Exception as e:
#                 logger.error(f"Error loading web content {url}: {e}")
        
#         if documents:
#             splits = self.text_splitter.split_documents(documents)
#             self.web_vectorstore = FAISS.from_documents(splits, self.embeddings)
#             self.web_retriever = self.web_vectorstore.as_retriever(search_kwargs={"k": 3})
#             self.web_qa_chain = RetrievalQA.from_chain_type(
#                 llm=self.llm,
#                 chain_type="stuff",
#                 retriever=self.web_retriever,
#                 return_source_documents=True
#             )
#             logger.info(f"Loaded {len(documents)} web documents into web knowledge base")
    
#     def query_pdf_knowledge(self, query: str) -> str:
#         if not self.pdf_qa_chain:
#             return "No PDF documents loaded yet"
        
#         try:
#             result = self.pdf_qa_chain({"query": query})
#             return result["result"]
#         except Exception as e:
#             logger.error(f"Error querying PDF documents: {e}")
#             return f"Error querying PDF documents: {e}"
    
    
#         if not self.web_qa_chain:
#             return "No web content loaded yet"
        
#         try:
#             result = self.web_qa_chain({"query": query})
#             return result["result"]
#         except Exception as e:
#             logger.error(f"Error querying web documents: {e}")
#             return f"Error querying web documents: {e}"
    
#     def search_web(self, query: str) -> str:
#         try:
#             results = self.search_tool.run(query)
            
#             # Optionally: automatically add search results to web knowledge base
#             # search_doc = Document(
#             #     page_content=results,
#             #     metadata={"source": "search_results", "query": query}
#             # )
#             # self.add_to_web_knowledge([search_doc])
            
#             return results
#         except Exception as e:
#             logger.error(f"Web search error: {e}")
#             return f"Error searching web: {e}"
    
#     def add_to_web_knowledge(self, documents: List[Document]) -> None:
#         if not documents:
#             return
        
#         splits = self.text_splitter.split_documents(documents)
        
#         if self.web_vectorstore:
#             self.web_vectorstore.add_documents(splits)
#         else:
#             self.web_vectorstore = FAISS.from_documents(splits, self.embeddings)
        
#         self.web_retriever = self.web_vectorstore.as_retriever(search_kwargs={"k": 3})
#         self.web_qa_chain = RetrievalQA.from_chain_type(
#             llm=self.llm,
#             chain_type="stuff",
#             retriever=self.web_retriever,
#             return_source_documents=True
#         )
#         logger.info(f"Added {len(documents)} documents to web knowledge base")
    
#     def query(self, user_query: str) -> Dict[str, Any]:
#         try:
#             result = self.agent.run(user_query)
#             return {
#                 "answer": result,
#                 "sources": "Generated from agent tools"
#             }
#         except Exception as e:
#             logger.error(f"Error processing query: {e}")
#             return {
#                 "answer": f"Sorry, I encountered an error: {e}",
#                 "sources": []
#             }

# def main() -> None:
#     rag_system = AgenticRAGSystem()
    
#     # Load PDF documents
#     pdf_files = ["./chapter3.pdf"]  # Add your PDF files here
#     rag_system.load_pdf_documents(pdf_files)
    
#     # Load web content (optional - can also rely on live search)
#     web_urls = [
#         "https://example.com/page1",
#         "https://example.com/page2"
#     ]  # Add URLs you want to pre-load
#     rag_system.load_web_content(web_urls)

#     while True:
#         user_query = input("\nEnter your query (or 'quit' to exit): ")
#         if user_query.lower() == 'quit':
#             break
        
#         result = rag_system.query(user_query)
#         print(f"\nAnswer:\n{result['answer']}")
        
#         if result["sources"]:
#             print(f"\nSources: {result['sources']}")

# if __name__ == "__main__":
#     main()


# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_community.tools import DuckDuckGoSearchRun
# from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
# from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import RetrievalQA
# from langchain.agents import AgentExecutor, Tool, initialize_agent, AgentType
# from langchain.schema import Document
# import os
# from dotenv import load_dotenv
# import logging
# from typing import List, Dict, Any, Optional

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# load_dotenv()
# GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# class AgenticRAGSystem:
#     def __init__(self):
#         self.llm = ChatGoogleGenerativeAI(
#             model="gemini-1.5-flash",
#             google_api_key=GOOGLE_API_KEY,
#             temperature=0.1
#         )
        
#         self.embeddings = GoogleGenerativeAIEmbeddings(
#             model="models/embedding-001",
#             google_api_key=GOOGLE_API_KEY
#         )
        
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200
#         )
        
#         # Separate vector stores for PDF and web content
#         self.pdf_vectorstore = None
#         self.web_vectorstore = None
        
#         self.pdf_retriever = None
#         self.web_retriever = None
        
#         self.pdf_qa_chain = None
#         self.web_qa_chain = None
        
#         self.search_tool = DuckDuckGoSearchRun()
        
#         # Threshold for considering a PDF answer as "good enough"
#         self.pdf_confidence_threshold = 0.7
    
#     def load_pdf_documents(self, pdf_paths: List[str]) -> None:
#         documents = []
#         for pdf_path in pdf_paths:
#             try:
#                 loader = PyPDFLoader(pdf_path)
#                 docs = loader.load()
#                 documents.extend(docs)
#                 logger.info(f"Loaded PDF: {pdf_path}")
#             except Exception as e:
#                 logger.error(f"Error loading PDF {pdf_path}: {e}")
        
#         if documents:
#             splits = self.text_splitter.split_documents(documents)
#             self.pdf_vectorstore = FAISS.from_documents(splits, self.embeddings)
#             self.pdf_retriever = self.pdf_vectorstore.as_retriever(search_kwargs={"k": 3})
#             self.pdf_qa_chain = RetrievalQA.from_chain_type(
#                 llm=self.llm,
#                 chain_type="stuff",
#                 retriever=self.pdf_retriever,
#                 return_source_documents=True
#             )
#             logger.info(f"Loaded {len(documents)} PDF documents into PDF knowledge base")
    
#     def load_web_content(self, urls: List[str]) -> None:
#         documents = []
#         for url in urls:
#             try:
#                 loader = WebBaseLoader(url)
#                 docs = loader.load()
#                 documents.extend(docs)
#                 logger.info(f"Loaded web content: {url}")
#             except Exception as e:
#                 logger.error(f"Error loading web content {url}: {e}")
        
#         if documents:
#             splits = self.text_splitter.split_documents(documents)
#             self.web_vectorstore = FAISS.from_documents(splits, self.embeddings)
#             self.web_retriever = self.web_vectorstore.as_retriever(search_kwargs={"k": 3})
#             self.web_qa_chain = RetrievalQA.from_chain_type(
#                 llm=self.llm,
#                 chain_type="stuff",
#                 retriever=self.web_retriever,
#                 return_source_documents=True
#             )
#             logger.info(f"Loaded {len(documents)} web documents into web knowledge base")
    
#     def query_pdf_knowledge(self, query: str) -> Dict[str, Any]:
#         if not self.pdf_qa_chain:
#             return {
#                 "answer": "No PDF documents loaded yet",
#                 "confidence": 0,
#                 "sources": []
#             }
        
#         try:
#             result = self.pdf_qa_chain({"query": query})
            
#             # Simple confidence calculation based on similarity score
#             confidence = min(1.0, max(0.0, result.get("score", 0.5)))  # Default to 0.5 if no score
            
#             return {
#                 "answer": result["result"],
#                 "confidence": confidence,
#                 "sources": [doc.metadata for doc in result["source_documents"]]
#             }
#         except Exception as e:
#             logger.error(f"Error querying PDF documents: {e}")
#             return {
#                 "answer": f"Error querying PDF documents: {e}",
#                 "confidence": 0,
#                 "sources": []
#             }
    
#     def query_web_knowledge(self, query: str) -> Dict[str, Any]:
#         if not self.web_qa_chain:
#             return {
#                 "answer": "No web content loaded yet",
#                 "confidence": 0,
#                 "sources": []
#             }
        
#         try:
#             result = self.web_qa_chain({"query": query})
            
#             # Simple confidence calculation based on similarity score
#             confidence = min(1.0, max(0.0, result.get("score", 0.5)))  # Default to 0.5 if no score
            
#             return {
#                 "answer": result["result"],
#                 "confidence": confidence,
#                 "sources": [doc.metadata for doc in result["source_documents"]]
#             }
#         except Exception as e:
#             logger.error(f"Error querying web documents: {e}")
#             return {
#                 "answer": f"Error querying web documents: {e}",
#                 "confidence": 0,
#                 "sources": []
#             }
    
#     def search_web(self, query: str) -> Dict[str, Any]:
#         try:
#             results = self.search_tool.run(query)
            
#             return {
#                 "answer": results,
#                 "confidence": 0.5,  # Default confidence for web search
#                 "sources": [{"source": "web_search", "query": query}]
#             }
#         except Exception as e:
#             logger.error(f"Web search error: {e}")
#             return {
#                 "answer": f"Error searching web: {e}",
#                 "confidence": 0,
#                 "sources": []
#             }
    
#     def is_good_answer(self, answer: str) -> bool:
#         """Simple heuristic to determine if an answer is good enough"""
#         negative_phrases = [
#             "don't know",
#             "no information",
#             "not found",
#             "no answer",
#             "error",
#             "sorry"
#         ]
        
#         answer_lower = answer.lower()
#         return not any(phrase in answer_lower for phrase in negative_phrases)
    
#     def query(self, user_query: str) -> Dict[str, Any]:
#         # First try PDF knowledge base
#         pdf_result = self.query_pdf_knowledge(user_query)
        
#         # Check if PDF answer is good enough
#         if (pdf_result["confidence"] >= self.pdf_confidence_threshold and 
#             self.is_good_answer(pdf_result["answer"])):
#             return {
#                 "answer": pdf_result["answer"],
#                 "sources": pdf_result["sources"],
#                 "source_type": "pdf"
#             }
        
#         # If PDF answer not good enough, try pre-loaded web knowledge
#         web_result = self.query_web_knowledge(user_query)
        
#         # Check if web knowledge answer is good enough
#         if (web_result["confidence"] >= self.pdf_confidence_threshold and 
#             self.is_good_answer(web_result["answer"])):
#             return {
#                 "answer": web_result["answer"],
#                 "sources": web_result["sources"],
#                 "source_type": "web_knowledge"
#             }
        
#         # If still no good answer, perform live web search
#         search_result = self.search_web(user_query)
#         return {
#             "answer": search_result["answer"],
#             "sources": search_result["sources"],
#             "source_type": "web_search"
#         }

# def main() -> None:
#     rag_system = AgenticRAGSystem()
    
#     # Load PDF documents
#     pdf_files = ["./chapter3.pdf"]  
#     rag_system.load_pdf_documents(pdf_files)
    
#     web_urls = [
#         "https://example.com/page1",
#         "https://example.com/page2"
#     ]  # Add URLs you want to pre-load
#     rag_system.load_web_content(web_urls)

#     while True:
#         user_query = input("\nEnter your query (or 'quit' to exit): ")
#         if user_query.lower() == 'quit':
#             break
        
#         result = rag_system.query(user_query)
#         print(f"\nAnswer ({result['source_type']}):\n{result['answer']}")
        
#         if result["sources"]:
#             print(f"\nSources:")
#             for i, source in enumerate(result["sources"], 1):
#                 print(f"{i}. {source.get('source', 'Unknown')}")
#                 if 'page' in source:
#                     print(f"   Page: {source['page']}")

# if __name__ == "__main__":
#     main()



# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import RetrievalQA
# import os
# from dotenv import load_dotenv
# import logging
# from typing import List, Dict, Any

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# load_dotenv()
# GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# class AgenticRAGSystem:
#     def __init__(self):
#         self.llm = ChatGoogleGenerativeAI(
#             model="gemini-1.5-flash",
#             google_api_key=GOOGLE_API_KEY,
#             temperature=0.1
#         )
        
#         self.embeddings = GoogleGenerativeAIEmbeddings(
#             model="models/embedding-001",
#             google_api_key=GOOGLE_API_KEY
#         )
        
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200
#         )
        
#         self.pdf_vectorstore1 = None 
#         self.pdf_vectorstore2 = None 
        
#         self.pdf_retriever1 = None
#         self.pdf_retriever2 = None
        
#         self.pdf_qa_chain1 = None
#         self.pdf_qa_chain2 = None
        
#         self.pdf_confidence_threshold = 0.7
    
#     def load_pdf_documents1(self, pdf_paths: List[str]) -> None:
#         """Load first collection of PDF documents"""
#         documents = []
#         for pdf_path in pdf_paths:
#             try:
#                 loader = PyPDFLoader(pdf_path)
#                 docs = loader.load()
#                 documents.extend(docs)
#                 logger.info(f"Loaded PDF to collection 1: {pdf_path}")
#             except Exception as e:
#                 logger.error(f"Error loading PDF {pdf_path}: {e}")
        
#         if documents:
#             splits = self.text_splitter.split_documents(documents)
#             self.pdf_vectorstore1 = FAISS.from_documents(splits, self.embeddings)
#             self.pdf_retriever1 = self.pdf_vectorstore1.as_retriever(search_kwargs={"k": 3})
#             self.pdf_qa_chain1 = RetrievalQA.from_chain_type(
#                 llm=self.llm,
#                 chain_type="stuff",
#                 retriever=self.pdf_retriever1,
#                 return_source_documents=True
#             )
#             logger.info(f"Loaded {len(documents)} PDF documents into PDF collection 1")
    
#     def load_pdf_documents2(self, pdf_paths: List[str]) -> None:
#         """Load second collection of PDF documents"""
#         documents = []
#         for pdf_path in pdf_paths:
#             try:
#                 loader = PyPDFLoader(pdf_path)
#                 docs = loader.load()
#                 documents.extend(docs)
#                 logger.info(f"Loaded PDF to collection 2: {pdf_path}")
#             except Exception as e:
#                 logger.error(f"Error loading PDF {pdf_path}: {e}")
        
#         if documents:
#             splits = self.text_splitter.split_documents(documents)
#             self.pdf_vectorstore2 = FAISS.from_documents(splits, self.embeddings)
#             self.pdf_retriever2 = self.pdf_vectorstore2.as_retriever(search_kwargs={"k": 3})
#             self.pdf_qa_chain2 = RetrievalQA.from_chain_type(
#                 llm=self.llm,
#                 chain_type="stuff",
#                 retriever=self.pdf_retriever2,
#                 return_source_documents=True
#             )
#             logger.info(f"Loaded {len(documents)} PDF documents into PDF collection 2")
    
#     def query_pdf_knowledge1(self, query: str) -> Dict[str, Any]:
#         """Query the first PDF collection"""
#         if not self.pdf_qa_chain1:
#             return {
#                 "answer": "No PDF documents loaded in collection 1 yet",
#                 "confidence": 0,
#                 "sources": []
#             }
        
#         try:
#             result = self.pdf_qa_chain1({"query": query})
            
#             # Simple confidence calculation based on similarity score
#             confidence = min(1.0, max(0.0, result.get("score", 0.5)))  # Default to 0.5 if no score
            
#             return {
#                 "answer": result["result"],
#                 "confidence": confidence,
#                 "sources": [doc.metadata for doc in result["source_documents"]]
#             }
#         except Exception as e:
#             logger.error(f"Error querying PDF collection 1: {e}")
#             return {
#                 "answer": f"Error querying PDF collection 1: {e}",
#                 "confidence": 0,
#                 "sources": []
#             }
    
#     def query_pdf_knowledge2(self, query: str) -> Dict[str, Any]:
#         """Query the second PDF collection"""
#         if not self.pdf_qa_chain2:
#             return {
#                 "answer": "No PDF documents loaded in collection 2 yet",
#                 "confidence": 0,
#                 "sources": []
#             }
        
#         try:
#             result = self.pdf_qa_chain2({"query": query})
            
#             # Simple confidence calculation based on similarity score
#             confidence = min(1.0, max(0.0, result.get("score", 0.5)))  # Default to 0.5 if no score
            
#             return {
#                 "answer": result["result"],
#                 "confidence": confidence,
#                 "sources": [doc.metadata for doc in result["source_documents"]]
#             }
#         except Exception as e:
#             logger.error(f"Error querying PDF collection 2: {e}")
#             return {
#                 "answer": f"Error querying PDF collection 2: {e}",
#                 "confidence": 0,
#                 "sources": []
#             }
    
#     def is_good_answer(self, answer: str) -> bool:
#         """Simple heuristic to determine if an answer is good enough"""
#         negative_phrases = [
#             "don't know",
#             "no information",
#             "not found",
#             "no answer",
#             "error",
#             "sorry"
#         ]
        
#         answer_lower = answer.lower()
#         return not any(phrase in answer_lower for phrase in negative_phrases)
    
#     def query(self, user_query: str) -> Dict[str, Any]:
#         """Query both PDF collections and return the best answer"""
#         # First try PDF collection 1
#         pdf_result1 = self.query_pdf_knowledge1(user_query)
        
#         # Check if PDF collection 1 answer is good enough
#         if (pdf_result1["confidence"] >= self.pdf_confidence_threshold and 
#             self.is_good_answer(pdf_result1["answer"])):
#             return {
#                 "answer": pdf_result1["answer"],
#                 "sources": pdf_result1["sources"],
#                 "source_type": "pdf_collection_1"
#             }
        
#         # If PDF collection 1 answer not good enough, try PDF collection 2
#         pdf_result2 = self.query_pdf_knowledge2(user_query)
        
#         # Check if PDF collection 2 answer is good enough
#         if (pdf_result2["confidence"] >= self.pdf_confidence_threshold and 
#             self.is_good_answer(pdf_result2["answer"])):
#             return {
#                 "answer": pdf_result2["answer"],
#                 "sources": pdf_result2["sources"],
#                 "source_type": "pdf_collection_2"
#             }
        
#         # If neither collection has a good answer, combine the results
#         combined_answer = (
#             f"From PDF Collection 1:\n{pdf_result1['answer']}\n\n"
#             f"From PDF Collection 2:\n{pdf_result2['answer']}"
#         )
        
#         combined_sources = pdf_result1["sources"] + pdf_result2["sources"]
        
#         return {
#             "answer": combined_answer,
#             "sources": combined_sources,
#             "source_type": "combined_pdf_collections"
#         }

# def main() -> None:
#     rag_system = AgenticRAGSystem()
    
#     # Load first collection of PDF documents
#     pdf_files1 = ["./chapter3.pdf"] 
#     rag_system.load_pdf_documents1(pdf_files1)
    
#     # Load second collection of PDF documents
#     pdf_files2 = ["./chapter4.pdf"] 
#     rag_system.load_pdf_documents2(pdf_files2)

#     while True:
#         user_query = input("\nEnter your query (or 'quit' to exit): ")
#         if user_query.lower() == 'quit':
#             break
        
#         result = rag_system.query(user_query)
#         print(f"\nAnswer ({result['source_type']}):\n{result['answer']}")
        
#         if result["sources"]:
#             print(f"\nSources:")
#             for i, source in enumerate(result["sources"], 1):
#                 print(f"{i}. {source.get('source', 'Unknown')}")
#                 if 'page' in source:
#                     print(f"   Page: {source['page']}")

# if __name__ == "__main__":
#     main()


# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import RetrievalQA
# import os
# from dotenv import load_dotenv
# import logging
# from typing import List, Dict, Any

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# load_dotenv()
# GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# class AgenticRAGSystem:
#     def __init__(self):
#         self.llm = ChatGoogleGenerativeAI(
#             model="gemini-1.5-flash",
#             google_api_key=GOOGLE_API_KEY,
#             temperature=0.1
#         )
        
#         self.embeddings = GoogleGenerativeAIEmbeddings(
#             model="models/embedding-001",
#             google_api_key=GOOGLE_API_KEY
#         )
        
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200
#         )
        
#         self.pdf_vectorstore1 = None 
#         self.pdf_vectorstore2 = None 
        
#         self.pdf_retriever1 = None
#         self.pdf_retriever2 = None
        
#         self.pdf_qa_chain1 = None
#         self.pdf_qa_chain2 = None
        
#         self.pdf_confidence_threshold = 0.7
    
#     def load_pdf_documents1(self, pdf_paths: List[str]) -> None:
#         """Load first collection of PDF documents"""
#         documents = []
#         for pdf_path in pdf_paths:
#             try:
#                 loader = PyPDFLoader(pdf_path)
#                 docs = loader.load()
#                 documents.extend(docs)
#                 logger.info(f"Loaded PDF to collection 1: {pdf_path}")
#             except Exception as e:
#                 logger.error(f"Error loading PDF {pdf_path}: {e}")
        
#         if documents:
#             splits = self.text_splitter.split_documents(documents)
#             self.pdf_vectorstore1 = FAISS.from_documents(splits, self.embeddings)
#             self.pdf_retriever1 = self.pdf_vectorstore1.as_retriever(search_kwargs={"k": 3})
#             self.pdf_qa_chain1 = RetrievalQA.from_chain_type(
#                 llm=self.llm,
#                 chain_type="stuff",
#                 retriever=self.pdf_retriever1,
#                 return_source_documents=True
#             )
#             logger.info(f"Loaded {len(documents)} PDF documents into PDF collection 1")
    
#     def load_pdf_documents2(self, pdf_paths: List[str]) -> None:
#         """Load second collection of PDF documents"""
#         documents = []
#         for pdf_path in pdf_paths:
#             try:
#                 loader = PyPDFLoader(pdf_path)
#                 docs = loader.load()
#                 documents.extend(docs)
#                 logger.info(f"Loaded PDF to collection 2: {pdf_path}")
#             except Exception as e:
#                 logger.error(f"Error loading PDF {pdf_path}: {e}")
        
#         if documents:
#             splits = self.text_splitter.split_documents(documents)
#             self.pdf_vectorstore2 = FAISS.from_documents(splits, self.embeddings)
#             self.pdf_retriever2 = self.pdf_vectorstore2.as_retriever(search_kwargs={"k": 3})
#             self.pdf_qa_chain2 = RetrievalQA.from_chain_type(
#                 llm=self.llm,
#                 chain_type="stuff",
#                 retriever=self.pdf_retriever2,
#                 return_source_documents=True
#             )
#             logger.info(f"Loaded {len(documents)} PDF documents into PDF collection 2")
    
#     def query_pdf_knowledge1(self, query: str) -> Dict[str, Any]:
#         """Query the first PDF collection"""
#         if not self.pdf_qa_chain1:
#             return {
#                 "answer": "No PDF documents loaded in collection 1 yet",
#                 "confidence": 0
#             }
        
#         try:
#             result = self.pdf_qa_chain1({"query": query})
#             confidence = min(1.0, max(0.0, result.get("score", 0.5)))
#             return {
#                 "answer": result["result"],
#                 "confidence": confidence
#             }
#         except Exception as e:
#             logger.error(f"Error querying PDF collection 1: {e}")
#             return {
#                 "answer": f"Error querying PDF collection 1: {e}",
#                 "confidence": 0
#             }
    
#     def query_pdf_knowledge2(self, query: str) -> Dict[str, Any]:
#         """Query the second PDF collection"""
#         if not self.pdf_qa_chain2:
#             return {
#                 "answer": "No PDF documents loaded in collection 2 yet",
#                 "confidence": 0
#             }
        
#         try:
#             result = self.pdf_qa_chain2({"query": query})
#             confidence = min(1.0, max(0.0, result.get("score", 0.5)))
#             return {
#                 "answer": result["result"],
#                 "confidence": confidence
#             }
#         except Exception as e:
#             logger.error(f"Error querying PDF collection 2: {e}")
#             return {
#                 "answer": f"Error querying PDF collection 2: {e}",
#                 "confidence": 0
#             }
    
#     def is_good_answer(self, answer: str) -> bool:
#         """Simple heuristic to determine if an answer is good enough"""
#         negative_phrases = [
#             "don't know",
#             "no information",
#             "not found",
#             "no answer",
#             "error",
#             "sorry"
#         ]
        
#         answer_lower = answer.lower()
#         return not any(phrase in answer_lower for phrase in negative_phrases)
    
#     def query(self, user_query: str) -> Dict[str, Any]:
#         """Query both PDF collections and return the answer with quality indication"""
#         # First try PDF collection 1
#         pdf_result1 = self.query_pdf_knowledge1(user_query)
        
#         # Check if PDF collection 1 answer is good enough
#         if (pdf_result1["confidence"] >= self.pdf_confidence_threshold and 
#             self.is_good_answer(pdf_result1["answer"])):
#             return {
#                 "answer": pdf_result1["answer"],
#                 "quality": "good"
#             }
        
#         # If PDF collection 1 answer not good enough, try PDF collection 2
#         pdf_result2 = self.query_pdf_knowledge2(user_query)
        
#         # Check if PDF collection 2 answer is good enough
#         if (pdf_result2["confidence"] >= self.pdf_confidence_threshold and 
#             self.is_good_answer(pdf_result2["answer"])):
#             return {
#                 "answer": pdf_result2["answer"],
#                 "quality": "good"
#             }
        
#         # If neither collection has a good answer, combine the results
#         combined_answer = (
#             f"From PDF Collection 1:\n{pdf_result1['answer']}\n\n"
#             f"From PDF Collection 2:\n{pdf_result2['answer']}"
#         )
        
#         return {
#             "answer": combined_answer,
#             "quality": "bad"
#         }

# def main() -> None:
#     rag_system = AgenticRAGSystem()
    
#     # Load first collection of PDF documents
#     pdf_files1 = ["./chapter3.pdf"] 
#     rag_system.load_pdf_documents1(pdf_files1)
    
#     # Load second collection of PDF documents
#     pdf_files2 = ["./chapter4.pdf"] 
#     rag_system.load_pdf_documents2(pdf_files2)

#     while True:
#         user_query = input("\nEnter your query (or 'quit' to exit): ")
#         if user_query.lower() == 'quit':
#             break
        
#         result = rag_system.query(user_query)
#         print(f"\nAnswer (quality: {result['quality']}):\n{result['answer']}")

# if __name__ == "__main__":
#     main()



from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.agents import Tool
from langchain.agents import AgentExecutor
from langchain.agents import create_react_agent
import os
from dotenv import load_dotenv
import logging
from typing import List, Dict, Any, Optional, Tuple
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

class AgenticRAGSystem:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1
        )
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        self.knowledge_sources = {}  
        self.retrievers = {}
        self.qa_chains = {}
        
        self.confidence_threshold = 0.7
        self.max_iterations = 3
        
        # Initialize agent tools
        self.tools = []
        self.agent_executor = None
        
    def add_knowledge_source(self, source_name: str, pdf_paths: List[str]) -> None:
        """Add a new knowledge source with PDF documents"""
        documents = []
        for pdf_path in pdf_paths:
            try:
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Loaded PDF to {source_name}: {pdf_path}")
            except Exception as e:
                logger.error(f"Error loading PDF {pdf_path}: {e}")
        
        if documents:
            splits = self.text_splitter.split_documents(documents)
            vectorstore = FAISS.from_documents(splits, self.embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            
            # Store components
            self.knowledge_sources[source_name] = {
                "vectorstore": vectorstore,
                "retriever": retriever,
                "qa_chain": qa_chain
            }
            
            # Add as a tool for the agent
            self.tools.append(
                Tool(
                    name=source_name,
                    func=lambda query: self.query_single_source(source_name, query),
                    description=f"Useful for answering questions about {source_name}. "
                               f"Input should be a fully formed question."
                )
            )
            
            logger.info(f"Loaded {len(documents)} PDF documents into {source_name}")
    
    def query_single_source(self, source_name: str, query: str) -> str:
        """Query a single knowledge source"""
        if source_name not in self.knowledge_sources:
            return f"Knowledge source {source_name} not found"
        
        try:
            result = self.knowledge_sources[source_name]["qa_chain"]({"query": query})
            return result["result"]
        except Exception as e:
            logger.error(f"Error querying {source_name}: {e}")
            return f"Error querying {source_name}: {e}"
    
    def initialize_agent(self):
        """Initialize the agent with tools and reasoning capabilities"""
        if not self.tools:
            raise ValueError("No knowledge sources/tools available to initialize agent")
            
        # Define the agent's prompt template
        prompt_template = """Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought:{agent_scratchpad}"""
        
        prompt = PromptTemplate.from_template(prompt_template)
        
        # Create the agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=self.max_iterations,
            handle_parsing_errors=True
        )
    
    def evaluate_response_quality(self, response: str) -> Tuple[float, str]:
        """Evaluate the quality of a response and suggest improvements"""
        evaluation_prompt = f"""
        Evaluate the following response to a user query:
        
        Response: {response}
        
        Please provide:
        1. A quality score between 0 (poor) and 1 (excellent)
        2. Suggestions for improvement if score < 0.7
        3. Any missing information that should be included
        
        Return your evaluation in the format:
        SCORE: [score]
        SUGGESTIONS: [suggestions]
        MISSING: [missing information]
        """
        
        try:
            evaluation = self.llm.invoke(evaluation_prompt).content
            score = float(re.search(r"SCORE:\s*([0-9.]+)", evaluation).group(1))
            suggestions = re.search(r"SUGGESTIONS:\s*(.+)", evaluation, re.DOTALL)
            suggestions = suggestions.group(1).strip() if suggestions else ""
            missing = re.search(r"MISSING:\s*(.+)", evaluation, re.DOTALL)
            missing = missing.group(1).strip() if missing else ""
            
            return score, f"{suggestions}\n{missing}".strip()
        except Exception as e:
            logger.error(f"Error evaluating response quality: {e}")
            return 0.5, "Unable to evaluate response quality"
    
    def refine_query(self, original_query: str, feedback: str) -> str:
        """Refine the original query based on feedback"""
        refinement_prompt = f"""
        The original query was: {original_query}
        
        Based on the following feedback:
        {feedback}
        
        Please generate an improved version of the query that would likely yield better results.
        Return only the improved query without any additional commentary.
        """
        
        try:
            refined_query = self.llm.invoke(refinement_prompt).content
            return refined_query.strip('"\' \n')
        except Exception as e:
            logger.error(f"Error refining query: {e}")
            return original_query
    
    def agentic_query(self, user_query: str) -> Dict[str, Any]:
        """Perform an agentic RAG query with iterative refinement"""
        if not self.agent_executor:
            self.initialize_agent()
        
        current_query = user_query
        best_answer = None
        best_score = 0
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration} - Query: {current_query}")
            
            try:
                result = self.agent_executor.invoke({"input": current_query})
                answer = result["output"]
                
                # Evaluate the answer quality
                score, feedback = self.evaluate_response_quality(answer)
                logger.info(f"Response score: {score}")
                
                # Update best answer if this one is better
                if score > best_score:
                    best_answer = answer
                    best_score = score
                
                # If we have a good enough answer, return it
                if score >= self.confidence_threshold:
                    return {
                        "answer": answer,
                        "quality": "excellent",
                        "iterations": iteration,
                        "score": score
                    }
                
                # Otherwise refine the query and try again
                if iteration < self.max_iterations:
                    current_query = self.refine_query(current_query, feedback)
                    logger.info(f"Refined query: {current_query}")
            
            except Exception as e:
                logger.error(f"Error in agentic query iteration {iteration}: {e}")
                if best_answer:
                    return {
                        "answer": best_answer,
                        "quality": "partial",
                        "error": str(e),
                        "iterations": iteration,
                        "score": best_score
                    }
                else:
                    return {
                        "answer": f"Error processing your query: {e}",
                        "quality": "error",
                        "iterations": iteration,
                        "score": 0
                    }
        
        return {
            "answer": best_answer if best_answer else "Unable to generate a satisfactory answer",
            "quality": "good" if best_score >= 0.5 else "partial",
            "iterations": iteration,
            "score": best_score,
            "note": "Consider rephrasing your query or providing more context"
        }

def main() -> None:
    rag_system = AgenticRAGSystem()
    
    # Add knowledge sources
    rag_system.add_knowledge_source("Metals and Non-metals", ["./chapter3.pdf"])
    rag_system.add_knowledge_source("carbon and its compound", ["./chapter4.pdf"])
    
    rag_system.initialize_agent()

    while True:
        user_query = input("\nEnter your query (or 'quit' to exit): ")
        if user_query.lower() == 'quit':
            break
        
        result = rag_system.agentic_query(user_query)
        print(f"\nAnswer (quality: {result['quality']}, confidence: {result['score']:.2f}):")
        print(result["answer"])
        if result.get("note"):
            print("\nNote:", result["note"])

if __name__ == "__main__":
    main()