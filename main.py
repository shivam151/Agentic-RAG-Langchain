# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_community.tools import DuckDuckGoSearchRun
# from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
# from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import RetrievalQA
# from langchain.agents import AgentExecutor, create_react_agent
# from langchain.tools import Tool
# from langchain import hub
# from langchain.schema import Document
# from langchain.prompts import PromptTemplate

# from dotenv import load_dotenv
# import os
# import requests
# from bs4 import BeautifulSoup
# import logging
# from typing import List, Dict, Any, Optional
# import tempfile

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# load_dotenv()

# GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
# GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# class WebScrapingAgent:
#     """Agent responsible for web searching and scraping"""
    
#     def __init__(self, llm: ChatGoogleGenerativeAI):
#         self.llm = llm
#         self.search_tool = DuckDuckGoSearchRun()
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200
#         )
    
#     def search_web(self, query: str) -> str:
#         """Search the web for information"""
#         try:
#             results = self.search_tool.run(query)
#             return results
#         except Exception as e:
#             logger.error(f"Web search error: {e}")
#             return f"Error searching web: {e}"
    
#     def scrape_urls(self, urls: List[str]) -> List[Document]:
#         """Scrape content from URLs"""
#         documents = []
#         for url in urls:
#             try:
#                 loader = WebBaseLoader(url)
#                 docs = loader.load()
#                 documents.extend(docs)
#                 logger.info(f"Successfully scraped: {url}")
#             except Exception as e:
#                 logger.error(f"Error scraping {url}: {e}")
#         return documents
    
#     def extract_urls_from_search(self, search_results: str) -> List[str]:
#         """Extract URLs from search results (simplified)"""
#         urls = []
#         lines = search_results.split('\n')
#         for line in lines:
#             if 'http' in line:
#                 start = line.find('http')
#                 if start != -1:
#                     end = line.find(' ', start)
#                     if end == -1:
#                         end = len(line)
#                     url = line[start:end].strip('.,;)')
#                     urls.append(url)
#         return urls[:3]  # Limit to top 3 URLs
    
#     def process_web_query(self, query: str) -> List[Document]:
#         """Complete web processing pipeline"""
#         search_results = self.search_web(query)
#         urls = self.extract_urls_from_search(search_results)
        
#         # Scrape URLs
#         documents = []
#         if urls:
#             documents = self.scrape_urls(urls)
        
#         # Add search results as a document too
#         search_doc = Document(
#             page_content=search_results,
#             metadata={"source": "search_results", "query": query}
#         )
#         documents.append(search_doc)
        
#         return documents

# class PDFProcessingAgent:
#     """Agent responsible for PDF processing and RAG"""
    
#     def __init__(self, llm: ChatGoogleGenerativeAI, embeddings: GoogleGenerativeAIEmbeddings):
#         self.llm = llm
#         self.embeddings = embeddings
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200
#         )
#         self.vectorstore: Optional[FAISS] = None
#         self.qa_chain: Optional[RetrievalQA] = None
    
#     def load_pdf(self, pdf_path: str) -> List[Document]:
#         """Load and process PDF"""
#         try:
#             loader = PyPDFLoader(pdf_path)
#             documents = loader.load()
#             logger.info(f"Loaded PDF: {pdf_path}")
#             return documents
#         except Exception as e:
#             logger.error(f"Error loading PDF {pdf_path}: {e}")
#             return []
    
#     def create_vectorstore(self, documents: List[Document]) -> None:
#         """Create vector store from documents"""
#         if not documents:
#             logger.warning("No documents to create vectorstore")
#             return
        
#         splits = self.text_splitter.split_documents(documents)
#         self.vectorstore = FAISS.from_documents(splits, self.embeddings)
#         logger.info(f"Created vectorstore with {len(splits)} chunks")
    
#     def setup_qa_chain(self) -> None:
#         """Setup QA chain"""
#         if not self.vectorstore:
#             logger.error("No vectorstore available for QA chain")
#             return
        
#         retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
#         self.qa_chain = RetrievalQA.from_chain_type(
#             llm=self.llm,
#             chain_type="stuff",
#             retriever=retriever,
#             return_source_documents=True
#         )
#         logger.info("QA chain setup complete")
    
#     def query_documents(self, query: str) -> Dict[str, Any]:
#         print(query,"query")
#         """Query the document store"""
#         if not self.qa_chain:
#             return {
#                 "answer": "No documents loaded or QA chain not setup", 
#                 "source_documents": []
#             }
        
#         try:
#             result = self.qa_chain({"query": query})
#             return result
#         except Exception as e:
#             logger.error(f"Error querying documents: {e}")
#             return {
#                 "answer": f"Error querying documents: {e}", 
#                 "source_documents": []
#             }

# class MultiAgentRAGSystem:
#     """Main system orchestrating multiple agents"""
    
#     def __init__(self):
#         self.llm = ChatGoogleGenerativeAI(
#             model="gemini-2.0-flash",
#             google_api_key=GOOGLE_API_KEY,
#             temperature=0.1
#         )
        
#         self.embeddings = GoogleGenerativeAIEmbeddings(
#             model="models/embedding-001",
#             google_api_key=GOOGLE_API_KEY
#         )
        
#         # Initialize agents
#         self.web_agent = WebScrapingAgent(self.llm)
#         self.pdf_agent = PDFProcessingAgent(self.llm, self.embeddings)
        
#         self.all_documents: List[Document] = []
        
#     def load_pdf_documents(self, pdf_paths: List[str]) -> None:
#         """Load PDF documents"""
#         pdf_documents = []
#         for pdf_path in pdf_paths:
#             if os.path.exists(pdf_path):
#                 docs = self.pdf_agent.load_pdf(pdf_path)
#                 pdf_documents.extend(docs)
#             else:
#                 logger.warning(f"PDF file not found: {pdf_path}")
        
#         self.all_documents.extend(pdf_documents)
#         logger.info(f"Loaded {len(pdf_documents)} PDF documents")
    
#     def search_and_scrape_web(self, query: str) -> None:
#         """Search web and scrape content"""
#         web_documents = self.web_agent.process_web_query(query)
#         self.all_documents.extend(web_documents)
#         logger.info(f"Added {len(web_documents)} web documents")
    
#     def setup_combined_rag(self) -> None:
#         """Setup RAG system with all documents"""
#         if self.all_documents:
#             self.pdf_agent.create_vectorstore(self.all_documents)
#             self.pdf_agent.setup_qa_chain()
#             logger.info("Combined RAG system setup complete")
#         else:
#             logger.warning("No documents available for RAG setup")
    
#     def query_system(self, query: str) -> Dict[str, Any]:
#         """Query the combined system"""
#         result = self.pdf_agent.query_documents(query)
        
#         sources = []
#         if "source_documents" in result:
#             for doc in result["source_documents"]:
#                 source_info = {
#                     "content": doc.page_content[:200] + "...",
#                     "metadata": doc.metadata
#                 }
#                 sources.append(source_info)
        
#         return {
#             "answer": result["answer"],
#             "sources": sources,
#             "total_documents": len(self.all_documents)
#         }

# def save_vectorstore(vectorstore: FAISS, path: str) -> None:
#     """Save vectorstore to disk"""
#     try:
#         vectorstore.save_local(path)
#         logger.info(f"Vectorstore saved to {path}")
#     except Exception as e:
#         logger.error(f"Error saving vectorstore: {e}")

# def load_vectorstore(path: str, embeddings: GoogleGenerativeAIEmbeddings) -> Optional[FAISS]:
#     """Load vectorstore from disk"""
#     try:
#         vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
#         logger.info(f"Vectorstore loaded from {path}")
#         return vectorstore
#     except Exception as e:
#         logger.error(f"Error loading vectorstore: {e}")
#         return None


# def main() -> None:
#     rag_system = MultiAgentRAGSystem()
    
#     pdf_files = [
#         "./ncrt.pdf.pdf",  
#     ]
    
#     rag_system.load_pdf_documents(pdf_files)
    
#     web_query = "Education content"
#     print(f"\nSearching web for: {web_query}")
#     rag_system.search_and_scrape_web(web_query)
    
#     rag_system.setup_combined_rag()
    
#     while True:
#         user_query = input("\nEnter your query?")
#         if user_query.lower() == 'quit':
#             break
#         print(user_query,"user_query")
#         result = rag_system.query_system(user_query)
#         print(result["answer"])
        
#         if result["sources"]:
#             print(f"\n=== Sources ({len(result['sources'])}) ===")
#             for i, source in enumerate(result["sources"], 1):
#                 print(f"{i}. Source: {source['metadata'].get('source', 'Unknown')}")
#                 print(f"   Content preview: {source['content']}")
        
#         print(f"\nTotal documents in system: {result['total_documents']}")

# if __name__ == "__main__":
#     main()


# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_community.tools import DuckDuckGoSearchRun
# from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
# from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import RetrievalQA
# from langchain.schema import Document
# from dotenv import load_dotenv
# import os
# import logging
# from typing import List, Dict, Any, Optional
 
# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
 
# load_dotenv()
# GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
 
 
# class WebScrapingAgent:
#     def __init__(self, llm: ChatGoogleGenerativeAI):
#         self.llm = llm
#         self.search_tool = DuckDuckGoSearchRun()
#         self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
 
#     def search_web(self, query: str) -> str:
#         try:
#             results = self.search_tool.run(query)
#             return results
#         except Exception as e:
#             logger.error(f"Web search error: {e}")
#             return f"Error searching web: {e}"
 
#     def scrape_urls(self, urls: List[str]) -> List[Document]:
#         documents = []
#         for url in urls:
#             try:
#                 loader = WebBaseLoader(url)
#                 docs = loader.load()
#                 documents.extend(docs)
#                 logger.info(f"Successfully scraped: {url}")
#             except Exception as e:
#                 logger.error(f"Error scraping {url}: {e}")
#         return documents
 
#     def extract_urls_from_search(self, search_results: str) -> List[str]:
#         urls = []
#         lines = search_results.split('\n')
#         for line in lines:
#             if 'http' in line:
#                 start = line.find('http')
#                 if start != -1:
#                     end = line.find(' ', start)
#                     if end == -1:
#                         end = len(line)
#                     url = line[start:end].strip('.,;)')
#                     urls.append(url)
#         return urls[:3]
 
#     def process_web_query(self, query: str) -> List[Document]:
#         search_results = self.search_web(query)
#         urls = self.extract_urls_from_search(search_results)
#         documents = []
#         if urls:
#             documents = self.scrape_urls(urls)
#         search_doc = Document(
#             page_content=search_results,
#             metadata={"source": "search_results", "query": query}
#         )
#         documents.append(search_doc)
#         return documents
 
 
# class PDFProcessingAgent:
#     def __init__(self, llm: ChatGoogleGenerativeAI, embeddings: GoogleGenerativeAIEmbeddings):
#         self.llm = llm
#         self.embeddings = embeddings
#         self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         self.vectorstore: Optional[FAISS] = None
#         self.qa_chain: Optional[RetrievalQA] = None
 
#     def load_pdf(self, pdf_path: str) -> List[Document]:
#         try:
#             loader = PyPDFLoader(pdf_path)
#             documents = loader.load()
#             logger.info(f"Loaded PDF: {pdf_path}")
#             return documents
#         except Exception as e:
#             logger.error(f"Error loading PDF {pdf_path}: {e}")
#             return []
 
#     def create_vectorstore(self, documents: List[Document]) -> None:
#         if not documents:
#             logger.warning("No documents to create vectorstore")
#             return
#         splits = self.text_splitter.split_documents(documents)
#         self.vectorstore = FAISS.from_documents(splits, self.embeddings)
#         logger.info(f"Created vectorstore with {len(splits)} chunks")
 
#     def setup_qa_chain(self) -> None:
#         if not self.vectorstore:
#             logger.error("No vectorstore available for QA chain")
#             return
#         retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
#         self.qa_chain = RetrievalQA.from_chain_type(
#             llm=self.llm,
#             chain_type="stuff",
#             retriever=retriever,
#             return_source_documents=True
#         )   
#         logger.info("QA chain setup complete")
 
#     def query_documents(self, query: str) -> Dict[str, Any]:
#         if not self.qa_chain:
#             return {
#                 "answer": "No documents loaded or QA chain not setup",
#                 "source_documents": []
#             }
#         try:
#             result = self.qa_chain({"query": query})
#             return result
#         except Exception as e:
#             logger.error(f"Error querying documents: {e}")
#             return {
#                 "answer": f"Error querying documents: {e}",
#                 "source_documents": []
#             }
 
 
# class MultiAgentRAGSystem:
#     def __init__(self):
#         self.llm = ChatGoogleGenerativeAI(
#             model="gemini-2.0-flash",
#             google_api_key=GOOGLE_API_KEY,
#             temperature=0.1
#         )
#         self.embeddings = GoogleGenerativeAIEmbeddings(
#             model="models/embedding-001",
#             google_api_key=GOOGLE_API_KEY
#         )
#         self.web_agent = WebScrapingAgent(self.llm)
#         self.pdf_agent = PDFProcessingAgent(self.llm, self.embeddings)
#         self.all_documents: List[Document] = []
 
#     def load_pdf_documents(self, pdf_paths: List[str]) -> None:
#         pdf_documents = []
#         for pdf_path in pdf_paths:
#             if os.path.exists(pdf_path):
#                 docs = self.pdf_agent.load_pdf(pdf_path)
#                 pdf_documents.extend(docs)
#             else:
#                 logger.warning(f"PDF file not found: {pdf_path}")
#         self.all_documents.extend(pdf_documents)
#         logger.info(f"Loaded {len(pdf_documents)} PDF documents")
 
#     def setup_combined_rag(self) -> None:
#         if self.all_documents:
#             self.pdf_agent.create_vectorstore(self.all_documents)
#             self.pdf_agent.setup_qa_chain()
#             logger.info("Combined RAG system setup complete")
#         else:
#             logger.warning("No documents available for RAG setup")
 
#     def query_system(self, query: str) -> Dict[str, Any]:
#         result = self.pdf_agent.query_documents(query)
#         answer = result.get("answer", "").strip()
#         sources = result.get("source_documents", [])
 
#         if not answer or "No documents" in answer or len(answer) < 20:
#             logger.info("PDF RAG insufficient. Falling back to web.")
#             web_documents = self.web_agent.process_web_query(query)
#             if web_documents:
#                 self.all_documents.extend(web_documents)
#                 self.pdf_agent.create_vectorstore(self.all_documents)
#                 self.pdf_agent.setup_qa_chain()
#                 result = self.pdf_agent.query_documents(query)
#                 answer = result.get("answer", "")
#                 sources = result.get("source_documents", [])
 
#         formatted_sources = []
#         for doc in sources:
#             formatted_sources.append({
#                 "content": doc.page_content[:200] + "...",
#                 "metadata": doc.metadata
#             })
 
#         return {
#             "answer": answer,
#             "sources": formatted_sources,
#             "total_documents": len(self.all_documents)
#         }
 
 
# def main() -> None:
#     rag_system = MultiAgentRAGSystem()
#     pdf_files = ["./ncrt.pdf"]  
#     rag_system.load_pdf_documents(pdf_files)
#     rag_system.setup_combined_rag()
 
#     while True:
#         user_query = input("\nEnter your query? ")
#         if user_query.lower() == 'quit':
#             break
#         result = rag_system.query_system(user_query)
#         print(f"\nAnswer:\n{result['answer']}")
#         if result["sources"]:
#             print(f"\n=== Sources ({len(result['sources'])}) ===")
#             for i, source in enumerate(result["sources"], 1):
#                 print(f"{i}. Source: {source['metadata'].get('source', 'Unknown')}")
#                 print(f"   Content preview: {source['content']}")
#         print(f"\nTotal documents in system: {result['total_documents']}")
 
 
# if __name__ == "__main__":
#     main()




# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_community.tools import DuckDuckGoSearchRun
# from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
# from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import RetrievalQA
# from langchain.schema import Document
# from dotenv import load_dotenv
# import os
# import logging
# from typing import List, Dict, Any, Optional

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# load_dotenv()
# GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")


# class WebScrapingAgent:
#     def __init__(self, llm: ChatGoogleGenerativeAI):
#         self.llm = llm
#         self.search_tool = DuckDuckGoSearchRun()
#         self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

#     def search_web(self, query: str) -> str:
#         try:
#             results = self.search_tool.run(query)
#             return results
#         except Exception as e:
#             logger.error(f"Web search error: {e}")
#             return f"Error searching web: {e}"

#     def scrape_urls(self, urls: List[str]) -> List[Document]:
#         documents = []
#         for url in urls:
#             try:
#                 loader = WebBaseLoader(url)
#                 docs = loader.load()
#                 documents.extend(docs)
#                 logger.info(f"Successfully scraped: {url}")
#             except Exception as e:
#                 logger.error(f"Error scraping {url}: {e}")
#         return documents

#     def extract_urls_from_search(self, search_results: str) -> List[str]:
#         urls = []
#         lines = search_results.split('\n')
#         for line in lines:
#             if 'http' in line:
#                 start = line.find('http')
#                 if start != -1:
#                     end = line.find(' ', start)
#                     if end == -1:
#                         end = len(line)
#                     url = line[start:end].strip('.,;)')
#                     urls.append(url)
#         return urls[:3]

#     def process_web_query(self, query: str) -> List[Document]:
#         search_results = self.search_web(query)
#         urls = self.extract_urls_from_search(search_results)
#         documents = []
#         if urls:
#             documents = self.scrape_urls(urls)
#         search_doc = Document(
#             page_content=search_results,
#             metadata={"source": "search_results", "query": query}
#         )
#         documents.append(search_doc)
#         return documents


# class PDFProcessingAgent:
#     def __init__(self, llm: ChatGoogleGenerativeAI, embeddings: GoogleGenerativeAIEmbeddings):
#         self.llm = llm
#         self.embeddings = embeddings
#         self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         self.vectorstore: Optional[FAISS] = None
#         self.qa_chain: Optional[RetrievalQA] = None

#     def load_pdf(self, pdf_path: str) -> List[Document]:
#         try:
#             loader = PyPDFLoader(pdf_path)
#             documents = loader.load()
#             logger.info(f"Loaded PDF: {pdf_path}")
#             return documents
#         except Exception as e:
#             logger.error(f"Error loading PDF {pdf_path}: {e}")
#             return []

#     def create_vectorstore(self, documents: List[Document]) -> None:
#         if not documents:
#             logger.warning("No documents to create vectorstore")
#             return
#         splits = self.text_splitter.split_documents(documents)
#         self.vectorstore = FAISS.from_documents(splits, self.embeddings)
#         logger.info(f"Created vectorstore with {len(splits)} chunks")

#     def setup_qa_chain(self) -> None:
#         if not self.vectorstore:
#             logger.error("No vectorstore available for QA chain")
#             return
#         retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
#         self.qa_chain = RetrievalQA.from_chain_type(
#             llm=self.llm,
#             chain_type="stuff",
#             retriever=retriever,
#             return_source_documents=True
#         )
#         logger.info("QA chain setup complete")

#     def query_documents(self, query: str) -> Dict[str, Any]:
#         if not self.qa_chain:
#             return {
#                 "answer": "No documents loaded or QA chain not setup",
#                 "source_documents": []
#             }
#         try:
#             result = self.qa_chain({"query": query})
#             return result
#         except Exception as e:
#             logger.error(f"Error querying documents: {e}")
#             return {
#                 "answer": f"Error querying documents: {e}",
#                 "source_documents": []
#             }


# class MultiAgentRAGSystem:
#     def __init__(self):
#         self.llm = ChatGoogleGenerativeAI(
#             model="gemini-2.0-flash",
#             google_api_key=GOOGLE_API_KEY,
#             temperature=0.1
#         )
#         self.embeddings = GoogleGenerativeAIEmbeddings(
#             model="models/embedding-001",
#             google_api_key=GOOGLE_API_KEY
#         )
#         self.web_agent = WebScrapingAgent(self.llm)
#         self.pdf_agent = PDFProcessingAgent(self.llm, self.embeddings)
#         self.all_documents: List[Document] = []

#     def load_pdf_documents(self, pdf_paths: List[str]) -> None:
#         pdf_documents = []
#         for pdf_path in pdf_paths:
#             if os.path.exists(pdf_path):
#                 docs = self.pdf_agent.load_pdf(pdf_path)
#                 pdf_documents.extend(docs)
#             else:
#                 logger.warning(f"PDF file not found: {pdf_path}")
#         self.all_documents.extend(pdf_documents)
#         logger.info(f"Loaded {len(pdf_documents)} PDF documents")

#     def setup_combined_rag(self) -> None:
#         if self.all_documents:
#             self.pdf_agent.create_vectorstore(self.all_documents)
#             self.pdf_agent.setup_qa_chain()
#             logger.info("Combined RAG system setup complete")
#         else:
#             logger.warning("No documents available for RAG setup")

#     def query_system(self, query: str) -> Dict[str, Any]:
#         # Step 1: Try PDF (book-based) response first
#         result = self.pdf_agent.query_documents(query)
#         answer = result.get("answer", "").strip()
#         print(answer,"answer")
#         sources = result.get("source_documents", [])
#         print(sources,"sources")

#         is_valid = answer and not answer.lower().startswith("no documents") and len(answer) > 20

#         if not is_valid:
#             logger.info("No good answer found in PDF. Falling back to web search.")
#             web_documents = self.web_agent.process_web_query(query)
#             if web_documents:
#                 self.all_documents.extend(web_documents)
#                 self.pdf_agent.create_vectorstore(self.all_documents)
#                 self.pdf_agent.setup_qa_chain()
#                 result = self.pdf_agent.query_documents(query)
#                 answer = result.get("answer", "").strip()
#                 sources = result.get("source_documents", [])

#         formatted_sources = []
#         for doc in sources:
#             formatted_sources.append({
#                 "content": doc.page_content[:200] + "...",
#                 "metadata": doc.metadata
#             })

#         return {
#             "answer": answer or "Sorry, no answer found.",
#             "sources": formatted_sources,
#             "total_documents": len(self.all_documents)
#         }


# def main() -> None:
#     rag_system = MultiAgentRAGSystem()
#     pdf_files = ["./chapter3.pdf"] 
#     rag_system.load_pdf_documents(pdf_files)
#     rag_system.setup_combined_rag()

#     while True:
#         user_query = input("\nEnter your query? ")
#         if user_query.lower() == 'quit':
#             break
#         result = rag_system.query_system(user_query)
#         print(f"\nAnswer:\n{result['answer']}")
#         if result["sources"]:
#             print(f"\n=== Sources ({len(result['sources'])}) ===")
#             for i, source in enumerate(result["sources"], 1):
#                 print(f"{i}. Source: {source['metadata'].get('source', 'Unknown')}")
#                 print(f"   Content preview: {source['content']}")
#         print(f"\nTotal documents in system: {result['total_documents']}")


# if __name__ == "__main__":
#     main()



from langgraph.prebuilt import ToolInvocation
import json
from langchain_core.messages import FunctionMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated
from langchain_core.agents import AgentActionMessageLog, AgentFinish
from langchain_core.messages import BaseMessage
import operator

# Define the agent state
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    query: str
    documents: List[Document]
    answer: str
    sources: List[Dict[str, Any]]

class AgenticRAGSystem:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        self.web_agent = WebScrapingAgent(self.llm)
        self.pdf_agent = PDFProcessingAgent(self.llm, self.embeddings)
        self.all_documents: List[Document] = []
        
        # Define tools
        self.tools = [
            {
                "name": "query_pdf_knowledge",
                "description": "Query the PDF knowledge base for answers to questions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The question to ask the PDF knowledge base"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "search_web",
                "description": "Search the web for current information when PDF knowledge is insufficient",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to look up on the web"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Initialize workflow
        self.workflow = StateGraph(AgentState)
        self.workflow.add_node("agent", self.call_model)
        self.workflow.add_node("action", self.call_tool)
        
        # Define edges
        self.workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "continue": "action",
                "end": END
            }
        )
        
        self.workflow.add_edge("action", "agent")
        self.workflow.set_entry_point("agent")
        
        self.app = self.workflow.compile()
    
    def load_pdf_documents(self, pdf_paths: List[str]) -> None:
        pdf_documents = []
        for pdf_path in pdf_paths:
            if os.path.exists(pdf_path):
                docs = self.pdf_agent.load_pdf(pdf_path)
                pdf_documents.extend(docs)
            else:
                logger.warning(f"PDF file not found: {pdf_path}")
        self.all_documents.extend(pdf_documents)
        self.pdf_agent.create_vectorstore(self.all_documents)
        self.pdf_agent.setup_qa_chain()
        logger.info(f"Loaded {len(pdf_documents)} PDF documents")
    
    def call_model(self, state: AgentState):
        messages = state["messages"]
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def call_tool(self, state: AgentState):
        last_message = state["messages"][-1]
        
        if not hasattr(last_message, "tool_calls"):
            raise ValueError("No tool calls in last message")
            
        tool_call = last_message.tool_calls[0]
        action = ToolInvocation(
            tool=tool_call["name"],
            tool_input=json.loads(tool_call["args"])
        )
        
        # Execute the tool
        if action.tool == "query_pdf_knowledge":
            result = self.pdf_agent.query_documents(action.tool_input["query"])
            response = {
                "answer": result["answer"],
                "sources": [doc.metadata for doc in result["source_documents"]]
            }
        elif action.tool == "search_web":
            search_results = self.web_agent.search_web(action.tool_input["query"])
            urls = self.web_agent.extract_urls_from_search(search_results)
            documents = self.web_agent.scrape_urls(urls)
            self.all_documents.extend(documents)
            
            # Update the vectorstore with new documents
            self.pdf_agent.create_vectorstore(self.all_documents)
            self.pdf_agent.setup_qa_chain()
            
            # Query the updated knowledge base
            result = self.pdf_agent.query_documents(action.tool_input["query"])
            response = {
                "answer": result["answer"],
                "sources": [doc.metadata for doc in result["source_documents"]]
            }
        else:
            response = {"error": f"Unknown tool: {action.tool}"}
        
        function_message = FunctionMessage(
            content=str(response),
            name=action.tool
        )
        
        return {
            "messages": [function_message],
            "answer": response.get("answer", ""),
            "sources": response.get("sources", [])
        }
    
    def should_continue(self, state: AgentState):
        last_message = state["messages"][-1]
        
        if not hasattr(last_message, "tool_calls"):
            return "end"
        
        return "continue"
    
    def query(self, user_query: str) -> Dict[str, Any]:
        # Initialize the state
        state = {
            "messages": [HumanMessage(content=user_query)],
            "query": user_query,
            "documents": self.all_documents,
            "answer": "",
            "sources": []
        }
        
        # Run the workflow
        result = self.app.invoke(state)
        
        # Get the final answer
        final_answer = None
        for message in reversed(result["messages"]):
            if isinstance(message, AIMessage) and not hasattr(message, "tool_calls"):
                final_answer = message.content
                break
        
        return {
            "answer": final_answer or "Sorry, I couldn't find an answer.",
            "sources": state["sources"],
            "total_documents": len(self.all_documents)
        }

def main() -> None:
    rag_system = AgenticRAGSystem()
    pdf_files = ["./chapter3.pdf"] 
    rag_system.load_pdf_documents(pdf_files)

    while True:
        user_query = input("\nEnter your query (or 'quit' to exit): ")
        if user_query.lower() == 'quit':
            break
        
        result = rag_system.query(user_query)
        print(f"\nAnswer:\n{result['answer']}")
        
        if result["sources"]:
            print(f"\n=== Sources ({len(result['sources'])}) ===")
            for i, source in enumerate(result["sources"], 1):
                print(f"{i}. Source: {source.get('source', 'Unknown')}")
                if 'page' in source:
                    print(f"   Page: {source['page']}")
        
        print(f"\nTotal documents in system: {result['total_documents']}")

if __name__ == "__main__":
    main()