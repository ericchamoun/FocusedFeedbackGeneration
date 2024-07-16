from asyncio import sleep
import langchain
from agents.agent import Agent
from googleapiclient.discovery import build
from agents.question_answerer import QuestionAnswerer
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from contextlib import contextmanager
from utils.html2lines import url2text
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

langchain.verbose = False
langchain.debug = False


class SearchAgent(Agent):

    api_key = config.google_api_key
    search_engine_id = config.search_engine_id

    # List of blacklisted websites consisting of peer reviewing websites, social media/non-scientific sources and websites that cause bugs
    blacklist = [
        "jstor.org", 
        "facebook.com", 
        "twitter.com",
        "reddit.com",
        "ftp.cs.princeton.edu", 
        "ed.gov",
        "nlp.cs.princeton.edu",
        "openreview",
        "f1000research",
        "peerj",

    ]

    cache = None

    def __init__(self) -> None:
        super().__init__()
        self.cache = {}
        self.question_answerer = QuestionAnswerer()

    def __google_search__(self, search_term, **kwargs):
        service = build("customsearch", "v1", developerKey=self.api_key)
        res = service.cse().list(q=search_term, cx=self.search_engine_id, **kwargs).execute()

        if "items" in res:
            return res['items']
        else:
            return []

    @contextmanager
    def suppress_stdout(self):
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout

    def handle_individual_result(self, search_result, is_pdf = 0):
        embeddings = OpenAIEmbeddings(model=config.embedding_model,disallowed_special=())
        text_splitter = CharacterTextSplitter(
            separator= "\n",
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len
        )
        evidence  = "" if is_pdf else search_result["evidence"]
        if is_pdf:
            try:
                loader = PyPDFLoader(search_result["link"])
                pages = loader.load_and_split()

                for page in pages:
                    evidence += page.page_content
            except:
#                print("WARNING: PDF processing has failed for an individual result, ignoring.")
                return ""
        
        try:
            chunks = text_splitter.create_documents(text_splitter.split_text(evidence))
        except:
#            print("WARNING: Text chunking has failed for an individual result, ignoring.")
            return ""

        vectorstore = Chroma.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        prompt = PromptTemplate(input_variables=['context', 'question'], template="Answer the question based on the context below. IF the question cannot be answered based on the context, return exactly 'I don't know' . . \n\n{context}\n\nQuestion: {question}\nAnswer:")


        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(model_name=config.llm), chain_type="stuff",
            retriever=retriever, return_source_documents=True, chain_type_kwargs={"prompt": prompt})

        result = qa({"query": search_result["search_string"]})
        result = result["result"]
        vectorstore.delete_collection()

        # Template dictates that "If you don't know the answer, just say that you don't know, don't try to make up an answer"
        if "I don't know" in result or "does not provide information about" in result or "I'm sorry" in result\
                or "I do not know" in result or "it is not clear" in result or (len(result) == 0):
            return ""

        return result


    def process_search_results(self, results, search_string):
        for result in results:
            link = str(result["link"])
            invalid_search_result = False
            search_result = {"search_string": search_string}

            for b_link in self.blacklist:
                if b_link in link:
                    invalid_search_result = True
        
            if link.endswith(".doc"):
                invalid_search_result = True

            if invalid_search_result:
                continue

            if link.endswith(".pdf"):
                
                search_result["link"] = link
                answer = self.handle_individual_result(search_result, is_pdf = 1).replace("\n"," ").replace("- ","")
                if answer == "":
                    continue

                # According to the query, a different paragraph may be stored so we need different keys
                storage_key = link + " " + search_string

                if storage_key in self.cache:
                    yield self.cache[storage_key]
                else:
                    self.cache[storage_key] = {"answer": answer, "source": link}
                    yield self.cache[storage_key]

            storage_key = link + " " + search_string
            # We cache all downloaded websites. For now, each searcher has one cache. If we use multiple searchers we may want to unify. We may also want to place a limit on how many documents we store.
            if storage_key in self.cache:
                yield self.cache[storage_key]
            else:
                evidence = url2text(link)
                if (len(evidence) == 0) or (evidence == "[]"):
                    continue
                search_result["evidence"] = evidence
                answer = self.handle_individual_result(search_result, is_pdf = 0)
                self.cache[storage_key] = {"answer": answer, "source": link}
                yield self.cache[storage_key]

    def get_paper_results(self, search_string, paper_url):
        source = "the paper"
        search_result = {"search_string" : search_string, "link": paper_url}
        source_doc = self.handle_individual_result( search_result, is_pdf = 1)

        if source_doc != "":
            storage_key = paper_url + " " + search_string
            if storage_key in self.cache:
                return [self.cache[storage_key]]
            else:
                self.cache[storage_key] = {"answer": source_doc, "source": source}
                return [self.cache[storage_key]]
        else:
            return []

    def get_google_search_results(self, search_string, sort_date=None, page=0):
        search_results = []
        sort = None if sort_date is None else ("date:r:19000101:"+sort_date)

        for _ in range(3):
            try:
                search_results += self.__google_search__(
                    search_string,
                    num=10,
                    start=0,
                    sort=sort,
                    dateRestrict=None,
                    gl="US"
                )
            except:
                print("[Investigator]: I encountered an error trying to search +\""+search_string+"\". Maybe the connection dropped. Trying again in 3 seconds...")
                sleep(3)
        return self.process_search_results(search_results, search_string)

    def get_search_query(self, question):
        return [question]

    def answer_question(self, question, paper_url, from_paper , date_asked=None):
        if not question.endswith("?"):
            question += "?"

        search_queries = self.get_search_query(question)
        stored_answers = []
        for query in search_queries:
            page = 0
            if from_paper:
                search_results = self.get_paper_results(query, paper_url)
            else:
                search_results = self.get_google_search_results(query, sort_date=date_asked, page=page)
            # Check if we have any answers to the question:
            for s in search_results:
                answer = s["answer"]
                source = s["source"]

                answer = self.question_answerer.get_answer_from_evidence(question, answer, source)
                if answer["answer"] == "unavailable":
                    continue

                stored_answers.append(answer)

                if len(stored_answers) == 0:
                    stored_answers.append(answer)

                if len(stored_answers) > 0:
                    break


        if len(stored_answers) == 0:
            return {
                "actor": "Investigator",
                "answers": [],
                "description": "I could not find any answers.",
                "action": "answer the question \""+question+"\""
            }

        resolution = {
            "actor": "Investigator",
            "answers": stored_answers,
            "description": "I found the following answers:\n" + "\n".join([" * According to " + a["source"] + ", \""+ a["backing"] + "\"" for a in stored_answers]),
            "action": "answer the question \""+question+"\"",
            "question": question
        }
        print(resolution["description"])
        return resolution



    def get_actions(self):
        return [
            "Actor: Investigator | Action: Answer question using the paper | Parameters: question | Description: Answer the provided question from the provided paper. It is important that the query that is searched is a question ending in '?'.",
            "Actor: Investigator | Action: Answer question using Google | Parameters: question | Description: Use google search to try to answer the provided question. It is important that the query that is searched on Google is a question ending in '?'.",

        ]
    
    
    def interpret(self, action):
        if action["action"] == "Do a google search":
            return self.get_google_search_results(action["parameters"]["query"])
        elif action["action"] == "Answer question using the paper":
            return self.answer_question(action["parameters"]["question"], action["paper_url"], from_paper = 1)
        elif action["action"] == "Answer question using Google":
            return self.answer_question(action["parameters"]["question"], action["paper_url"], from_paper = 0)
        else:
            return {
                "actor": "Investigator",
                "description": "This action is not in my set of possible instructions.",
                "action": action["action"]
            }
