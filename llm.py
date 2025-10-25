from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# v1.0으로 바뀌면서 기존의 chat history는 langgraph에서 제공이 돼 일단은 MessagesPlaceholder로 구현함.
# chat history에 관련한 코드는 💬로 주석담.
chat_history = []


def get_llm(model="gpt-5-mini"):
    llm = ChatOpenAI(model=model)

    return llm


def get_retriever():
    TAX_DB_NAME = "tax-markdown"
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")
    pc = Pinecone()
    pinecone_db = PineconeVectorStore.from_existing_index(
        index_name=TAX_DB_NAME, embedding=embedding_function
    )
    retriever = pinecone_db.as_retriever()

    return retriever


def get_keyword_chain():
    llm = get_llm()

    keyword_mapping = """
      - 연봉/총수익/소득(수익)과 관련된 표현을 포괄하는 단어 -> "종합소득"으로 변경한다.
      - 직장인/사람(납세의무자) 등 인적 주체를 의미하는 단어 -> "거주자"로 변경한다.
    """

    # keyword mapping prompt
    keyword_prompt_content = f"""
      당신은 프롬프트 엔지니어링 전문가이자 쿼리를 재작성하는 전문가입니다..
      주요 임무는 사용자 질의에서 핵심 키워드를 정확히 추출하고, 제공된 #키워드 매핑 규칙을 통해 
      아래 키워드 매핑 규칙을 참고하여 #사용자 쿼리의 query를 재작성해주세요.

      단, 아래의 규칙은 준수하세요.
      - query를 변경할 필요가 없다면 변경하지 않아도 됩니다.
      - 사용자 쿼리를 유지한채 단어만 # 키워드 매핑 규칙을 참고해 변환하면 됩니다.

      #키워드 매핑 규칙
      {keyword_mapping}

      #사용자 쿼리
      query: {{query}}
    """

    keyword_prompt = PromptTemplate.from_template(keyword_prompt_content)
    keyword_chain = (
        {"query": RunnablePassthrough()} | keyword_prompt | llm | StrOutputParser()
    )

    return keyword_chain


# invoke가 아닌 stream을 통한 ai message 생성은 generator이기 때문에
# AiMessage의 content는 string을 기대하지만 generator를 받아 오류가 남을 해결하기 위한 함수
def get_string_from_stream(stream_response):
    result = ""

    for chunk in stream_response:
        result += chunk

    return result


def get_qa_chain():
    llm = get_llm()
    system_prompt = """
      당신은 최고의 한국 소득세 전문가 입니다. #Context를 참고해서 질문에 답변해주세요

      #Context
      {context}

      #User Query
      Question: {query}
    """

    # 💬
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Question: {query}"),
        ]
    )

    retriever = get_retriever()
    query_chain = (
        {
            "query": RunnablePassthrough(),
            "context": retriever,
            "chat_history": lambda x: chat_history,
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    return query_chain


def get_ai_message(query):
    keyword_chain = get_keyword_chain()
    qa_chain = get_qa_chain()
    final_chain = keyword_chain | qa_chain
    
    # ! gpt-5에서는 조직 인증 후에 stream service를 이용할 수 있어 그냥 invoke로 대체함
    # ! 즉 무료 계정은 안 된다.... ㅜㅜ
    # result = final_chain.stream(query)
    result = final_chain.invoke(query)

    # 💬 history에 최근 user, ai message를 담아 MessagesPlaceholder에 담기
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=result))

    print(chat_history)

    return result
