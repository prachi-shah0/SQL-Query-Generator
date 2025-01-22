import os
import gc
from dotenv import load_dotenv
from langchain_core.messages import AIMessage,HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain.llms import OpenAI
from transformers import AutoTokenizer,AutoModelForCausalLM
import streamlit as st

#os.environ.pop("HTTPS_PROXY", None)

load_dotenv()

model=AutoModelForCausalLM.from_pretrained("mlfoundations-dev/oh-dcft-v3.1-gpt-4o-mini")
toeknizer=AutoTokenizer.from_pretrained("mlfoundations-dev/oh-dcft-v3.1-gpt-4o-mini")

#api_key=os.getenv("OPENAI_API_KEY")
#GROQ_API_KEY="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"
#os.environ["GROQ_API_KEY"]=api_key

def init_db(user:str,password:str,host:str,port:str,database:str)->SQLDatabase:
    db_uri=f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

def get_chain(db:SQLDatabase):
    template="""
    <SCHEMA>{schema}</SCHEMA>
    Conversation history:{chat_history}
    For example:
    Question:show data for the region Europe
    SQL QUERY:select * from online_sales_data where region='Europe';
    Your turn:
    Question:{question}
    SQL Query:
    """
    prompt=ChatPromptTemplate.from_template(template=template)
    #llm = OpenAI(
       # model="gpt-4o-mini",
        #groq_proxy='http://127.0.0.1:2081',
        #temperature=0.7,
        #api_key=api_key
    #)

    def gschema(_):
        return db.get_table_info()

    return (
        RunnablePassthrough.assign(schema=gschema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query:str,db:SQLDatabase,chat_history:list):
    s_chain=get_chain(db)

    t="""
    Ensure the response is concise and formatted as follows:
    - **Result Explanation**:  
    [Write the natural language explanation of the query results here.]

    - **SQL Query Used**:  
    [Write the SQL query here.]

    Context:
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}

    Your response should only include the explanation and the query in the format specified above.
    """
    prompt=ChatPromptTemplate.from_template(template=t)
    #llm = OpenAI(
        #model="gpt-4o-mini",
        #groq_proxy='http://127.0.0.1:2081',
        #temperature=0.7,
        #api_key=api_key
    #)

    chain=(
        RunnablePassthrough.assign(query=s_chain).assign(
            schema=lambda _:db.get_table_info(),
            response=lambda vars:db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({ 
        "question":user_query,
        "chat_history":chat_history,
    })

def r_chat():
    st.session_state.chat_history=[
        AIMessage(content="Hello! I am a SQL assitant.")
    ]
    gc.collect()

if "chat_history" not in st.session_state:
    r_chat()
    
st.set_page_config(page_title="SQL QUERY GENERATOR",page_icon=":speech_balloon:")

st.title("SQL Query Generator")

with st.sidebar:
    st.subheader("Settings")
    st.write("Connect to the database and start chatting with the chatbot")

    host=st.text_input("Host",value="localhost",key="Host")
    port=st.text_input("Port",value="3306",key="Port")
    user=st.text_input("User",value="root",key="user")
    password=st.text_input("Password",value="",key="Password")
    database=st.text_input("Database",value="online_sales",key="Database")

    if st.button("Connect"):
        with st.spinner("COnnecting to the database"):
            try:
                db=init_db(user,password,host,port,database)
                st.session_state.db=db
                st.success("Connected to the database")
            except Exception as e:
                st.error(f"Error while connecting the database:{e}")
        
    if st.button("Clear Chat"):
        r_chat()
    
for message in st.session_state.chat_history:
    if isinstance (message,AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message,HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    
user_query=st.chat_input("Type message")
if user_query and user_query.strip():
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        try:
            inputs=tokenizer(user_query,return_tensors='tf')
            outputs=model.generate(inputs['input_ids'],max_length=250,num_return_sequence=1)
            response=tokenizer.decode(outputs[0],skip_special_token=True)
            #response=get_response(user_query,st.session_state.db,st.session_state.chat_history)
            st.markdown(response)
            st.session_state.chat_history.append(AIMessage(content=response))
        except Exception as e:
            error_message=f"Error processing request:{e}"
            st.markdown(error_message)
            st.session_state.chat_history.append(AIMessage(content=error_message))