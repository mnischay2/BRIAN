import ollama

from scripts.configs.config import CONF

def refined_prompt_generator(query=None,target_tool=None,target_max_run=int(1)):
    sys_message = f"your goal is to generate a user query that is specific and actionable. The user query is: {query}."  
    
    if target_tool:
        target_tool = target_tool.lower()
        if target_tool == "rag_retriever":
            sys_message = f"your goal is to generate an array of {str(max(CONF.get('RAG_PHASES', 2), target_max_run))} short refined phrases to be more specific and actionable for the RAG retriever. The user query is: {query}. The RAG retriever will search the knowledge base for relevant information."
        
        elif target_tool == "web_searcher":
            sys_message = f"your goal is to generate an array of {str(max(CONF.get('WEB_PHASES', 2), target_max_run))} refined queries to be more specific and actionable for the WEB searcher. The user query is: {query}. The WEB searcher will search the web for relevant information."
        
        elif target_tool is not None:
            sys_message = f"your goal is to refine the user query to be more specific and actionable for the target tool. The user query is: {query}. THe tool is will be used for : {target_tool}."
        
    response_ = ollama.chat(                       
        model=CONF["LLM_MODEL"],
        messages=[
            {   "role": "system",
                "content": f"{sys_message} \n Only generate the refined query without any additional text or explanation. If the query is already specific and actionable, return it as is."
            },],
        stream=False,
        think=False,
        )
    
    return response_['message']['content']
