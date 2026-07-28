from langchain_nvidia_ai_endpoints import ChatNVIDIA , NVIDIAEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from langgraph.graph import StateGraph , START , END

from dotenv import load_dotenv
from pydantic import BaseModel , Field
from typing import TypedDict , List , Literal



# set the environment variable
load_dotenv()


# -------------------------------------- Form LLM ----------------------------------------------

llm = ChatNVIDIA(model = 'openai/gpt-oss-120b')


# -------------------------------------- RAG Proprocessing Step ----------------------------------------------

def preprocess_steps():

    try:

        # load the document
        docs = (

            PyPDFLoader(r'D:\GenAI\LangGraph\document\Company_Policies.pdf').load() + 
            PyPDFLoader(r'D:\GenAI\LangGraph\document\Company_Profile.pdf').load() + 
            PyPDFLoader(r'D:\GenAI\LangGraph\document\Company_Policies.pdf').load()
        )

        # split the document
        chunks = RecursiveCharacterTextSplitter(chunk_size = 600 , chunk_overlap = 150).split_documents(docs)

        # embedding
        embedding = NVIDIAEmbeddings(model='nvidia/nv-embedcode-7b-v1')
        vector_store = FAISS.from_documents(chunks , embedding)

        # form retriever
        retriver = vector_store.as_retriever(search_type = 'mmr' , search_kwargs = {'k':3})


        return retriver

    except Exception as e:
        print('Error occured loading the document')


retriever = preprocess_steps()



# -------------------------------------- Define State ----------------------------------------------
# Define state
class State(TypedDict):
    question : str
    docs : List[Document]
    need_retrieval : bool

    relevant_docs : List[Document]
    is_relevant : bool

    context : str
    answer : str

    # What we actually send to vector retriever
    retrieval_query : str
    rewrite_query : str
    rewrite_tries : int

    # Post-generation verification
    is_sup : Literal['fully_support' , 'partially_support' , 'no_support']
    evidence : list[str]

    retries : int

    # usefulness check
    is_use : Literal['useful' , 'not_useful']
    use_reason : str


# -------------------------------------- RAG Proprocessing Step ----------------------------------------------

# Define pydantic model for structure output
class RetrieverDecision(BaseModel):
    should_retrieve : bool = Field(... , description='True if external documents are needed to answer relaibly, else False')


# form the prompt
decide_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            'You decide whether retrieval is needed.\n'
            'Return JSON that matches this schema:\n'
            '{{should_retrieve : boolan}}\n\n'
            'Guidelines:\n'
            '- Should_retrieve=True if answering requires specific facts, citations , info likely not in the model.\n'
            '- Should_retrieve=False for general explanations , definations, or reasoning that does not need source.\n'
            '- If unsure , choose True'
        ),
        ('human' , 'Question : {question}'),
    ]
)

# IMPORTANT : no content for structure output
should_retrieve_llm = llm.with_structured_output(RetrieverDecision)

# Define the node(Whether the retrival is needed)
def decide_retrieval(state : State) -> dict:

    decision : RetrieverDecision = should_retrieve_llm.invoke(

        decide_retrieval_prompt.format_messages(
            question= state['question']
            
        )
    )

    return {'need_retrieval' : decision.should_retrieve}


# Router function
def route_after_relevant(state : State):

    if state.get('need_retrieval'):
        return 'retrieve'
    
    else:
        return 'generate_direct'

    
# -------------------------------------- Direct Generate Node ----------------------------------------------
# form the prompt
direct_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            'Answer the question using only your general knowledge.\n'
            'Do NOT assume access to external documents.\n'
            'If you unsure or the answer require specific sources, say:\n'
            'I do not know based on my general knowledge.'
        ),
        ('human' , '{question}')
    ]
)

# Define node (for direct generate answer using LLM)
def generate_direct(state : State):

    out = llm.invoke(

        direct_generation_prompt.format_messages(question = state['question'])
    )

    return {'answer' : out.content}


# -------------------------------------- Retrieve Node  ----------------------------------------------


def retrieve(state : State):

    question = state.get('retrival_query') or state.get('question')
    return {'docs' : retriever.invoke(question)}


# -------------------------------------- Relevant Checking Decision Node ----------------------------------------------

class RelavenceDecision(BaseModel):

    is_relevant : bool = Field(... , description='True if the documnent helps answer the question , else False')

is_relevant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are judging document relevance at a TOPIC level.\n"
            "Return JSON matching the schema.\n\n"
            "A document is relevant if it discusses the same entity or topic area as the question.\n"
            "It does NOT need to contain the exact answer.\n\n"
            "Examples:\n"
            "- HR policies are relevant to questions about notice period, probation, termination, benefits.\n"
            "- Pricing documents are relevant to questions about refunds, trials, billing terms.\n"
            "- Company profile is relevant to questions about leadership, culture, size, or strategy.\n\n"
            "Do NOT decide whether the document fully answers the question.\n"
            "That will be checked later by IsSUP.\n"
            "When unsure, return is_relevant=true."
        ),
        ("human", "Question:\n{question}\n\nDocument:\n{document}"),
    ]
)

relevance_llm = llm.with_structured_output(RelavenceDecision)

def is_relevant(state : State):

    relevant_doc : List[Document] = []

    for doc in state.get('docs'):

        decision : RelavenceDecision = relevance_llm.invoke(

            is_relevant_prompt.format_messages(
                question = state['question'],
                document = doc.page_content
            )
        )


        if decision.is_relevant:
            relevant_doc.append(doc)


    return {'relevant_docs' : relevant_doc}



# helper function(Routing function) Routing after relevant
def route_after_relevance(state : State) -> Literal['generate_from_context' , 'no_relevant_docs']:

    if state['relevant_docs'] and len(state['relevant_docs']) > 0:
        return 'generate_from_context'
    
    else:
        return 'no_relevant_docs'

# -------------------------------------- Generate from context Node ----------------------------------------------


rag_generate_prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            'You are a business RAG asisstant.\n'
            'Answer the user question using ONLY the provided context.\n'
            'If the context does not contain enough information , say.\n'
            'NO relevant document found.\n'
            'Do not use outsider knowledge.\n'
        ),
        (
            'human',
            'Question : \n{question}\n\n'
            'Context  : \n{context}\n\n'
        )
    ]
)

def generate_from_context(state : State):

    # stuff relevant docs into one block
    context = '\n\n'.join([d.page_content for d in state.get('relevant_docs' , [])]).strip()
    

    if not context:
        return {'answer' : 'No answer found.' , 'context' : ''}
    

    out = llm.invoke(

        rag_generate_prompt.format_messages(
            question = state['question'],
            context = context
        )
    )

    return {'answer' : out.content , 'context' : context}








# -------------------------------------- Get no answer Node ----------------------------------------------
# Define node(when no answer)
def no_answer_found(state : State):
    
    return {'answer' : 'No answer found.' , 'context' : ''}


# -------------------------------------- Answer support Decision ----------------------------------------------
# Define pydantic object for structure output
class IsSUPDecision(BaseModel):

    issup : Literal['fully_support' , 'paritally_support' , 'no_support']
    evidence : List[str] = Field(default_factory=list)


# form the prompt
issup_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are verifying whether the ANSWER is supported by the CONTEXT.\n"
            "Return JSON with keys: issup, evidence.\n"
            "issup must be one of: fully_supported, partially_supported, no_support.\n\n"
            "How to decide issup:\n"
            "- fully_supported:\n"
            "  Every meaningful claim is explicitly supported by CONTEXT, and the ANSWER does NOT introduce\n"
            "  any qualitative/interpretive words that are not present in CONTEXT.\n"
            "  (Examples of disallowed words unless present in CONTEXT: culture, generous, robust, designed to,\n"
            "  supports professional development, best-in-class, employee-first, etc.)\n\n"
            "- partially_supported:\n"
            "  The core facts are supported, BUT the ANSWER includes ANY abstraction, interpretation, or qualitative\n"
            "  phrasing not explicitly stated in CONTEXT (e.g., calling policies 'culture', saying leave is 'generous',\n"
            "  or inferring outcomes like 'supports professional development').\n\n"
            "- no_support:\n"
            "  The key claims are not supported by CONTEXT.\n\n"
            "Rules:\n"
            "- Be strict: if you see ANY unsupported qualitative/interpretive phrasing, choose partially_supported.\n"
            "- If the answer is mostly unrelated to the question or unsupported, choose no_support.\n"
            "- Evidence: include up to 3 short direct quotes from CONTEXT that support the supported parts.\n"
            "- Do not use outside knowledge."
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Answer:\n{answer}\n\n"
            "Context:\n{context}\n"
            
        ),
    ]
)


issup_llm = llm.with_structured_output(IsSUPDecision)

# Define node(whether generate answer support the context and question)
def is_sup(state : State):

    decision : IsSUPDecision = issup_llm.invoke(

        issup_prompt.format_messages(
            question = state['question'],
            answer = state['answer'],
            context = state['context']
        )
    )

    return {'is_sup' : decision.issup , 'evidence' : decision.evidence}



MAX_RETRIES = 10

# Routing function(Routing after is_support document)
def route_after_issup(state : State):

    # accept fi fully support
    if state.get('issup') == 'fully_support':
        return 'accept_answer'
    
    # Stop if we have already tried enough
    if state.get('retries' , 0) >= MAX_RETRIES:
        return 'accept_answer'
    
    # otherwise revise again
    return 'revise_answer'


# -------------------------------------- Accept Decision Node ----------------------------------------------
# Define node(when answer support answer and question we accept and END)
def accept_answer(state : State):
    return {}


# -------------------------------------- Regenerate the answer Node ----------------------------------------------

# form the prompt
revise_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a STRICT reviser.\n\n"
            "You must output based on the following format:\n\n"
            "FORMAT (quote-only answer):\n"
            "- <direct quote from the CONTEXT>\n"
            "- <direct quote from the CONTEXT>\n\n"
            "Rules:\n"
            "- Use ONLY the CONTEXT.\n"
            "- Do NOT add any new words besides bullet dashes and the quotes themselves.\n"
            "- Do NOT explain anything.\n"
            "- Do NOT say 'context', 'not mentioned', 'does not mention', 'not provided', etc.\n"
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Current Answer:\n{answer}\n\n"
            "CONTEXT:\n{context}"
        ),
    ]
)

# Define revise node(if answer is not support then regenerate)
def revise_answer(state : State):

    out = llm.invoke(

        revise_prompt.format_messages(
            question = state['question'],
            answer = state['answer' , ''],
            context = state['context' , '']
        )
    )

    return {

        'answer' : out.content,
        'retries' : state.get('retries' , 0) + 1
    }
# -------------------------------------- Checking useful of answer (Node) ----------------------------------------------

# pydantic object for structure output
class IsUseful(BaseModel):

    is_use : Literal['useful' , 'not_useful']
    reason : str

# form the prompt
isuse_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are judging USEFULNESS of the ANSWER for the QUESTION.\n\n"
            "Goal:\n"
            "- Decide if the answer actually addresses what the user asked.\n\n"
            "Return JSON with keys: isuse, reason.\n"
            "isuse must be one of: useful, not_useful.\n\n"
            "Rules:\n"
            "- useful: The answer directly answers the question or provides the requested specific info.\n"
            "- not_useful: The answer is generic, off-topic, or only gives related background without answering.\n"
            "- Do NOT use outside knowledge.\n"
            "- Do NOT re-check grounding (IsSUP already did that). Only check: 'Did we answer the question?'\n"
            "- Keep reason to 1 short line."
        ),
        (
            "human",
            "Question:\n{question}\n\nAnswer:\n{answer}"
        ),
    ]
)

isuse_llm = llm.with_structured_output(IsUseful)

# Define function(whether generate answer is useful)
def is_use(state : State):

    decision : IsUseful = isuse_llm.invoke(

        isuse_prompt.format(

            question = state.get('question'),
            answer = state.get('answer' , '')
        )
    )

    return {'is_use' : decision.is_use , 'use_reason' : decision.reason}




# Router function(After the Useful checker node)

MAXIMAM_TRIES = 5
def route_after_isuse(state : State):

    if state.get('is_use').lower().strip() == 'useful':
        return 'END'
    
    elif state.get('is_use').lower().strip() == 'not_useful' and state.get('max_tries') > MAXIMAM_TRIES:
        return 'no_answer_found'
    
    else:
        return 'rewrite_query'


# -------------------------------------- ReWrite Decision Node ----------------------------------------------

# pydantic object for structure output
class ReWriteDecision(BaseModel):
    retrieval_query : str = Field(description='ReWritten query optimized for vector retrieval againest internal company PDFs.')


# form the prompt
rewrite_for_relevant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's QUESTION into a query optimized for vector retrieval over INTERNAL company PDFs.\n\n"
            "Rules:\n"
            "- Keep it short (6–16 words).\n"
            "- Preserve key entities (e.g., NexaAI, plan names).\n"
            "- Add 2–5 high-signal keywords that likely appear in policy/pricing docs.\n"
            "- Remove filler words.\n"
            "- Do NOT answer the question.\n"
            "- Output JSON with key: retrieval_query\n\n"
            "Examples:\n"
            "Q: 'Do NexaAI plans include a free trial?'\n"
            "-> {{'retrieval_query': 'NexaAI free trial duration trial period plans'}}\n\n"
            "Q: 'What is NexaAI refund policy?'\n"
            "-> {{'retrieval_query': 'NexaAI refund policy cancellation refund timeline charges'}}"
        ),
        (
            "human",
            "QUESTION:\n{question}\n\n"
            "Previous retrieval query:\n{retrieval_query}\n\n"
            "Answer (if any):\n{answer}"
        ),
    ]
)

rewrite_llm = llm.with_structured_output(ReWriteDecision)


# Define function(Rewrite the question and repeat the process)
def rewrite_query(state : State):

    decision : ReWriteDecision = rewrite_llm.invoke(

        rewrite_for_relevant_prompt.from_messages(
            question = state.get('question'),
            retrieval_query = state.get('retrieval_query' , ''),
            answer = state.get('answer')
        )
    )

    return {

        'retrieval_query' : decision.retrieval_query,
        'rewrite_tries' : state.get('rewrite_tries' , 0) + 1,
        
        # optional : reset the state values so that next tries state is clean
        'docs' : [],
        'relevant_docs' : [],
        'context' : ''
    }

# -------------------------------------- Form the builder ----------------------------------------------


builder = StateGraph(State)



# -------------------------------------- Add the nodes ----------------------------------------------
builder.add_node('decide_retrieval' , decide_retrieval)
builder.add_node('generate_direct' , generate_direct)
builder.add_node('retrieve' , retrieve)

builder.add_node('is_relevant' , is_relevant)
builder.add_node('generate_from_context' , generate_from_context)
builder.add_node('no_answer_found' , no_answer_found)

builder.add_node('is_sup' , is_sup)
builder.add_node('revise_answer' , revise_answer)

builder.add_node('is_use' , is_use)
builder.add_node('rewrite_query' , rewrite_query)
builder.add_node('accept_answer' , accept_answer)


# -------------------------------------- Add the edges ----------------------------------------------

builder.add_edge(START , 'decide_retrieval')

builder.add_conditional_edges(
    'decide_retrieval',

    route_after_relevant,
    {
        'generate_direct' : 'generate_direct',
        'retrieve' : 'retrieve'
    }
)


builder.add_edge('generate_direct' , END)

builder.add_edge('retrieve' , 'is_relevant')

builder.add_conditional_edges(
    'is_relevant',

    route_after_relevance,
    {
        'no_answer_found' : 'no_answer_found',
        'generate_from_context' : 'generate_from_context'
    }
)
builder.add_edge('generate_from_context' , 'is_sup')

builder.add_conditional_edges(
    'is_sup',
    route_after_issup,
    {
        'accept_answer' : 'accept_answer',
        'revise_answer' : 'revise_answer'
    }
)

builder.add_edge('revise_answer' , 'is_sup')
builder.add_edge('accept_answer' , 'is_use')


builder.add_conditional_edges(
    'is_use',
    route_after_isuse,
    {
        'rewrite_query' : 'rewrite_query',
        'no_answer_found' : 'no_answer_found',
        'END' : END
    }
)

builder.add_edge('rewrite_query' , 'retrieve')

builder.add_edge('no_answer_found' , END)

# -------------------------------------- Compile ----------------------------------------------


# compile the graph
graph = builder.compile()






