from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.retriever import get_retriever

SYSTEM = """You are a document assistant. Answer using ONLY the numbered context passages below.
Cite inline with [1], [2], etc. If multiple passages support a claim, cite all: [1][3].
If the answer isn't in the context, say: "I couldn't find that in the uploaded documents."
Do not use outside knowledge.

{context}"""


def _llm():
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=settings.llm_model, openai_api_key=settings.openai_api_key, temperature=0)
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(
        model=settings.llm_model,
        anthropic_api_key=settings.anthropic_api_key,
        temperature=0,
        max_tokens=1024,
    )


def _fmt_context(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        parts.append(f"[{i}] ({m.get('source')} p.{m.get('page')})\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


@dataclass
class Response:
    answer: str
    sources: list[dict] = field(default_factory=list)
    query: str = ""

    @classmethod
    def from_docs(cls, answer: str, docs: list[Document], query: str) -> "Response":
        return cls(
            answer=answer,
            sources=[
                {
                    "index": i + 1,
                    "source": d.metadata.get("source", ""),
                    "page": d.metadata.get("page"),
                    "total_pages": d.metadata.get("total_pages"),
                    "excerpt": d.page_content[:300] + ("..." if len(d.page_content) > 300 else ""),
                }
                for i, d in enumerate(docs)
            ],
            query=query,
        )


def ask(question: str, collection: str = "default") -> Response:
    retriever = get_retriever(collection)
    docs = retriever.invoke(question)

    if not docs:
        return Response(answer="I couldn't find that in the uploaded documents.", query=question)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        ("human", "{question}"),
    ])
    chain = prompt | _llm() | StrOutputParser()
    answer = chain.invoke({"context": _fmt_context(docs), "question": question})

    return Response.from_docs(answer=answer, docs=docs, query=question)
