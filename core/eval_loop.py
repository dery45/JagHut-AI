import re
from typing import Optional
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ─── Eval LLM (Qwen2.5-1.5B) ───────────────────────────────────────────────────
_eval_model = OllamaLLM(model="qwen2.5:1.5b", temperature=0, num_ctx=4096)

_FAITHFULNESS_PROMPT = ChatPromptTemplate.from_template("""
You are evaluating an AI answer for faithfulness.
Faithfulness = does the answer only use information present in the context below?

Context (retrieved documents):
{context}

Answer to evaluate:
{answer}

Rate faithfulness as exactly one word: HIGH, MEDIUM, or LOW.
- HIGH: answer is fully supported by context
- MEDIUM: answer is mostly supported, minor additions
- LOW: answer contains significant information not in context (hallucination)

Rating:""")

_RELEVANCE_PROMPT = ChatPromptTemplate.from_template("""
You are evaluating retrieved documents for relevance.
Question asked: {question}

Retrieved document snippets:
{snippets}

Rate relevance as exactly one word: HIGH, MEDIUM, or LOW.
- HIGH: documents directly answer the question
- MEDIUM: documents are related but not directly answering
- LOW: documents are mostly off-topic

Rating:""")

_faith_chain = _FAITHFULNESS_PROMPT | _eval_model | StrOutputParser()
_rel_chain   = _RELEVANCE_PROMPT    | _eval_model | StrOutputParser()


def _parse_rating(raw: str) -> str:
    """Extract HIGH/MEDIUM/LOW from model output."""
    clean = raw.strip().upper()
    for level in ("HIGH", "MEDIUM", "LOW"):
        if level in clean:
            return level
    return "UNKNOWN"


# ─── Heuristik lexical (fallback cepat) ──────────────────────────────────────

def _lexical_faithfulness(answer: str, context: str) -> Optional[str]:
    """
    Heuristik cepat: cek overlap kata penting antara jawaban dan konteks.
    Return HIGH/LOW/None (None = tidak konklusif, perlu LLM eval).
    """
    if not context or not answer:
        return "UNKNOWN"

    # Kata-kata yang sering muncul kalau model halusinasi tanpa konteks
    hallucination_phrases = [
        "maaf, saya tidak memiliki",
        "saya tidak tahu",
        "tidak ada informasi",
        "i don't have",
        "i cannot find",
    ]
    answer_lower = answer.lower()
    for phrase in hallucination_phrases:
        if phrase in answer_lower:
            return "LOW"   # model sendiri bilang tidak ada data → retrieval buruk

    # Cek overlap token antara jawaban dan konteks
    def tokens(text):
        return set(re.findall(r'\b\w{4,}\b', text.lower()))

    ans_tokens = tokens(answer)
    ctx_tokens = tokens(context)
    if not ans_tokens:
        return None

    overlap = len(ans_tokens & ctx_tokens) / len(ans_tokens)
    if overlap >= 0.40:
        return "HIGH"
    if overlap >= 0.20:
        return None   # tidak konklusif → pakai LLM
    return "LOW"


def _lexical_relevance(question: str, docs: list) -> Optional[str]:
    """
    Heuristik cepat untuk relevansi: cek apakah keyword dari pertanyaan
    muncul di dokumen yang diambil.
    """
    if not docs:
        return "LOW"

    q_tokens = set(re.findall(r'\b\w{4,}\b', question.lower()))
    if not q_tokens:
        return None

    combined = " ".join(d.page_content for d in docs).lower()
    d_tokens = set(re.findall(r'\b\w{4,}\b', combined))

    overlap = len(q_tokens & d_tokens) / len(q_tokens)
    if overlap >= 0.50:
        return "HIGH"
    if overlap >= 0.25:
        return None   # tidak konklusif
    return "LOW"


# ─── Main eval function ───────────────────────────────────────────────────────

def evaluate(
    question:  str,
    docs:      list,
    context:   str,
    answer:    str,
    use_llm:   bool = True,
) -> dict:
    """
    Evaluasi satu query. Dipanggil dari main.py setelah jawaban digenerate.

    Args:
        question  : pertanyaan asli user
        docs      : list LangChain Document yang dipakai
        context   : string konteks yang dikirim ke LLM
        answer    : jawaban yang dihasilkan LLM
        use_llm   : pakai LLM eval kalau heuristik tidak konklusif (default True)

    Returns:
        dict dengan keys: faithfulness, relevance, flag, reason, method
    """
    faith  = _lexical_faithfulness(answer, context)
    rel    = _lexical_relevance(question, docs)
    method = "lexical"

    # Kalau heuristik tidak konklusif dan LLM eval diaktifkan
    if use_llm and (faith is None or rel is None):
        method = "llm"
        snippets = "\n---\n".join(
            f"[{i+1}] {d.page_content[:200]}" for i, d in enumerate(docs[:5])
        )
        try:
            if faith is None:
                raw   = _faith_chain.invoke({"context": context[:2000], "answer": answer[:500]})
                faith = _parse_rating(raw)
            if rel is None:
                raw = _rel_chain.invoke({"question": question, "snippets": snippets})
                rel = _parse_rating(raw)
        except Exception as e:
            print(f"   ⚠️  Eval LLM error: {e}")
            faith = faith or "UNKNOWN"
            rel   = rel   or "UNKNOWN"

    # Fallback kalau masih None
    faith = faith or "UNKNOWN"
    rel   = rel   or "UNKNOWN"

    #     Flag jika ada yang LOW
    flagged = (faith == "LOW" or rel == "LOW")
    reason  = ""
    if faith == "LOW":
        reason += "faithfulness LOW (possible hallucination); "
    if rel == "LOW":
        reason += "relevance LOW (retrieval miss); "
    if not docs:
        reason += "no documents retrieved; "
        flagged = True

    return {
        "faithfulness": faith,
        "relevance":    rel,
        "flag":         flagged,
        "reason":       reason.strip(),
        "method":       method,
        "doc_count":    len(docs),
    }