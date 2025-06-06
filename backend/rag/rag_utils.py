
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.anthropic import Anthropic

from llama_index.core import Document, Settings
from llama_index.llms.gemini import Gemini

from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.query_engine import CitationQueryEngine, SubQuestionQueryEngine

import json
from typing import Union, List, Dict
import os
static_index = None # Set it at the end of the file
Settings.llm = None

openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

def build_index_core(
    data_input,
    chunk_size: int = 8192,
    embedding_model: str = "text-embedding-ada-002",
    llm_model: str = None,
    temperature: float = 0.0,
    show_progress: bool = True,
) -> VectorStoreIndex:
    """
    Build a VectorStoreIndex from paper metadata containing abstracts.

    Args:
        data_input: list of documents
        chunk_size (int): Chunk size for document splitting.
        embedding_model (str): Embedding model to use.
        llm_model (str): LLM model to use.
        temperature (float): Temperature setting for the LLM.
        show_progress (bool): Whether to show progress during index construction.

    Returns:
        VectorStoreIndex: The constructed vector store index.
    """
    # Configure LlamaIndex settings
    if llm_model == "gpt-4o-mini":
        Settings.llm = OpenAI(model=llm_model, temperature=temperature, max_tokens=16384, api_key=openai_api_key)
        Settings.chunk_size = chunk_size
        Settings.embed_model = OpenAIEmbedding(model=embedding_model, api_key=openai_api_key)
    if llm_model == "claude-3-7-sonnet-latest":
        Settings.llm = Anthropic(model=llm_model, temperature=temperature, max_tokens=16384, api_key=anthropic_api_key)
        Settings.chunk_size = chunk_size
        Settings.embed_model = OpenAIEmbedding(model=embedding_model, max_tokens=16384, api_key=openai_api_key)
    if llm_model == "gemini-2.0-flash":
        Settings.llm = Gemini(model="models/gemini-2.0-flash", api_key=google_api_key)
        Settings.chunk_size = chunk_size
        Settings.embed_model = GoogleGenAIEmbedding(model_name="text-embedding-004", api_key=google_api_key)

    print(len(data_input))
    # Build and return the vector index
    index = VectorStoreIndex([])

    all_documents = data_input  # Replace with your list of Document objects

    # Initialize the text splitter
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=200)

    batch_size = 20
    for i in range(0, len(all_documents), batch_size):
        batch = all_documents[i:i + batch_size]

        # Split documents into chunks
        split_documents = []
        for doc in batch:
            chunks = text_splitter.split_text(doc.text)
            for chunk in chunks:
                split_documents.append(Document(text=chunk, metadata=doc.metadata))

        # Insert nodes into the index
        index.insert_nodes(split_documents)
    return index

def build_index(
    data_input: Union[str, List[Dict]],
    chunk_size: int = 8192,
    embedding_model: str = "text-embedding-ada-002",
    llm_model: str = None,
    temperature: float = 0.0,
    show_progress: bool = True
) -> VectorStoreIndex:
    """
    Build a VectorStoreIndex from paper metadata containing abstracts.

    Args:
        data_input (str or List[Dict]): Path to a JSON file or a list of dictionaries containing paper metadata.
        chunk_size (int): Chunk size for document splitting.
        embedding_model (str): Embedding model to use.
        llm_model (str): LLM model to use.
        temperature (float): Temperature setting for the LLM.
        show_progress (bool): Whether to show progress during index construction.

    Returns:
        VectorStoreIndex: The constructed vector store index.
    """
    # Load data from JSON file or use provided list
    if isinstance(data_input, str):
        with open(data_input, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = data_input

    documents = []
    for entry in data:
        abstract = entry.get("abstract", "")
        title = entry.get("title", "")
        if abstract and title and len(abstract.strip()) > 20:
            metadata = {
                "title": title,
                "publication_year": entry.get("publication_year", ""),
                "authors": entry.get("authors", []),
                "doi": entry.get("DOI", ""),
                "affiliations": entry.get("affiliations", []),
                "affiliations_string": ", ".join(entry.get("affiliations", [])),
                "countries": entry.get("countries", []),
                "countries_string": ", ".join(filter(None, entry.get("countries", [])))
            }
            documents.append(Document(text=abstract, metadata=metadata))

    if llm_model == "gpt-4o-mini":
        Settings.llm = OpenAI(model=llm_model, temperature=temperature, max_tokens=16384, api_key=openai_api_key)
        Settings.chunk_size = chunk_size
        Settings.embed_model = OpenAIEmbedding(model=embedding_model, api_key=openai_api_key)
    if llm_model == "claude-3-7-sonnet-latest":
        Settings.llm = Anthropic(model=llm_model, temperature=temperature, max_tokens=16384, api_key=anthropic_api_key)
        Settings.chunk_size = chunk_size
        Settings.embed_model = OpenAIEmbedding(model=embedding_model, max_tokens=16384, api_key=openai_api_key)
    if llm_model == "gemini-2.0-flash":
        Settings.llm = Gemini(model="models/gemini-2.0-flash", api_key=google_api_key)
        Settings.chunk_size = chunk_size
        Settings.embed_model = GoogleGenAIEmbedding(model_name="text-embedding-004", api_key=google_api_key)


    # Build and return the vector index
    index = VectorStoreIndex.from_documents(documents, show_progress=show_progress)
    return index


def paper_answer(prompt, paper):
    context = f"{paper.get('title', '')}:{paper.get('abstract', '')} \n\n"
    context = context + f"Full Text:\n{paper.get('fulltext', '')}\n\n"

    answer_prompt = (
        f"{context}"
        f"Given the context above, provide a detailed answer to the posted question: {prompt}  "
        "Do not use any additional knowledge, and only use the information provided above"
        "If you cannot answer the question, state the limiting factors"
    )

    # Generate comparison paragraph with citations
    answer_response = Settings.llm.complete(answer_prompt).text

    print("\n--- Answer Response ---\n")
    print(answer_response.strip())
    return answer_response


def institutional_summary(summaries, institution, prompt=None):
    # summaries = data.get("institutional_summary", [])
    print(institution)
    if not summaries:
        return  # Skip if no summaries

    # Step 1: Build comparison context
    comparison_context = ""
    for s in summaries:
        comparison_context += f"[{s['index']}] {s['summary']}\n\n"

    # Step 2: Build references
    references = "\n".join(
        f"[{institution}_{s['index']}] {s['title']} by {s['authors']}. DOI: https://doi.org/{s['doi']}" if s.get('doi')
        else f"[{institution}_{s['index']}] {s['title']} by {s['authors']} (DOI unavailable)"
        for s in summaries
    )
    if prompt:
        institutional_summary = (
            f"{comparison_context}"
            f"Write an overview of the contributions of {institution}, based on the given works above and the given user question:"
            f"User question: {prompt}"            
            "Do not use any additional knowledge, and only use the information provided above. "
            f"Use inline citations like [{institution}_1], [{institution}_2], etc., to refer to them. "
            f"After the paragraph, list the full references you used with the same inline citation style [{institution}_1], authors and DOIs.\n\n"
            f"{references}"
        )
    else:
        institutional_summary = (
            f"{comparison_context}"
            f"Write an overview of the contributions of {institution}, based on the given works above. "
            "Do not use any additional knowledge, and only use the information provided above. "
            f"Use inline citations like [{institution}_1], [{institution}_2], etc., to refer to them. "
            f"After the paragraph, list the full references you used with the same inline citation style [{institution}_1], authors and DOIs.\n\n"
            f"{references}"
        )

    # Step 4: Get LLM response
    institutional_response = Settings.llm.complete(institutional_summary).text.strip()

    # fixing_prompt = (
    #    f"{institutional_response}"
    #    "Given the response above, please remove all unused references in the end"
    #    "Do not change the text in any other way, and do not rename references"
    # )
    # fixing_response = Settings.llm.complete(fixing_prompt).text
    # print(fixing_response)

    # Step 5: Store response
    return institutional_response


def institutional_summary_structured(structured_output):
    for institution, data in structured_output.items():
        nodes_with_scores = data.get("top_documents", [])

        # Convert NodeWithScore(TextNode) to dict format expected by institutional_summary()
        summaries = []
        for i, nws in enumerate(nodes_with_scores):
            element = {
                "index": i + 1,
                "title": nws.node.metadata.get("title", ""),
                "authors": nws.node.metadata.get("authors", []),
                "publication_year": nws.node.metadata.get("publication_year", []),
                "doi": nws.node.metadata.get("doi", ""),
                "summary": nws.node.text
            }
            summaries.append(element)

        data["institutional_summary"] = institutional_summary(summaries, institution)

    return structured_output

def compare_universities(top_universities, single_university, input_prompt):
    """
    Compare the research contributions of a single university against a set of top universities
    using only the provided institutional summaries.

    Args:
        top_universities (dict): Dictionary with university names as keys and a 'institutional_summary' field in values.
        single_university (dict): A single university's data, just the textual response.
        input_prompt (str): Optional guiding prompt to focus the comparison.

    Returns:
        str: A formatted and citation-corrected comparative analysis.
    """
    comparison_context = "# Summaries of Top Research Institutions:\n"
    for idx, (institution, data) in enumerate(top_universities.items(), start=1):
        comparison_context += f"[{idx}] {data['institutional_summary']}\n\n"

    comparison_context += "# Summary of the Institution to Compare:\n"
    comparison_context += f"[{len(top_universities) + 1}] {single_university}\n\n"

    comparison_prompt = (
        f"{comparison_context}"
        "Based solely on the summaries above, write a comparative analysis discussing how the institution in the second section "
        "compares to the group of institutions in the first section. "
        f"Focus on research strengths, uniqueness, or overlaps as relevant, so you answer the given input question: {input_prompt.strip()} "
        f"Use inline citations like [Harvard_1], [Cambridge_2], etc., to refer to them, similar as they have been used in the context. "
        "Do not use any outside information—only use what is stated above. "
        "After the comparison paragraph, include a list of references matching the inline citation numbers (e.g. Harvard_1, with institution names, authors, and DOIs if available.\n\n"
    )

    comparison_response = Settings.llm.complete(comparison_prompt).text

    return fix_output_citations(comparison_response)


def summarize_universities(structured_institution):
    for institution, data in structured_institution.items():
        top_docs = data["top_documents"]
        summaries = summarize(top_docs)
        data["summary"] = summaries
    return structured_institution
def summarize_university(docs):
    return summarize(docs)

def summarize(documents):
    summaries = []
    for i, node in enumerate(documents, start=1):
        content = node.get_content() if hasattr(node, "get_content") else getattr(node, "text", str(node))
        meta = getattr(node, "metadata", {})
        title = meta.get("title", f"Document {i}")
        abstract = meta.get("abstract", "")
        authors = meta.get("authors", "Unknown authors")
        doi = meta.get("doi", meta.get("DOI", ""))
        publication_year = meta.get("publication_year", meta.get("publication_year", ""))
        prompt = (
            f"Paper Title: {title}\n"
            f"Authors: {authors}\n"
            f"Abstract:\n{abstract}\n\nFull Text:\n{content}\n\n"
            "Summarize the key contributions and findings of the paper above, using only the provided content."
        )
        summary_response = Settings.llm.complete(prompt).text
        summaries.append({
            "index": i,
            "title": title,
            "authors": authors,
            "year": publication_year,
            "doi": doi,
            "summary": summary_response.strip()
        })
    return summaries


from collections import defaultdict, Counter

def extract_top_universities(docs, query, exclude=None, current_year=2025, top_k_institutions=3, top_docs_per_inst=15, country_code = None):
    """
    Analyze retrieved documents and extract top contributing universities.

    Args:
        docs (list): List of retrieved document objects, each with .text and .metadata
        query (str): The comparison query
        current_year (int): The current year for filtering
        top_k_institutions (int): Number of top institutions to select
        top_docs_per_inst (int): Number of top documents per institution

    Returns:
        dict: Structured output with top institutions, publication counts, and top documents
    """
    cutoff_year = current_year - 5

    # Step 1: Filter relevant documents

    filtered_docs = docs #[doc for doc in docs if is_relevant(doc)]

    # Step 2: Count publications per institution
    institution_counter = Counter()
    inst_to_docs = defaultdict(list)

    exclude_set = set([exclude]) if isinstance(exclude, str) else set(exclude or [])
    print(docs[0].metadata)
    for doc in filtered_docs:
        institutions = doc.metadata.get("affiliations", [])
        for inst in institutions:
            inst_s = inst.strip()
            if inst_s not in exclude_set:
                institution_counter[inst_s] += 1
                inst_to_docs[inst_s].append(doc)
    top_institutions = [inst for inst, _ in institution_counter.most_common(top_k_institutions)]
    print(top_institutions)

    # Step 3: Score and rank documents per institution
    embed_model = OpenAIEmbedding()
    query_embedding = embed_model.get_query_embedding(query)

    def score_document(doc, query_emb):
        doc_emb = embed_model.get_text_embedding(doc.text)
        return sum(a * b for a, b in zip(doc_emb, query_emb))  # dot product

    structured_output = {}
    for inst in top_institutions:
        inst_docs = inst_to_docs[inst]
        scored_docs = [(doc, score_document(doc, query_embedding)) for doc in inst_docs]
        top_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)[:top_docs_per_inst]

        structured_output[inst] = {
            "num_publications": len(inst_docs),
            "top_documents": [doc for doc, _ in top_docs],
        }

    return structured_output


def overview(summaries, prompt):
    comparison_context = ""
    for s in summaries:
        comparison_context += f"[{s['index']}] {s['summary']}\n\n"

    references = "\n".join(
        f"[{s['index']}] {s['title']} by {s['authors']}. DOI: https://doi.org/{s['doi']}" if s['doi']
        else f"[{s['index']}] {s['title']} by {s['authors']} (DOI unavailable)"
        for s in summaries
    )

    overview_prompt = (
        f"{comparison_context}"
        "Write a detailed overview using the related works above, to answer the following input question with as much detail as possible: "
        f"{prompt} "
        "##Additional instructions: "
        "- Do not use any additional knowledge, and only use the information provided above"
        "- Use inline citations like [1], [2], etc., to refer to them. "
        "- After the paragraph, list the full references you used with numbers and DOIs.\n\n"
        f"{references}"
    )

    # Generate comparison paragraph with citations
    overview_response = Settings.llm.complete(overview_prompt).text

    return fix_output_citations(overview_response)

# --- 6. Compare new paper with inline [n] references and numbered list ---
# new_title = new_papers[0].get("title", "")
# new_authors = new_papers[0].get("authors", "")
# new_abstract = new_papers[0].get("abstract", "")
def compare_papers(summaries, prompt, paper):
    # Compose context + citation references
    comparison_context = f"New Paper Title: {paper.get('title', '')}\nAuthors: {paper.get('authors', '')}\nAbstract: {paper.get('abstract', '')}\n\n"
    comparison_context = comparison_context + f"Full Text:\n{paper.get('fulltext', '')}\n\n"
    comparison_context += "# Summaries of Related Works:\n"
    for s in summaries:
        comparison_context += f"[{s['index']}] {s['summary']}\n\n"

    references = "\n".join(
        f"[{s['index']}] {s['title']} by {s['authors']}. DOI: https://doi.org/{s['doi']}" if s['doi']
        else f"[{s['index']}] {s['title']} by {s['authors']} (DOI unavailable)"
        for s in summaries
    )

    comparison_prompt = (
        f"{comparison_context}"
        "Given the new paper above, in comparison with the related works, please answer the following question in as much detail as possible: "
        f"{prompt} "
        "Do not use any additional knowledge, and only use the information provided above"
        "Use inline citations like [1], [2], etc., to refer to them. "
        "After the paragraph, list the full references you used with numbers and DOIs.\n\n"
        f"{references}"
    )

    # Generate comparison paragraph with citations
    comparison_response = Settings.llm.complete(comparison_prompt).text


    return fix_output_citations(comparison_response)


from datetime import datetime
from copy import deepcopy
import re


def generate_topic_queries(info_dict):
    topic = info_dict.get("topic")
    if not topic:
        return info_dict  # No topic to generate queries for

    prompt = f"""
You are a research assistant tasked with retrieving academic literature.

Given the topic: "{topic}", generate a list of 4 sub-topics, that help to retrieve relevant literature to answer the question.

The topics should be suitable for guiding a literature search.

Return only the list of topics, without numbering or bullets.
"""

    output = Settings.llm.complete(prompt).text.strip()

    # Clean each line to remove numbers, dashes, etc.
    lines = output.splitlines()
    questions = []
    for line in lines:
        clean_line = re.sub(r"^\s*(?:\d+[\).]?|\-|\*)\s*", "", line)  # remove leading "1. ", "- ", etc.
        if clean_line:
            questions.append(clean_line.strip())

    updated_dict = deepcopy(info_dict)
    updated_dict["queries"] = questions
    return updated_dict


def generate_questions(prompt, paper=None, llm_model = None, chunk_size: int = 8192, temperature = 0,embedding_model: str = "text-embedding-ada-002",):
    if llm_model == "gpt-4o-mini":
        Settings.llm = OpenAI(model=llm_model, temperature=temperature, max_tokens=16384)
        Settings.chunk_size = chunk_size
        Settings.embed_model = OpenAIEmbedding(model=embedding_model)
    if llm_model == "claude-3-7-sonnet-latest":
        Settings.llm = Anthropic(model=llm_model, temperature=temperature, max_tokens=16384)
        Settings.chunk_size = chunk_size
        Settings.embed_model = OpenAIEmbedding(model=embedding_model)
    if llm_model == "gemini-2.0-flash":
        Settings.llm = Gemini(model="models/gemini-2.0-flash", max_tokens=16384)
        Settings.chunk_size = chunk_size
        Settings.embed_model = GoogleGenAIEmbedding(model_name="text-embedding-004")

    current_year = datetime.now().year
    if paper:
        input_prompt = prompt +"\n" + paper.get("title", "") + paper.get("abstract", "")
    else:
        input_prompt = prompt
    generation_prompt = (
        "You are a scientific researcher tasked with analyzing and comparing scientific literature. "
        "You will be given a question posed by fellow researchers. "
        "Your task is to determine whether additional scientific documents are needed to answer the question meaningfully.\n\n"

        "Respond with a object which has the following structure:\n"
        "{\n"
        '  "type": "<comparison|university|paper|overview|university_university|single_university>",\n'
        '  "university_name": "<university name or null>",\n'
        '  "comparison_university_name": "<university name or null>",\n'
        '  "country": "<country code or null>",\n'
        f'  "year_start": <integer or null>,  // e.g., {current_year - 5}\n'
        f'  "year_end": <integer or null>,    // e.g., {current_year}\n'
        '   "topic": <topic of interest>, // e.g. Topic Modeling, Reinforcement Learning, Language Generation'
        "}\n\n"

        "**Explanation of the 'type' field:**\n"
        "- Use 'university' if the goal is to compare research contributions between institutions (e.g., 'What has the University of Aberdeen contributed compared to others?').\n"
        "- Use 'university_university' if the goal is to compare research contributions between exactly two institutions (e.g., 'What has the University of Aberdeen contributed compared to the University of Edinburgh?').\n"
        "- Use 'single_university' if the goal is to state the contributions of exactly one institution (e.g., 'What has the University of Aberdeen contributed to Information Retrieval over the last 5 years?').\n"
        "- Use 'comparison' if the goal is to compare the content or impact of a specific paper to other scientific work (e.g., 'How does the paper compare to other relevant literature? + a title and an abstract'\n\n"
        "- Use 'paper' if the goal is to answer a question about the content of a specific paper.\n\n"
        "- Use 'overview' if the goal is to compare the content of scientific work, without any input paper"

        "**Explanation of the 'university_name' field:**\n"
        "- Use the name of the university or the research institution, if it is given in input question, otherwise return null.\n"

        "**Explanation of the 'comparison_university_name' field:**\n"
        "- Use this field if more than one name is given, insert the name of the university or the research institution, otherwise return null.\n"

        "**Explanation of the 'country' field:**\n"
        "- Use this field if a university type is chosen, and the comparison should be limited to a specific country, as specified in the query.\n"
        "- Insert the country code for the country which is given in the query (e.g., US, GB, AT) otherwise return null.\n"
        "- Set this field to null, if the question does not specify a country.\n"

        "**Explanation of the 'year_start' and 'year_end' fields:**\n"
        "- Add here the years based on which the query should be filtered, if given in the input query, otherwise return null.\n"
        f"- If the question contains phrases as 'last 5 years', do calculations based on {current_year}"

        "**Explanation of the 'topic' field:**\n"
        "- Use this field to specify the topic under interest, e.g. Topic Modeling, Reinforcement Learning, Language Generation.\n"

        f"Here is the input question:\n{input_prompt}"
    )
    output = Settings.llm.complete(generation_prompt).text.strip()
    output = output.replace("```json", "")
    output = output.replace("```", "")
    print(output)
    info = json.loads(output)
    info_with_queries = generate_topic_queries(info)
    print(info_with_queries)
    return info_with_queries


def summarize(documents):
    summaries = []
    for i, node in enumerate(documents, start=1):
        content = node.get_content() if hasattr(node, "get_content") else getattr(node, "text", str(node))
        meta = getattr(node, "metadata", {})
        title = meta.get("title", f"Document {i}")
        abstract = meta.get("abstract", "")
        authors = meta.get("authors", "Unknown authors")
        doi = meta.get("doi", meta.get("DOI", ""))
        publication_year = meta.get("publication_year", meta.get("publication_year", ""))
        prompt = (
            f"Paper Title: {title}\n"
            f"Authors: {authors}\n"
            f"Abstract:\n{abstract}\n\nFull Text:\n{content}\n\n"
            "Summarize the key contributions and findings of the paper above, using only the provided content."
        )

        summary_response = Settings.llm.complete(prompt).text
        summaries.append({
            "index": i,
            "title": title,
            "authors": authors,
            "year": publication_year,
            "doi": doi,
            "summary": summary_response.strip()
        })
    return summaries



from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
# --- 4. Retrieve top-K relevant documents ---
def retrieve(query, exclude=None, top_k=5, institution=None, year_start=None, year_end=None, country=None, index=static_index):
    """
    Retrieve documents for a query, optionally filtering by DOI or affiliation.

    Args:
        query (str): The search query.
        exclude (str, optional): DOI to exclude from results.
        top_k (int): Number of top results to retrieve.
        affiliation (str, optional): Affiliation substring to filter (e.g., "TU Wien").

    Returns:
        List of deduplicated retrieved nodes.
    """
    filters = []

    if exclude:
        filters.append(MetadataFilter(key="doi", value=exclude, operator="!="))

    if institution:
        filters.append(MetadataFilter(key="affiliations_string", value=institution, operator="contains"))

    if year_start:
        filters.append(MetadataFilter(key="publication_year", value=year_start, operator=">="))

    if year_end:
        filters.append(MetadataFilter(key="publication_year", value=year_end, operator="<="))

    if country:
        filters.append(MetadataFilter(key="countries_string", value=country, operator="contains"))

    metadata_filters = MetadataFilters(filters=filters) if filters else None

    # Get retriever with optional metadata filters
    retriever = index.as_retriever(similarity_top_k=top_k, filters=metadata_filters)

    # LLM-enhanced query engine (used internally, but not needed for plain retrieval here)
    citation_engine = CitationQueryEngine(retriever=retriever, llm=Settings.llm)

    retrieved_nodes = retriever.retrieve(query)

    #for i, item in enumerate(retrieved_nodes):
    #    print("retrieved_node:", i, " title: ", item.metadata.get('title'))

    # Deduplicate by DOI or title
    unique = {}
    for node in retrieved_nodes:
        meta = getattr(node, "metadata", {})
        key = meta.get("doi") or meta.get("title")
        if key and key not in unique:
            unique[key] = node

    return list(unique.values())

def compare_two_universities(university, compare_university, prompt):
    """
    Compare the research contributions of two specific universities using only their institutional summaries.

    Args:
        university (dict): Dictionary with keys 'name' and 'institutional_summary'.
        compare_university (dict): Dictionary with keys 'name' and 'institutional_summary'.
        prompt (str): Optional guiding prompt to focus the comparison.

    Returns:
        str: A formatted and citation-corrected comparative analysis.
    """
    context = "# Summary of the First Institution:\n"
    context += f"{university}\n\n"

    context += "# Summary of the Second Institution:\n"
    context += f"{compare_university}\n\n"

    comparison_prompt = (
        f"{context}"
        "Based solely on the summaries above, write a comparative analysis discussing how the second institution "
        "compares to the first one. Focus on research strengths, uniqueness, or overlaps as relevant. "
        "Especially try to answer the following information need given by the user:"
        f"{prompt.strip()} "
        f"Use inline citations like [Harvard_1], [Cambridge_2], etc., to refer to them, similar as they have been used in the context. "
        "Do not use any outside information—only use what is stated above. "
        "After the comparison paragraph, include a list of references matching the inline citation numbers (e.g. Harvard_1, with institution names, authors, and DOIs if available.\n\n"
    )
    print(Settings.llm)
    comparison_response = Settings.llm.complete(comparison_prompt).text

    return fix_output_citations(comparison_response)

def fix_output_citations(input):
    fixing_prompt = (
        f"{input}\n\n"
        "Please correct the inline citation numbers so that:\n"
        "- The numbering is continuous and starts from [1]\n"
        "- All unused references are removed\n"
        "- Identical or duplicate references are deduplicated and referred to by a single number\n"
        "- If a reference belongs to two institutions (so it is duplicated), rewrite the sentence to make it clear that this was a collaboration between multiple institutions\n"
        "- The references at the end follow the same numbering format ([1], [2], etc.) and correspond correctly to their inline use\n"
        "- For the references in the end follow APA style, e.g.:\n"
        "  [1] Smith, J., & Doe, A. (2020). Title of the paper. https://doi.org/10.1234/example\n"
        "- Do adapt author lists, and replace authors after the third author with 'et al.'\n"
        "- Do not change any of the main comparison text other than adjusting the citation numbers, institution naming, and updating the reference list accordingly."
        "- If not done before, add a line break between every different citation in the reference section to improve readability\n"
    )
    fixing_response = Settings.llm.complete(fixing_prompt).text
    return fixing_response

#static_index = build_index("data/works.json")
#retriever = static_index.as_retriever(similarity_top_k=5)
#citation_engine = CitationQueryEngine(retriever=retriever, llm=Settings.llm)