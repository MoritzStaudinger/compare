import json

from backend.core.reader import CoreReader
from backend.openalex.utils import get_most_relevant_institutions_for_queries, get_ror_id, search_paper, \
    get_institution_profile, get_university_name_from_ror
from backend.rag.rag_utils import build_index, generate_questions, retrieve, extract_top_universities, build_index_core, \
    summarize_university, institutional_summary, summarize_universities, institutional_summary_structured, \
    compare_universities, compare_papers, overview, paper_answer, summarize, compare_two_universities, \
    fix_output_citations, use_abstracts_universities, use_abstracts_university
from backend.utils.utils import remove_duplicates_by_doi, remove_duplicates_by_doi_dict

def stream_fn(msg: str):
    payload = {"update": msg}
    return [f"data: {json.dumps(payload)}\n\n".encode("utf-8")]

def handle_single_university_wrapper(*args, stream=False, **kwargs):
    if stream:
        return handle_single_university(*args, **kwargs)
    else:
        # Consume the generator and return the final result
        result = b""
        for chunk in handle_single_university(*args, **kwargs):
            result += chunk
        return result


def handle_single_university(data, prompt, online, model, top_k=15, stream_fn=None):
    def emit(msg):
        if stream_fn:
            yield from stream_fn(msg)

    queries = data["queries"]
    profiles = []
    all_retrieved_docs = []

    # 1. Data collection and indexing
    if online:
        yield from emit("Fetching ROR ID for the university.")
        ror_id = get_ror_id(data['university_name'])
        print("ROR ID:", ror_id)

        yield from emit("Searching for papers and building index.")
        paper_list = []

        # Profile papers
        profile = get_institution_profile(
            data['topic'], ror_id, queries=queries,
            year_start=data['year_start'], year_end=data['year_end']
        )
        profiles.append(profile)

        # Topic-wide paper sampling
        topic_papers = search_paper(
            data['topic'], institution_id=ror_id,
            year_start=data['year_start'], year_end=data['year_end'],
            samples=100
        )
        paper_list.extend(topic_papers)

        # Query-specific papers
        for query in queries:
            papers = search_paper(
                query, institution_id=ror_id,
                year_start=data['year_start'], year_end=data['year_end'],
                samples=100
            )
            paper_list.extend(papers)

        paper_list = remove_duplicates_by_doi_dict(paper_list)
        yield from emit(f"Found and indexed {len(paper_list)} unique papers.")
        index = build_index(paper_list)

        # Retrieve top-k papers per query
        for query in queries:
            docs = retrieve(
                query, institution=data['university_name'], top_k=top_k,
                year_start=data['year_start'], year_end=data['year_end'], index=index
            )
            all_retrieved_docs.append(docs)

    else:
        yield from emit("Using offline retrieval.")
        for query in queries:
            docs = retrieve(
                query, institution=data['university_name'], top_k=top_k,
                year_start=data['year_start'], year_end=data['year_end']
            )
            all_retrieved_docs.append(docs)

    total_retrieved = sum(len(docs) for docs in all_retrieved_docs)
    yield from emit(f"Retrieved {total_retrieved} documents across {len(queries)} queries (top {top_k} per query).")

    # 2. Summarization
    yield from emit("Generating summaries.")
    flat_docs = [doc for sublist in all_retrieved_docs for doc in sublist]

    summary = use_abstracts_university(flat_docs) #summarize_university(flat_docs)
    full_summary = institutional_summary(summary, data['university_name'], prompt=prompt)
    fixed_summary = fix_output_citations(full_summary)

    # Final output
    result = {
        "summary": fixed_summary,
        "profiles": profiles,
        "retrieved_docs_count": total_retrieved,
        "query_count": len(queries),
        "indexed_papers": len(paper_list) if online else None
    }

    yield f"data: {json.dumps(result)}\n\n".encode("utf-8")


def process_input(prompt, paper=None, online=False, model="gpt-4o-mini"):
    data = generate_questions(prompt, paper, llm_model=model)
    data_type = data.get("type")

    if data_type == "university":
        return handle_university(data, prompt, online=online)
    if data_type == "single_university":
        return handle_single_university_wrapper(data, prompt, online=online, model=model)
    elif data_type == "university_university":
        return handle_university_university_wrapper(data, prompt, online=online)
    elif data_type == "comparison":
        return handle_comparison(data, prompt, paper, online=online, fulltext=True), None
    elif data_type == "paper":
        return handle_paper(data, prompt, paper, online=online), None
    elif data_type == 'overview':
        return handle_overview(data, prompt, online=online, fulltext=True), None
    else:
        raise ValueError(f"Unknown type: {data_type}")

def process_input_streaming(prompt, paper=None, online=False, model="gpt-4o-mini", stream_fn=None):
    def emit(msg):
        if stream_fn:
            yield from stream_fn(msg)

    yield from emit("Classifying input")
    data = generate_questions(prompt, paper, llm_model=model)

    data_type = data.get("type")
    yield from emit(f"Detected query type: {data_type}")

    if data_type == "single_university":
        yield from handle_single_university_wrapper(data, prompt, online=online, model=model, stream=True, stream_fn=stream_fn)

    elif data_type == "university":
        yield from handle_university_wrapper(data, prompt, online=online, model=model, stream=True, stream_fn=stream_fn)

    elif data_type == "university_university":
        yield from handle_university_university_wrapper(data, prompt, online=online, stream=True, stream_fn=stream_fn)

    elif data_type == "comparison":
        yield from handle_comparison_wrapper(data, prompt, paper, online=online, fulltext=True, stream=True, stream_fn=stream_fn)

    elif data_type == "paper":
        yield from handle_paper_wrapper(data, prompt, paper, online=online, stream=True, stream_fn=stream_fn)

    elif data_type == "overview":
        yield from handle_overview_wrapper(data, prompt, online=online, fulltext=True, stream=True, stream_fn=stream_fn)

    else:
        yield from emit(f"Unknown query type: {data_type}")
        raise ValueError(f"Unknown type: {data_type}")


def process_input_structured(data, prompt, paper=None, online=False, model="gpt-4o-mini", no_cache=False):
    data_type = data.get("type")

    if data_type == "university":
        return handle_university(data, prompt, online=online)
    if data_type == "single_university":
        return handle_single_university(data, prompt, online=online, model=model)
    elif data_type == "university_university":
        return handle_university_university(data, prompt, online=online)
    elif data_type == "comparison":
        return handle_comparison(data, prompt, paper, online=online, fulltext=True), None
    elif data_type == "paper":
        return handle_paper(data, prompt, paper, online=online), None
    elif data_type == 'overview':
        return handle_overview(data, prompt, online=online, fulltext=True), None
    else:
        raise ValueError(f"Unknown type: {data_type}")


def handle_university_wrapper(*args, stream=False, **kwargs):
    if stream:
        return handle_university(*args, **kwargs)
    else:
        result = b""
        for chunk in handle_university(*args, **kwargs):
            result += chunk
        return result

def handle_university(data, prompt, online=False, model="gpt-4o-mini", stream_fn=None):
    def emit(msg):
        if stream_fn:
            yield from stream_fn(msg)

    queries = data["queries"]
    profiles = []

    if not online:
        yield from emit("📄 Retrieving documents (offline mode)...")
        all_results = [
            retrieve(query, top_k=1000, year_start=data['year_start'], year_end=data['year_end'],
                     country=data['country']) for query in queries
        ]
        docs = [item for sublist in all_results for item in sublist]
        docs = remove_duplicates_by_doi(docs)

        yield from emit("📑 Getting top documents for the university...")
        single_university_docs_ml = [
            retrieve(query, institution=data['university_name'], top_k=15,
                     year_start=data['year_start'], year_end=data['year_end']) for query in queries
        ]

        yield from emit("🏫 Extracting top universities...")
        top_universities = extract_top_universities(docs, query=prompt, exclude=data['university_name'])

    else:
        yield from emit("🔍 Getting ROR ID and profile...")
        ror_id = get_ror_id(data['university_name'])
        yield from emit("🌍 Finding relevant institutions...")
        most_relevant_institutions = get_most_relevant_institutions_for_queries(
            queries, year_start=data['year_start'], year_end=data['year_end'],
            country_code=data['country'], limit=3, exclude_ror=ror_id)
        print("Most relevant institutions:", most_relevant_institutions)
        yield from emit("Create Research Profile")
        profile = get_institution_profile(data['topic'], ror_id, queries=queries,
                                          year_start=data['year_start'], year_end=data['year_end'])
        profiles.append(profile)

        yield from emit("📄 Searching and indexing papers...")
        paper_list = []
        paper_list.extend(
            search_paper(data['topic'], institution_id=ror_id,
                         year_start=data['year_start'], year_end=data['year_end'], samples=100))

        for query in queries:
            for institution in most_relevant_institutions:
                papers = search_paper(query, institution_id=institution[0],
                                      year_start=data['year_start'], year_end=data['year_end'], samples=100)
                paper_list.extend(papers)

            papers_solo = search_paper(query, institution_id=ror_id,
                                       year_start=data['year_start'], year_end=data['year_end'], samples=100)
            paper_list.extend(papers_solo)

        paper_list = remove_duplicates_by_doi_dict(paper_list)
        index = build_index(paper_list, llm_model=model)

        yield from emit(f"📚 Indexed {len(paper_list)} papers.")

        all_results = [
            retrieve(query, top_k=10, year_start=data['year_start'], year_end=data['year_end'],
                     country=data['country'], index=index) for query in queries
        ]
        docs = [item for sublist in all_results for item in sublist]
        docs = remove_duplicates_by_doi(docs)

        yield from emit("📑 Getting top documents for the university...")
        single_university_docs_ml = [
            retrieve(query, institution=data['university_name'], top_k=10,
                     year_start=data['year_start'], year_end=data['year_end'], index=index) for query in queries
        ]

        yield from emit("🏫 Extracting top universities...")
        top_universities = extract_top_universities(docs, query=prompt, exclude=get_university_name_from_ror(ror_id))

        yield from emit("📄 Getting profiles of top universities...")
        for institution in most_relevant_institutions:
            inst_profile = get_institution_profile(data['topic'], institution[0],
                                                   queries=queries, year_start=data['year_start'], year_end=data['year_end'])
            profiles.append(inst_profile)
    print(len(profiles))
    print(profiles)
    yield from emit("🧠 Summarizing the target university...")
    single_university_docs = [item for sublist in single_university_docs_ml for item in sublist]
    university_summary = use_abstracts_university(single_university_docs) #summarize_university(single_university_docs)
    university_over_summary = institutional_summary(university_summary, data['university_name'])

    yield from emit("🧠 Summarizing comparison universities...")
    top_universities = use_abstracts_universities(top_universities) #summarize_universities(top_universities)
    top_universities = institutional_summary_structured(top_universities)

    yield from emit("🆚 Comparing universities...")
    comparison = compare_universities(top_universities, university_over_summary, prompt)
    output = fix_output_citations(comparison)

    yield f"data: {json.dumps({'summary': output, 'profiles': profiles})}\n\n".encode("utf-8")



def handle_university_university(data, prompt, top_k=15, online=False, stream_fn=None):
    def emit(msg):
        if stream_fn:
            yield from stream_fn(msg)

    queries = data["queries"]
    profiles = []
    total_indexed_papers = 0
    university_docs_ml = []
    comparison_university_docs_ml = []

    if online:
        yield from emit("Fetching ROR IDs for both universities.")
        ror_id = get_ror_id(data['university_name'])
        comp_ror_id = get_ror_id(data['comparison_university_name'])

        yield from emit("Building institutional profiles.")
        profile = get_institution_profile(data['topic'], ror_id, queries=queries,
                                          year_start=data['year_start'], year_end=data['year_end'])
        comp_profile = get_institution_profile(data['topic'], comp_ror_id, queries=queries,
                                               year_start=data['year_start'], year_end=data['year_end'])
        profiles.extend([profile, comp_profile])

        yield from emit("Collecting and indexing papers.")
        paper_list = []

        # Add topic-based papers
        paper_list.extend(search_paper(data['topic'], institution_id=ror_id,
                                       year_start=data['year_start'], year_end=data['year_end'], samples=100))
        paper_list.extend(search_paper(data['topic'], institution_id=comp_ror_id,
                                       year_start=data['year_start'], year_end=data['year_end'], samples=100))

        # Add query-specific papers
        for query in queries:
            paper_list.extend(search_paper(query, institution_id=ror_id,
                                           year_start=data['year_start'], year_end=data['year_end'], samples=100))
            paper_list.extend(search_paper(query, institution_id=comp_ror_id,
                                           year_start=data['year_start'], year_end=data['year_end'], samples=100))

        paper_list = remove_duplicates_by_doi_dict(paper_list)
        total_indexed_papers = len(paper_list)
        yield from emit(f"Indexed {total_indexed_papers} unique papers.")

        index = build_index(paper_list)

        yield from emit("Retrieving top documents per university and query.")
        university_docs_ml = [
            retrieve(query, institution=data['university_name'], top_k=top_k,
                     year_start=data['year_start'], year_end=data['year_end'], index=index)
            for query in queries
        ]
        comparison_university_docs_ml = [
            retrieve(query, institution=data['comparison_university_name'], top_k=top_k,
                     year_start=data['year_start'], year_end=data['year_end'], index=index)
            for query in queries
        ]

    else:
        yield from emit("Offline document retrieval for both universities.")
        university_docs_ml = [
            retrieve(query, institution=data['university_name'], top_k=top_k,
                     year_start=data['year_start'], year_end=data['year_end'])
            for query in queries
        ]
        comparison_university_docs_ml = [
            retrieve(query, institution=data['comparison_university_name'], top_k=top_k,
                     year_start=data['year_start'], year_end=data['year_end'])
            for query in queries
        ]

    total_docs_univ = sum(len(docs) for docs in university_docs_ml)
    total_docs_comp = sum(len(docs) for docs in comparison_university_docs_ml)
    yield from emit(f"Retrieved {total_docs_univ} docs for {data['university_name']} and {total_docs_comp} for {data['comparison_university_name']}.")

    # ===== Summarization and Comparison =====
    yield from emit(f"Summarizing papers from {data['university_name']}...")
    university_docs = [item for sublist in university_docs_ml for item in sublist]
    university_summary = use_abstracts_university(university_docs) #summarize_university(university_docs)
    university_overview = institutional_summary(university_summary, data['university_name'])

    yield from emit(f"Summarizing papers from {data['comparison_university_name']}.")
    comparison_docs = [item for sublist in comparison_university_docs_ml for item in sublist]
    comparison_summary = use_abstracts_university(comparison_docs) #summarize_university(comparison_docs)
    comparison_overview = institutional_summary(comparison_summary, data['comparison_university_name'])

    yield from emit("Comparing both universities.")
    comparison = compare_two_universities(university_overview, comparison_overview, prompt)
    output = fix_output_citations(comparison)

    final_result = {
        "summary": output,
        "profiles": profiles,
        "indexed_papers": total_indexed_papers if online else None,
        "retrieved_docs": {
            data['university_name']: total_docs_univ,
            data['comparison_university_name']: total_docs_comp
        },
        "query_count": len(queries),
        "mode": "online" if online else "offline"
    }

    yield f"data: {json.dumps(final_result)}\n\n".encode("utf-8")

def handle_university_university_wrapper(*args, stream=False, **kwargs):
    if stream:
        return handle_university_university(*args, **kwargs)
    else:
        result = b""
        for chunk in handle_university_university(*args, **kwargs):
            result += chunk
        return result

def handle_comparison_wrapper(*args, stream=False, **kwargs):
    if stream:
        return handle_comparison(*args, **kwargs)
    else:
        result = b""
        for chunk in handle_comparison(*args, **kwargs):
            result += chunk
        return result

def handle_comparison(data, prompt, paper, online=False, fulltext=False, top_k=30, top_k_internal=15, stream_fn=None):
    def emit(msg):
        if stream_fn:
            yield from stream_fn(msg)

    queries = data["queries"]
    all_results_ml = []
    retrieved_docs = []
    indexed_docs_count = 0
    mode = "offline"

    if online and fulltext:
        #mode = "online-fulltext"
        yield from emit("Loading full-text documents for topic, paper, and queries.")
        loader = CoreReader()
        full_docs = []

        for content in [data['topic'], paper['title'], paper['abstract']] + queries:
            docs = loader.load_data(content, max_results=top_k)
            full_docs.extend(docs)

        indexed_docs_count = len(full_docs)
        yield from emit(f"Indexed {indexed_docs_count} full-text documents.")
        index = build_index_core(full_docs)

        yield from emit("Retrieving documents for comparison (fulltext).")
        all_results_ml = [
            retrieve(query, top_k=top_k_internal, year_start=data['year_start'], year_end=data['year_end'], index=index)
            for query in queries
        ]

    elif online and not fulltext:
        mode = "online-metadata"
        yield from emit("Searching metadata-only papers.")
        paper_list = []

        paper_list.extend(search_paper(data['topic'], year_start=data['year_start'], year_end=data['year_end'], samples=100))

        for query in queries:
            paper_list.extend(search_paper(query, year_start=data['year_start'], year_end=data['year_end'], samples=100))

        paper_list = remove_duplicates_by_doi_dict(paper_list)
        indexed_docs_count = len(paper_list)
        yield from emit(f"📚 Indexed {indexed_docs_count} metadata papers.")
        index = build_index(paper_list)

        yield from emit("Retrieving documents for comparison (metadata)")
        all_results_ml = [
            retrieve(query, top_k=top_k_internal, year_start=data['year_start'], year_end=data['year_end'], index=index)
            for query in queries
        ]
        flat_results = [item for sublist in all_results_ml for item in sublist]
        retrieved_docs = remove_duplicates_by_doi(flat_results)

    else:
        mode = "offline"
        yield from emit("Retrieving documents (offline mode)")
        all_results_ml = [
            retrieve(query, top_k=top_k_internal, year_start=data['year_start'], year_end=data['year_end'])
            for query in queries
        ]
        retrieved_docs = [item for sublist in all_results_ml for item in sublist]

    if not retrieved_docs:
        retrieved_docs = [item for sublist in all_results_ml for item in sublist]

    yield from emit(f"Summarizing {len(retrieved_docs)} retrieved documents.")
    summaries = summarize(retrieved_docs)

    yield from emit("Comparing with target paper")
    comparison = compare_papers(summaries, prompt, paper)
    output = fix_output_citations(comparison)

    result = {
        "summary": output
    }
    print(output)
    yield f"data: {json.dumps(result)}\n\n".encode("utf-8")


def handle_overview(data, prompt, online=False, fulltext=False, top_k=50, stream_fn=None):
    def emit(msg):
        if stream_fn:
            yield from stream_fn(msg)

    queries = data["queries"]
    all_results_ml = []
    retrieved_docs = []
    indexed_docs_count = 0

    if online and fulltext:
        mode = "online-fulltext"
        yield from emit("Loading full-text documents for topic and queries")
        loader = CoreReader()
        full_docs = []

        # Load fulltext documents for topic
        full_docs.extend(loader.load_data(data['topic'], max_results=top_k))

        # Load fulltext documents for each query
        for query in queries:
            full_docs.extend(loader.load_data(query, max_results=top_k))

        indexed_docs_count = len(full_docs)
        yield from emit(f"Indexed {indexed_docs_count} full-text documents.")
        index = build_index_core(full_docs)

        yield from emit("Retrieving documents from full-text index")
        all_results_ml = [
            retrieve(query, top_k=15, year_start=data['year_start'], year_end=data['year_end'], index=index)
            for query in queries
        ]

    elif online and not fulltext:
        mode = "online-metadata"
        yield from emit("Searching metadata-only papers")
        paper_list = []

        # Search papers by topic and queries
        paper_list.extend(search_paper(data['topic'], year_start=data['year_start'], year_end=data['year_end'], samples=100))
        for query in queries:
            paper_list.extend(search_paper(query, year_start=data['year_start'], year_end=data['year_end'], samples=100))

        paper_list = remove_duplicates_by_doi_dict(paper_list)
        indexed_docs_count = len(paper_list)
        yield from emit(f"📚 Indexed {indexed_docs_count} metadata papers.")
        index = build_index(paper_list)

        yield from emit("Retrieving documents from metadata index")
        all_results_ml = [
            retrieve(query, top_k=15, year_start=data['year_start'], year_end=data['year_end'], index=index)
            for query in queries
        ]

    else:
        mode = "offline"
        yield from emit("Retrieving documents (offline mode)")
        all_results_ml = [
            retrieve(query, top_k=15, year_start=data['year_start'], year_end=data['year_end'])
            for query in queries
        ]

    # Flatten and track retrieved docs
    retrieved_docs = [item for sublist in all_results_ml for item in sublist]
    yield from emit(f"Summarizing {len(retrieved_docs)} retrieved documents")

    summaries = summarize(retrieved_docs)

    yield from emit("📊 Generating overview summary.")
    overview_result = overview(summaries, prompt)
    output = fix_output_citations(overview_result)

    result = {
        "summary": output
    }

    yield f"data: {json.dumps(result)}\n\n".encode("utf-8")

def handle_overview_wrapper(*args, stream=False, **kwargs):
    if stream:
        return handle_overview(*args, **kwargs)
    else:
        result = b""
        for chunk in handle_overview(*args, **kwargs):
            result += chunk
        return result

def handle_paper_wrapper(*args, stream=False, **kwargs):
    if stream:
        return handle_paper(*args, **kwargs)
    else:
        result = b""
        for chunk in handle_paper(*args, **kwargs):
            result += chunk
        return result


def handle_paper(data, prompt, paper, online=False, stream_fn=None):
    def emit(msg):
        if stream_fn:
            yield from stream_fn(msg)

    answer = paper_answer(prompt, paper)
    output = fix_output_citations(answer)

    result = {
        "summary": output
    }
    # GOOD (what your frontend expects):
    yield f"data: {json.dumps(result)}\n\n".encode("utf-8")
