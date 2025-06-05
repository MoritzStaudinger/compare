import json

from backend.core.reader import CoreReader
from backend.openalex.utils import get_most_relevant_institutions_for_queries, get_ror_id, search_paper, \
    get_institution_profile, get_university_name_from_ror
from backend.rag.rag_utils import build_index, generate_questions, retrieve, extract_top_universities, build_index_core, \
    summarize_university, institutional_summary, summarize_universities, institutional_summary_structured, \
    compare_universities, compare_papers, overview, paper_answer, summarize, compare_two_universities, \
    fix_output_citations
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

    if online:
        yield from emit("🔍 Getting ROR ID...")
        ror_id = get_ror_id(data['university_name'])
        print("ROR ID:", ror_id)

        yield from emit("📄 Searching papers and building index...")
        paper_list = []
        profile = get_institution_profile(data['topic'], ror_id, queries=queries, year_start=data['year_start'],
                                          year_end=data['year_end'])
        profiles.append(profile)
        paper_list.extend(
            search_paper(data['topic'], institution_id=ror_id, year_start=data['year_start'], year_end=data['year_end'],
                         samples=100))

        for query in queries:
            papers_univ = search_paper(query, institution_id=ror_id, year_start=data['year_start'],
                                       year_end=data['year_end'], samples=100)
            paper_list.extend(papers_univ)

        paper_list = remove_duplicates_by_doi_dict(paper_list)
        index = build_index(paper_list)
        yield from emit(f"📚 Indexed {len(paper_list)} papers.")

        university_docs_ml = [
            retrieve(query, institution=data['university_name'], top_k=top_k, year_start=data['year_start'],
                     year_end=data['year_end'], index=index)
            for query in queries
        ]

        yield from emit(f"documents retrieved: {len(university_docs_ml)}")

    else:
        yield from emit("📄 Offline retrieval...")
        university_docs_ml = [
            retrieve(query, institution=data['university_name'], top_k=top_k, year_start=data['year_start'],
                     year_end=data['year_end'])
            for query in queries
        ]

    yield from emit("🧠 Summarizing papers...")
    university_docs = [item for sublist in university_docs_ml for item in sublist]
    university_summary = summarize_university(university_docs)
    university_over_summary = institutional_summary(university_summary, data['university_name'], prompt=prompt)
    output = fix_output_citations(university_over_summary)

    yield f"data: {json.dumps({'summary': output, 'profiles': profiles})}\n\n".encode("utf-8")


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

    yield from emit("Generating questions...")
    data = generate_questions(prompt, paper, llm_model=model)

    data_type = data.get("type")
    yield from emit(f"Detected query type: {data_type}")

    if data_type == "single_university":
        yield from handle_single_university_wrapper(data, prompt, online=online, stream=True, stream_fn=stream_fn)

    elif data_type == "university":
        yield from handle_university_wrapper(data, prompt, online=online, model=model, stream=True, stream_fn=stream_fn)

    elif data_type == "university_university":
        yield from handle_university_university_wrapper(data, prompt, online=online, stream=True, stream_fn=stream_fn)

    elif data_type == "comparison":
        result = handle_comparison_wrapper(data, prompt, paper, online=online, fulltext=True, stream=True, stream_fn=stream_fn)
        yield from emit("Comparison complete.")
        yield f"data: {json.dumps({'summary': result, 'profiles': None})}\n\n"

    elif data_type == "paper":
        result = handle_paper_wrapper(data, prompt, paper, online=online, stream=True, stream_fn=stream_fn)
        yield from emit("Answer generated from paper.")
        yield f"data: {json.dumps({'summary': result, 'profiles': None})}\n\n"

    elif data_type == "overview":
        result = handle_overview_wrapper(data, prompt, online=online, fulltext=True, stream=True, stream_fn=stream_fn)
        yield from emit("Overview complete.")
        yield f"data: {json.dumps({'summary': result, 'profiles': None})}\n\n"

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
            country_code=data['country'], limit=3)
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

    yield from emit("🧠 Summarizing the target university...")
    single_university_docs = [item for sublist in single_university_docs_ml for item in sublist]
    university_summary = summarize_university(single_university_docs)
    university_over_summary = institutional_summary(university_summary, data['university_name'])

    yield from emit("🧠 Summarizing comparison universities...")
    top_universities = summarize_universities(top_universities)
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

    if online:
        yield from emit("🔍 Getting ROR IDs...")
        ror_id = get_ror_id(data['university_name'])
        comp_ror_id = get_ror_id(data['comparison_university_name'])

        yield from emit("📄 Searching papers and building index...")
        paper_list = []
        profile = get_institution_profile(data['topic'], ror_id, queries=queries,
                                          year_start=data['year_start'], year_end=data['year_end'])
        comp_profile = get_institution_profile(data['topic'], comp_ror_id, queries=queries,
                                               year_start=data['year_start'], year_end=data['year_end'])
        profiles.extend([profile, comp_profile])

        paper_list.extend(search_paper(data['topic'], institution_id=ror_id,
                                       year_start=data['year_start'], year_end=data['year_end'], samples=100))
        paper_list.extend(search_paper(data['topic'], institution_id=comp_ror_id,
                                       year_start=data['year_start'], year_end=data['year_end'], samples=100))

        for query in queries:
            paper_list.extend(search_paper(query, institution_id=ror_id,
                                           year_start=data['year_start'], year_end=data['year_end'], samples=100))
            paper_list.extend(search_paper(query, institution_id=comp_ror_id,
                                           year_start=data['year_start'], year_end=data['year_end'], samples=100))

        paper_list = remove_duplicates_by_doi_dict(paper_list)
        index = build_index(paper_list)
        yield from emit(f"📚 Indexed {len(paper_list)} papers.")

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
        yield from emit("📄 Offline retrieval...")
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

    yield from emit("🧠 Summarizing university papers...")
    university_docs = [item for sublist in university_docs_ml for item in sublist]
    university_summary = summarize_university(university_docs)
    university_over_summary = institutional_summary(university_summary, data['university_name'])

    yield from emit("🧠 Summarizing comparison university papers...")
    comparison_university_docs = [item for sublist in comparison_university_docs_ml for item in sublist]
    comparison_university_summary = summarize_university(comparison_university_docs)
    comparison_university_over_summary = institutional_summary(comparison_university_summary,
                                                               data['comparison_university_name'])

    yield from emit("🆚 Comparing universities...")
    comparison = compare_two_universities(university_over_summary, comparison_university_over_summary, prompt)
    output = fix_output_citations(comparison)

    yield f"data: {json.dumps({'summary': output, 'profiles': profiles})}\n\n".encode("utf-8")

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

    if online and fulltext:
        yield from emit("📥 Loading full-text documents...")
        loader = CoreReader()
        full_docs = []
        docs = loader.load_data(data['topic'], max_results=top_k)
        full_docs.extend(docs)
        docs = loader.load_data(paper['title'], max_results=top_k)
        full_docs.extend(docs)
        docs = loader.load_data(paper['abstract'], max_results=top_k)
        full_docs.extend(docs)

        for query in queries:
            docs = loader.load_data(query, max_results=top_k)
            full_docs.extend(docs)

        yield from emit(f"📚 Indexed {len(full_docs)} full-text documents.")
        index = build_index_core(full_docs)

        yield from emit("🔍 Retrieving documents for comparison...")
        all_results_ml = [
            retrieve(query, top_k=top_k_internal, year_start=data['year_start'], year_end=data['year_end'], index=index)
            for query in queries
        ]

    elif online and not fulltext:
        yield from emit("📄 Searching metadata-only papers...")
        paper_list = []
        paper_list.extend(search_paper(data['topic'], year_start=data['year_start'], year_end=data['year_end'], samples=100))

        for query in queries:
            papers = search_paper(query, year_start=data['year_start'], year_end=data['year_end'], samples=100)
            paper_list.extend(papers)

        paper_list = remove_duplicates_by_doi_dict(paper_list)
        yield from emit(f"📚 Indexed {len(paper_list)} papers.")
        index = build_index(paper_list)

        yield from emit("🔍 Retrieving documents for comparison...")
        all_results_ml = [
            retrieve(query, top_k=top_k_internal, year_start=data['year_start'], year_end=data['year_end'], index=index)
            for query in queries
        ]
        all_results_ml = remove_duplicates_by_doi_dict(all_results_ml)

    else:
        yield from emit("📄 Offline retrieval...")
        all_results_ml = [
            retrieve(query, top_k=top_k_internal, year_start=data['year_start'], year_end=data['year_end'])
            for query in queries
        ]

    yield from emit("🧠 Summarizing retrieved documents...")
    all_results = [item for sublist in all_results_ml for item in sublist]
    summaries = summarize(all_results)

    yield from emit("🆚 Comparing papers...")
    comparison = compare_papers(summaries, prompt, paper)
    output = fix_output_citations(comparison)

    yield f"data: {json.dumps({'summary': output})}\n\n".encode("utf-8")


def handle_overview(data, prompt, online=False, fulltext=False, top_k=50, stream_fn=None):
    def emit(msg):
        if stream_fn:
            yield from stream_fn(msg)

    queries = data["queries"]

    if online and fulltext:
        yield from emit("📥 Loading full-text documents...")
        loader = CoreReader()
        full_docs = []
        docs = loader.load_data(data['topic'], max_results=top_k)
        full_docs.extend(docs)
        for query in queries:
            docs = loader.load_data(query, max_results=top_k)
            full_docs.extend(docs)

        yield from emit(f"📚 Indexed {len(full_docs)} full-text documents.")
        index = build_index_core(full_docs)

        yield from emit("🔍 Retrieving documents from full-text index...")
        all_results_ml = [
            retrieve(query, top_k=15, year_start=data['year_start'], year_end=data['year_end'], index=index)
            for query in queries
        ]

    elif online and not fulltext:
        yield from emit("📄 Searching metadata-only papers...")
        paper_list = []
        paper_list.extend(search_paper(data['topic'], year_start=data['year_start'], year_end=data['year_end'], samples=100))
        for query in queries:
            papers = search_paper(query, year_start=data['year_start'], year_end=data['year_end'], samples=100)
            paper_list.extend(papers)

        paper_list = remove_duplicates_by_doi_dict(paper_list)
        yield from emit(f"📚 Indexed {len(paper_list)} papers.")
        index = build_index(paper_list)

        yield from emit("🔍 Retrieving documents...")
        all_results_ml = [
            retrieve(query, top_k=15, year_start=data['year_start'], year_end=data['year_end'], index=index)
            for query in queries
        ]

    else:
        yield from emit("📄 Offline retrieval...")
        all_results_ml = [
            retrieve(query, top_k=15, year_start=data['year_start'], year_end=data['year_end'])
            for query in queries
        ]

    yield from emit("🧠 Summarizing retrieved documents...")
    all_results = [item for sublist in all_results_ml for item in sublist]
    summaries = summarize(all_results)

    yield from emit("📊 Generating overview...")
    overview_result = overview(summaries, prompt)
    output = fix_output_citations(overview_result)

    yield f"data: {json.dumps({'summary': output})}\n\n".encode("utf-8")

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

    yield from emit("🧠 Generating answer based on the selected paper...")
    answer = paper_answer(prompt, paper)
    output = fix_output_citations(answer)

    yield f"data: {json.dumps({'summary': output})}\n\n".encode("utf-8")
