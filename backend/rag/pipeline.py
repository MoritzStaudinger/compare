from backend.core.reader import CoreReader
from backend.openalex.utils import get_most_relevant_institutions_for_queries, get_ror_id, search_paper, get_institution_profile
from backend.rag.rag_utils import build_index, generate_questions, retrieve, extract_top_universities, build_index_core, \
    summarize_university, institutional_summary, summarize_universities, institutional_summary_structured, \
    compare_universities, compare_papers, overview, paper_answer, summarize, compare_two_universities, \
    fix_output_citations
from backend.utils.utils import remove_duplicates_by_doi, remove_duplicates_by_doi_dict


def handle_single_university(data, prompt, online, model, top_k=15):
    queries = data["queries"]
    profiles = []
    if online:
        ror_id = get_ror_id(data['university_name'])
        paper_list = []
        profile = get_institution_profile(data['topic'], ror_id, queries=queries, year_start=data['year_start'], year_end=data['year_end'])
        profiles.append(profile)
        paper_list.extend(search_paper(data['topic'], institution_id=ror_id, year_start=data['year_start'], year_end=data['year_end'], samples=100))
        for query in queries:
            papers_univ = search_paper(query, institution_id=ror_id, year_start=data['year_start'],
                                       year_end=data['year_end'], samples=100)
            paper_list.extend(papers_univ)

        paper_list = remove_duplicates_by_doi_dict(paper_list)
        index = build_index(paper_list)
        print("length of paper list", len(paper_list))
        university_docs_ml = [
            retrieve(query, institution=data['university_name'], top_k=top_k, year_start=data['year_start'],
                     year_end=data['year_end'], index=index) for query in queries]

    else:
        university_docs_ml = [
            retrieve(query, institution=data['university_name'], top_k=top_k, year_start=data['year_start'],
                     year_end=data['year_end']) for query in queries]

    university_docs = list(item for sublist in university_docs_ml for item in sublist)
    university_summary = summarize_university(university_docs)
    university_over_summary = institutional_summary(university_summary, data['university_name'])
    output = fix_output_citations(university_over_summary)
    return output, profiles


def process_input(prompt, paper=None, online=False, model="gpt-4o-mini"):
    data = generate_questions(prompt, paper, llm_model=model)
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

def handle_university(data, prompt, online=False, model="gpt-4o-mini"):
    queries = data["queries"]
    profiles = []
    if not online:
        all_results = [retrieve(query, top_k=1000, year_start=data['year_start'], year_end=data['year_end'],
                                country=data['country']) for query in queries]
        docs = list(item for sublist in all_results for item in sublist)
        docs = remove_duplicates_by_doi(docs)
        single_university_docs_ml = [
            retrieve(query, institution=data['university_name'], top_k=15, year_start=data['year_start'],
                     year_end=data['year_end']) for query in queries]
        top_universities = extract_top_universities(docs, query=prompt,
                                                    exclude=data['university_name'])  # query=" ".join(data["queries"]))
    else:
        most_relevant_institutions = get_most_relevant_institutions_for_queries(queries, year_start=data['year_start'],
                                                                                year_end=data['year_end'],
                                                                                country_code=data['country'], limit=3)
        ror_id = get_ror_id(data['university_name'])

        profile = get_institution_profile(data['topic'], ror_id, queries=queries, year_start=data['year_start'],
                                          year_end=data['year_end'])
        profiles.append(profile)
        paper_list = []
        paper_list.extend(
            search_paper(data['topic'], institution_id=ror_id, year_start=data['year_start'], year_end=data['year_end'],
                         samples=100))
        for query in queries:
            for institution in most_relevant_institutions:
                papers = search_paper(query, institution_id=institution[0], year_start=data['year_start'],
                                      year_end=data['year_end'], samples=100)
                paper_list.extend(papers)
            papers_solo = search_paper(query, institution_id=ror_id, year_start=data['year_start'],
                                       year_end=data['year_end'], samples=100)
            paper_list.extend(papers_solo)
        paper_list = remove_duplicates_by_doi_dict(paper_list)
        index = build_index(paper_list, llm_model=model)
        all_results = [
            retrieve(query, top_k=10, year_start=data['year_start'], year_end=data['year_end'], country=data['country'],
                     index=index) for query in queries]
        docs = list(item for sublist in all_results for item in sublist)
        docs = remove_duplicates_by_doi(docs)
        single_university_docs_ml = [
            retrieve(query, institution=data['university_name'], top_k=10, year_start=data['year_start'],
                     year_end=data['year_end'], index=index) for query in queries]
        top_universities = extract_top_universities(docs, query=prompt,
                                                    exclude=data['university_name'])  # query=" ".join(data["queries"]))
    for institution in most_relevant_institutions:
        profiles.append(profile = get_institution_profile(data['topic'], ror_id, queries=queries, year_start=data['year_start'], year_end=data['year_end']))

    single_university_docs = list(item for sublist in single_university_docs_ml for item in sublist)
    university_summary = summarize_university(single_university_docs)
    university_over_summary = institutional_summary(university_summary, data['university_name'])
    top_universities = summarize_universities(top_universities)
    top_universities = institutional_summary_structured(top_universities)
    comparison = compare_universities(top_universities, university_over_summary, prompt)
    return comparison, profiles


def handle_university_university(data, prompt, top_k=15, online=False):
    queries = data["queries"]
    profiles = []
    if online:
        ror_id = get_ror_id(data['university_name'])
        comp_ror_id = get_ror_id(data['comparison_university_name'])
        paper_list = []
        profile = get_institution_profile(data['topic'], ror_id, queries=queries, year_start=data['year_start'], year_end=data['year_end'])
        comp_profile = get_institution_profile(data['topic'], comp_ror_id, queries=queries, year_start=data['year_start'], year_end=data['year_end'])
        profiles.append(profile)
        profiles.append(comp_profile)
        paper_list.extend(
            search_paper(data['topic'], institution_id=ror_id, year_start=data['year_start'], year_end=data['year_end'],
                         samples=100))
        paper_list.extend(
            search_paper(data['topic'], institution_id=comp_ror_id, year_start=data['year_start'], year_end=data['year_end'],
                         samples=100))
        for query in queries:
            papers_univ = search_paper(query, institution_id=ror_id, year_start=data['year_start'],
                                       year_end=data['year_end'], samples=100)
            paper_list.extend(papers_univ)
            papers_comp = search_paper(query, institution_id=comp_ror_id, year_start=data['year_start'],
                                       year_end=data['year_end'], samples=100)
            paper_list.extend(papers_comp)
        paper_list = remove_duplicates_by_doi_dict(paper_list)
        index = build_index(paper_list)
        university_docs_ml = [
            retrieve(query, institution=data['university_name'], top_k=top_k, year_start=data['year_start'],
                     year_end=data['year_end'], index=index) for query in queries]
        comparison_university_docs_ml = [
            retrieve(query, institution=data['comparison_university_name'], top_k=top_k, year_start=data['year_start'],
                     year_end=data['year_end'], index=index) for query in queries]
    else:
        university_docs_ml = [
            retrieve(query, institution=data['university_name'], top_k=top_k, year_start=data['year_start'],
                     year_end=data['year_end']) for query in queries]
        comparison_university_docs_ml = [
            retrieve(query, institution=data['comparison_university_name'], top_k=top_k, year_start=data['year_start'],
                     year_end=data['year_end']) for query in queries]

    university_docs = list(item for sublist in university_docs_ml for item in sublist)
    comparison_university_docs = list(item for sublist in comparison_university_docs_ml for item in sublist)
    university_summary = summarize_university(university_docs)
    university_over_summary = institutional_summary(university_summary, data['university_name'])

    comparison_university_summary = summarize_university(comparison_university_docs)
    comparison_university_over_summary = institutional_summary(comparison_university_summary,
                                                               data['comparison_university_name'])
    comparison = compare_two_universities(university_over_summary, comparison_university_over_summary, prompt)
    return comparison, profiles


def handle_comparison(data, prompt, paper, online=False, fulltext=False, top_k = 30, top_k_internal= 15):
    queries = data["queries"]
    if online and fulltext:
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
        index = build_index_core(full_docs)
        all_results_ml = [
            retrieve(query, top_k=top_k_internal, year_start=data['year_start'], year_end=data['year_end'], index=index) for query
            in queries]
    elif online and not fulltext:
        paper_list = []
        paper_list.extend(
            search_paper(data['topic'], year_start=data['year_start'], year_end=data['year_end'],
                         samples=100))
        for query in queries:
            papers = search_paper(query, year_start=data['year_start'], year_end=data['year_end'], samples=100)
            paper_list.extend(papers)
        paper_list = remove_duplicates_by_doi_dict(paper_list)
        index = build_index(paper_list)
        all_results_ml = [
            retrieve(query, top_k=15, year_start=data['year_start'], year_end=data['year_end'], index=index) for query
            in queries]
        all_results_ml = remove_duplicates_by_doi_dict(all_results_ml)
    else:
        all_results_ml = [retrieve(query, top_k=15, year_start=data['year_start'], year_end=data['year_end']) for query
                          in queries]

    all_results = list(item for sublist in all_results_ml for item in sublist)
    summaries = summarize(all_results)
    comparison = compare_papers(summaries, prompt, paper)

    return comparison


def handle_overview(data, prompt, online=False, fulltext=False, top_k=50):
    queries = data["queries"]
    if online and fulltext:
        loader = CoreReader()
        full_docs = []
        docs = loader.load_data(data['topic'], max_results=top_k)
        full_docs.extend(docs)
        for query in queries:
            docs = loader.load_data(query, max_results=top_k)
            full_docs.extend(docs)
        print("Documents indexed", len(full_docs))
        index = build_index_core(full_docs)
        all_results_ml = [
            retrieve(query, top_k=15, year_start=data['year_start'], year_end=data['year_end'], index=index) for query
            in queries]
    elif online and not fulltext:
        paper_list = []
        paper_list.extend(
            search_paper(data['topic'], year_start=data['year_start'], year_end=data['year_end'],
                         samples=100))
        for query in queries:
            papers = search_paper(query, year_start=data['year_start'], year_end=data['year_end'], samples=100)
            paper_list.extend(papers)
        paper_list = remove_duplicates_by_doi_dict(paper_list)
        index = build_index(paper_list)
        all_results_ml = [
            retrieve(query, top_k=15, year_start=data['year_start'], year_end=data['year_end'], index=index) for query
            in queries]
    else:
        all_results_ml = [retrieve(query, top_k=15, year_start=data['year_start'], year_end=data['year_end']) for query
                          in queries]

    all_results = list(item for sublist in all_results_ml for item in sublist)
    summaries = summarize(all_results)
    overview_result = overview(summaries, prompt)
    return overview_result


def handle_paper(data, prompt, paper, online=False):
    answer = paper_answer(prompt, paper)
    return answer
