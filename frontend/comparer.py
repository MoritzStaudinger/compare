from itertools import product

import pandas as pd
import streamlit as st
import requests
import altair as alt
import fitz  # PyMuPDF

st.set_page_config(layout="wide")

with st.sidebar:
    st.title("📘 Description")

    st.markdown("""
This website helps **automatically compare research papers and universities** based on your query and optionally a PDF upload.

It supports answering prompts like:
- “What has the University of Aberdeen contributed to Medical Natural Language Processing?”
- “How does this paper compare to other recent work?”
- “Compare the contributions of TU Wien in the field of efficient information retrieval to the University of Glasgow contributions, over the last 5 years .”

### 🔍 Classification Logic
Each prompt is automatically classified and can answer the following types of questions:

- **`One to many university comparison`**  
  Compare research contributions between one institution and a broader context.  
  _Example:_  
  *“What has the University of Aberdeen contributed to Information Retrieval compared to others?”*

- **`One to one university comparison`**  
  Compare **exactly two** institutions.  
  _Example:_  
  *“What has the University of Aberdeen contributed to Information Retrieval compared to the University of Edinburgh?”*

- **`Single University Overview`**  
  Provide an overview of the research output of one institution.  
  _Example:_  
  *“What has the University of Aberdeen contributed to Information Retrieval over the last 5 years?”*

- **`Comparison of research papers`**  
  Compare a specific paper with other relevant literature.  
  _Requires:* PDF
  _Example:_  
  *“How does the paper compare to other relevant literature <specific criteria>?”*

- **`Question on a single paper`**  
  Answer questions about a **specific paper**.  
  _Example:_  
  *“What is the contribution of this paper?”*

- **`Overview of a domain`**  
  Summarize trends and contributions in a research field **without any specific input document**.  
  _Example:_  
  *“What are the current trends in multimodal learning?”*

---

### 🧠 Data Sources
The app uses different data depending on the classification:

- **OpenAlex**: affiliation + abstracts  
- **CORE UK**: fulltexts  

| Type              | Data Used                      |
|-------------------|-------------------------------|
| `comparison` / `overview` | fulltexts (CORE)         |
| `university`, `single_university`, `university_university` | affiliations + abstracts (OpenAlex) |
| `paper`           | fulltext of provided PDF       |

---

🔁 The system uses caching to avoid recomputation. You can always resubmit without cache if needed.
""")


st.title("Compare Research Papers and Institutions")

# --- Request Handler ---
def submit_request(prompt, model, pdf_file, no_cache=False):
    files = {"pdf_file": (pdf_file.name, pdf_file, "application/pdf")} if pdf_file else {}

    data = {
        "prompt": prompt,
        "online": "true",
        "model": model,
        "no_cache": str(no_cache).lower(),
    }
    response = requests.post("http://backend:8000/api/process_input", files=files, data=data)
    #response = requests.post("http://localhost:8000/api/process_input", files=files, data=data)
    return response


# --- UI Elements ---
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    prompt = st.text_area("Enter your prompt", value=st.session_state.get("prompt", ""), height=70)
with col2:
    model = st.selectbox(
        "Select Model",
        ("gpt-4o-mini", "claude-3-7-sonnet-latest", "gemini-2.0-flash"),
        index=("gpt-4o-mini", "claude-3-7-sonnet-latest", "gemini-2.0-flash").index(
            st.session_state.get("model", "gpt-4o-mini")
        )
    )
with col3:
    pdf_file = st.file_uploader("Upload a PDF", type="pdf")

with st.expander("🕘 Recent Queries (click to expand)", expanded=False):
    try:
        recent_response = requests.get("http://backend:8000/api/recent_caches")
        #recent_response = requests.get("http://localhost:8000/api/recent_caches")
        if recent_response.status_code == 200:
            recent_prompts = recent_response.json()
            if recent_prompts:
                for i, item in enumerate(recent_prompts):
                    col1, col2 = st.columns([6, 1])
                    with col1:
                        st.markdown(f"**{item['prompt'][:80]}...**  \n_Model:_ `{item['model']}`")
                    with col2:
                        if st.button("Use", key=f"use_prompt_{i}"):
                            st.session_state["prompt"] = item["prompt"]
                            st.session_state["model"] = item["model"]
                            st.rerun()
            else:
                st.info("No recent cache entries found.")
        else:
            st.warning("Could not load recent prompts.")
    except Exception as e:
        st.error(f"Failed to load cache history: {e}")


# --- Button Handling ---
submit_clicked = st.button("Submit")
resubmit_clicked = False

if submit_clicked or resubmit_clicked:
    if not prompt:
        st.warning("Please enter a prompt.")
        st.stop()
    else:
        no_cache = resubmit_clicked
        response = submit_request(prompt, model, pdf_file, no_cache=no_cache)

        if response.status_code == 200:
            result = response.json()
            st.session_state.last_result = result
        else:
            st.error(f"Error {response.status_code}: {response.text}")
            st.stop()

# --- Show Results ---
if "last_result" in st.session_state:
    result = st.session_state.last_result

    if result.get("cache_timestamp"):
        # Display cache warning and inline button
        with st.container():
            st.warning(f"Results are cached and may not reflect the latest data. "
                       f"Cache timestamp: {result['cache_timestamp']}")
            if st.button("🔄 Resubmit without cache", key="resubmit_button"):
                response = submit_request(prompt, model, pdf_file, no_cache=True)
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.last_result = result
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
                    st.stop()

    st.markdown("### Response")
    st.write(result["summary"])

    if result['profiles']:
        profiles = result['profiles']
        num_profiles = len(profiles)
        cols = st.columns(num_profiles)

        global_pub_year_max = 0
        global_author_year_max = 0
        global_collab_max = 0

        for profile in profiles:
            pubs_per_year = profile.get("PublicationsPerYear", {})
            if pubs_per_year:
                global_pub_year_max = max(global_pub_year_max, max(pubs_per_year.values()))

            researcher_pubs = profile.get('ResearchersWithPublicationsPerYear', [])
            if researcher_pubs:
                df = pd.DataFrame(researcher_pubs)
                if not df.empty:
                    global_author_year_max = max(global_author_year_max, df['Publications'].max())

            collabs = profile.get("Collaborator_Universities", [])
            if collabs:
                df = pd.DataFrame(collabs, columns=["University", "Publications"])
                if not df.empty:
                    global_collab_max = max(global_collab_max, df['Publications'].max())

        for col, profile in zip(cols, profiles):
            year_start = profile.get("Year Start")
            year_end = profile.get("Year End")
            pubs_per_year = profile.get("PublicationsPerYear", {})

            if not year_start or not year_end:
                if pubs_per_year:
                    years = sorted(pubs_per_year.keys())
                    year_start = years[0]
                    year_end = years[-1]

            period = f'{year_start} - {year_end}' if year_start and year_end else "N/A"
            name = profile["Institution"]
            publications = profile["Publications"]

            with col:
                st.markdown(f"##### {name}")
                st.markdown("###### Publications per Year")
                st.markdown(f"**Period:** {period}")

                if year_start and year_end:
                    year_start = int(year_start)
                    year_end = int(year_end)
                    full_years = list(range(year_start, year_end + 1))
                else:
                    full_years = []
                #full_years = list(range(year_start, year_end + 1))

                # Convert pubs_per_year dict to DataFrame
                df = pd.DataFrame(list(pubs_per_year.items()), columns=['Year', 'Publications'])
                df['Year'] = df['Year'].astype(int)

                # Create a DataFrame with all years and merge
                df_full = pd.DataFrame({'Year': full_years})
                df = pd.merge(df_full, df, on='Year', how='left')
                df['Publications'] = df['Publications'].fillna(0).astype(int)

                # Convert Year to string for consistent x-axis formatting
                df['Year'] = df['Year'].astype(str)

                # Plot
                chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X('Year:O', title='Year'),
                    y=alt.Y('Publications:Q', title='Number of Publications',
                            axis=alt.Axis(format='d', tickMinStep=1),
                            scale=alt.Scale(domain=[0, global_pub_year_max])),
                    tooltip=['Year', 'Publications']
                ).properties(
                    width=700,
                    height=400
                )
                st.altair_chart(chart, use_container_width=True)

                if profile['ResearchersWithPublicationsPerYear']:
                    publications_per_author_year = pd.DataFrame(profile['ResearchersWithPublicationsPerYear'])
                    publications_per_author_year['Year'] = publications_per_author_year['Year'].astype(int)

                    all_years = list(range(year_start, year_end + 1))
                    all_authors = publications_per_author_year['Author'].unique()
                    full_index = pd.DataFrame(list(product(all_years, all_authors)), columns=["Year", "Author"])

                    merged = pd.merge(full_index, publications_per_author_year, how='left',
                                      on=['Year', 'Author'])
                    merged['Publications'] = merged['Publications'].fillna(0).astype(int)
                    merged['Year'] = merged['Year'].astype(str)

                    st.markdown("###### Publications by Author per Year")
                    chart = alt.Chart(merged).mark_bar().encode(
                        x=alt.X('Year:O', title='Year', axis=alt.Axis(labelAngle=0)),
                        y=alt.Y('Publications:Q', title='Number of Publications',
                                scale=alt.Scale(domain=[0, global_author_year_max]),
                                axis=alt.Axis(format='d', tickMinStep=1)),
                        color=alt.Color('Author:N', legend=alt.Legend(orient='bottom', title=None, labelFontSize=10)),
                        tooltip=['Author', 'Year', 'Publications'],
                        xOffset='Author:N'
                    ).properties(
                        width=700,
                        height=400
                    )
                    st.altair_chart(chart, use_container_width=True)

                collaborators = profile.get("Collaborator_Universities", [])
                print(collaborators)
                if collaborators:
                    st.markdown("###### Collaborator Universities")
                    df_collab = pd.DataFrame(collaborators, columns=["University", "Publications"])
                    df_collab = df_collab.sort_values("Publications", ascending=False)

                    chart = alt.Chart(df_collab).mark_bar().encode(
                        x=alt.X('University:N', sort=df_collab["University"].tolist(),
                                title='Institution',
                                axis=alt.Axis(labelAngle=-75, labelOverlap=False)),
                        y=alt.Y('Publications:Q', title='Number of Publications',
                                scale=alt.Scale(domain=[0, global_collab_max]),
                                axis=alt.Axis(format='d', tickMinStep=1)),
                        tooltip=['University', 'Publications']
                    ).properties(
                        width=700,
                        height=400
                    )
                    st.altair_chart(chart, use_container_width=True)

                st.markdown(f"**Number of Publications:** {publications}")

    # --- Feedback Form ---
    with st.form(key=f"feedback_form"):
        st.markdown("### 💬 Feedback")
        rating = st.radio(
            "How useful was this response?",
            [1, 2, 3, 4, 5],
            format_func=lambda
                x: f"{x} - {'Very Poor' if x == 1 else 'Poor' if x == 2 else 'Neutral' if x == 3 else 'Good' if x == 4 else 'Excellent'}",
            horizontal=True
        )
        comments = st.text_area("Any additional comments or suggestions?")
        submit_feedback = st.form_submit_button("Submit Feedback")

        if submit_feedback:
            st.success("Thank you for your feedback!")

            # Optionally log or send feedback somewhere
            feedback_payload = {
                "prompt": prompt,
                "model": model,
                "file_hash": result["input"].get("file_hash"),
                "rating": rating,
                "comments": comments
            }

            try:
                feedback_response = requests.post("http://backend:8000/api/feedback", json=feedback_payload)
                #feedback_response = requests.post("http://localhost:8000/api/feedback", json=feedback_payload)
                if feedback_response.status_code == 200:
                    st.success("Thank you! Your feedback was saved.")
                else:
                    st.error(f"Feedback error: {feedback_response.status_code} - {feedback_response.text}")
            except Exception as e:
                st.error(f"Failed to send feedback: {e}")
