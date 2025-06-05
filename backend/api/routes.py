import glob
import hashlib
import json
import os
from datetime import datetime, timedelta
from json import JSONDecodeError

from flask import Blueprint, request, jsonify, stream_with_context, Response

from backend.rag.pipeline import process_input, process_input_streaming
from backend.rag.rag_utils import generate_questions
from backend.utils.utils import compute_upload_hash, extract_title_abstract_fulltext

api_bp = Blueprint('api', __name__)

CACHE_DIR = "data/cache"
CACHE_EXPIRY_DAYS = 7
os.makedirs(CACHE_DIR, exist_ok=True)

@api_bp.route('/process_input', methods=['POST'])
def process():
    prompt = request.form.get("prompt")
    model = request.form.get("model")
    online = request.form.get("online", False)
    no_cache = request.form.get("no_cache", "false").lower() == "true"

    # Collect and hash relevant request data
    cache_input = {
        "prompt": prompt,
        "model": model,
        "online": str(online),
    }

    # Handle file upload
    paper = {}
    path = ""
    pdf_file = request.files.get("pdf_file")
    if pdf_file:
        file_hash = compute_upload_hash(pdf_file)
        path = f"data/temp/{file_hash}.pdf"
        pdf_file.save(path)

        title, abstract, fulltext = extract_title_abstract_fulltext(path)
        paper = {"title": title, "abstract": abstract, "fulltext": fulltext}
        cache_input["file_hash"] = file_hash
    else:
        cache_input["file_hash"] = None

    # Generate request hash and cache path
    request_hash = hashlib.md5(json.dumps(cache_input, sort_keys=True).encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{request_hash}.json")

    # Try to serve from cache
    if not no_cache and os.path.exists(cache_path):
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if datetime.now() - cache_time < timedelta(days=CACHE_EXPIRY_DAYS):
            with open(cache_path, "r") as f:
                cached_data = json.load(f)
            return jsonify(cached_data)

    # Run pipeline
    result, profiles = process_input(prompt, paper, online, model)

    # Prepare output (include input metadata in cache for traceability)
    output = {
        "summary": result.strip(),
        "profiles": profiles,
        "input": {
            "prompt": prompt,
            "model": model,
            "online": online,
            "file_hash": cache_input["file_hash"]
        },
        "cache_timestamp": None  # Not cached
    }

    # Save to cache (excluding the timestamp again)
    with open(cache_path, "w") as f:
        json.dump({k: v for k, v in output.items() if k != "cache_timestamp"}, f)

    return jsonify(output)

@api_bp.route("/generate_query", methods=['POST('])
def generate_query():
    data = request.get_json()
    prompt = data.get("prompt")
    model = data.get("model")
    online = data.get("online", False)

    if not prompt or not model:
        return jsonify({"error": "Missing required fields"}), 400

    # Run the query generation pipeline
    query = generate_questions(prompt, None, llm_model=model)

    return jsonify({
        "query": query.strip(),
        "input": {
            "prompt": prompt,
            "model": model,
            "online": online
        }
    })

@api_bp.route('/process_input_stream', methods=['POST'])
def process_stream():
    prompt = request.form.get("prompt")
    model = request.form.get("model")
    online = request.form.get("online", False)
    no_cache = request.form.get("no_cache", "false").lower() == "true"

    # Prepare cache key
    cache_input = {
        "prompt": prompt,
        "model": model,
        "online": str(online),
    }

    # Handle uploaded PDF
    paper = {}
    pdf_file = request.files.get("pdf_file")
    if pdf_file:
        file_hash = compute_upload_hash(pdf_file)
        path = f"data/temp/{file_hash}.pdf"
        pdf_file.save(path)
        title, abstract, fulltext = extract_title_abstract_fulltext(path)
        paper = {"title": title, "abstract": abstract, "fulltext": fulltext}
        cache_input["file_hash"] = file_hash
    else:
        cache_input["file_hash"] = None

    # Caching logic
    request_hash = hashlib.md5(json.dumps(cache_input, sort_keys=True).encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{request_hash}.json")

    def event_stream():
        # Serve from cache if valid
        if not no_cache and os.path.exists(cache_path):
            cache_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if datetime.now() - cache_time < timedelta(days=CACHE_EXPIRY_DAYS):
                with open(cache_path, "r") as f:
                    cached_data = json.load(f)
                yield f"data: {json.dumps({'cached': True, **cached_data})}\n\n"
                yield "event: done\ndata: {}\n\n"
                return

        # Otherwise, run pipeline
        def stream_fn(message):
            return [f"data: {json.dumps({'update': message})}\n\n"]

        summary, profiles = None, None
        for message in process_input_streaming(prompt, paper=paper, online=online, model=model, stream_fn=stream_fn):
            if isinstance(message, bytes):
                decoded = message.decode("utf-8")
            else:
                decoded = message  # already a string

            if '"summary":' in decoded:
                try:
                    parsed = json.loads(decoded.replace("data: ", "").strip())
                    summary = parsed.get("summary")
                    print(parsed.get("profiles"))
                    profiles = parsed.get("profiles")
                except (UnicodeDecodeError, JSONDecodeError):
                    pass

            yield message  # original message (str or bytes), Flask will handle both

        # Save to cache if successful
        if summary is not None:
            output = {
                "summary": summary.strip(),
                "profiles": profiles,
                "input": cache_input,
                "cache_timestamp": datetime.now().isoformat(),
            }
            with open(cache_path, "w") as f:
                json.dump({k: v for k, v in output.items()}, f)

        yield "event: done\ndata: {}\n\n"

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")

@api_bp.route('/feedback', methods=['POST'])
def add_feedback():
    data = request.get_json()

    prompt = data.get("prompt")
    model = data.get("model")
    file_hash = data.get("file_hash")
    rating = data.get("rating")
    comments = data.get("comments")

    if not all([prompt, model]):
        return jsonify({"error": "Missing required fields"}), 400

    # Recompute the cache key
    cache_input = {
        "prompt": prompt,
        "model": model,
        "online": "true",  # assuming online mode is always used
        "file_hash": file_hash,
    }

    request_hash = hashlib.md5(json.dumps(cache_input, sort_keys=True).encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{request_hash}.json")

    if not os.path.exists(cache_path):
        return jsonify({"error": "No cached entry found for this input."}), 404

    # Load cache and update feedback field
    with open(cache_path, "r") as f:
        cached_data = json.load(f)

    feedback_entry = {
        "timestamp": datetime.now().isoformat(),
        "rating": rating,
        "comments": comments
    }

    if "feedback" not in cached_data:
        cached_data["feedback"] = []

    cached_data["feedback"].append(feedback_entry)

    with open(cache_path, "w") as f:
        json.dump(cached_data, f)

    return jsonify({"message": "Feedback saved successfully."}), 200


@api_bp.route('/recent_caches', methods=['GET'])
def recent_caches():
    cache_files = sorted(
        glob.glob(os.path.join(CACHE_DIR, "*.json")),
        key=os.path.getmtime,
        reverse=True
    )[:5]  # Get up to 5 most recent

    recent = []
    for path in cache_files:
        try:
            with open(path, "r") as f:
                data = json.load(f)
                if data.get("file_hash") is not None:
                    continue
            recent.append({
                "prompt": data.get("input", {}).get("prompt", "N/A"),
                "model": data.get("input", {}).get("model", "N/A"),
                "file_hash": data.get("input", {}).get("file_hash"),
                "summary": data.get("summary", "")[:250],  # preview
            })
        except Exception as e:
            continue  # skip broken files

    return jsonify(recent)