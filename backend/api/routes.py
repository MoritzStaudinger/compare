import glob
import hashlib
import json
import os
from datetime import datetime, timedelta

from flask import Blueprint, request, jsonify

from backend.rag.pipeline import process_input
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
            cached_data["cache_timestamp"] = cache_time.isoformat()
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