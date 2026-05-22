from flask import Flask, request, jsonify, render_template
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import pickle, time
app = Flask(__name__)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
REQUEST_COUNT = Counter(
    'spam_prediction_requests_total',
    'Total prediction requests',
    ['result']
)
REQUEST_LATENCY = Histogram(
    'spam_prediction_latency_seconds',
    'Prediction latency in seconds'
)

@app.route('/')
def home():
    """Serve the web UI."""
    return render_template('index.html')

@app.route('/health')
def health():
    """Kubernetes liveness probe endpoint."""
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    """ML prediction endpoint — called by the frontend via fetch()."""
    start = time.time()

    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "Provide JSON with 'message' field"}), 400

    msg    = data['message']
    pred   = model.predict([msg])[0]
    prob   = model.predict_proba([msg])[0]
    result = "SPAM" if pred == 1 else "NOT SPAM"

    REQUEST_COUNT.labels(result=result).inc()
    REQUEST_LATENCY.observe(time.time() - start)

    return jsonify({
        "message":    msg,
        "prediction": result,
        "confidence": round(float(max(prob)) * 100, 2)
    })

@app.route('/metrics')
def metrics():
    """Prometheus scrape endpoint."""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)