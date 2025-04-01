from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import logging
import webbrowser
import threading

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load the dataset with error handling
csv_path = 'grapes_data.csv'
if not os.path.exists(csv_path):
    logger.error(f"CSV file not found at {csv_path}")
    raise FileNotFoundError(f"CSV file not found at {csv_path}")
try:
    data = pd.read_csv(csv_path)
    logger.info("CSV loaded successfully")
except Exception as e:
    logger.error(f"Failed to load CSV: {str(e)}")
    raise Exception(f"Failed to load CSV: {str(e)}")

# Combine relevant columns for similarity matching
data['combined'] = data['Grape Type'] + ' ' + data['Description'] + ' ' + data['Taste Profile'] + ' ' + data['Uses']

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['combined'])

# Function to get grape info based on index
def get_grape_info(index):
    grape = data.iloc[index]
    return (
        f"**Grape Type:** {grape['Grape Type']}\n"
        f"**Color:** {grape['Color']}\n"
        f"**Price:** ₹{grape['Price (₹ / 500g)']} per 500g\n"
        f"**Description:** {grape['Description']}\n"
        f"**Taste Profile:** {grape['Taste Profile']}\n"
        f"**Origin:** {grape['Origin']}\n"
        f"**Uses:** {grape['Uses']}\n"
        f"**Availability:** {grape['Availability']}"
    )

# Function to get response based on user input
def get_response(user_input):
    try:
        user_input = user_input.lower().strip()
        logger.info(f"Processing input: {user_input}")
        user_vector = vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vector, tfidf_matrix)
        best_match_index = similarities.argmax()
        similarity_score = similarities[0, best_match_index]

        if similarity_score > 0.2:  # Threshold for a decent match
            response = get_grape_info(best_match_index)
            logger.info(f"Response generated: {response[:100]}...")  # Truncate for log readability
            return response
        logger.warning("No match found above threshold")
        return "Sorry, I couldn’t find a match. Try asking about a specific grape (e.g., 'Thompson Seedless')!"
    except Exception as e:
        logger.error(f"Error in get_response: {str(e)}")
        return f"Sorry, something went wrong! Details: {str(e)}"

# Serve the chat interface
@app.route('/')
def index():
    logger.info("Serving chat.html")
    return app.send_static_file('chat.html')

# Chat endpoint
@app.route('/chat', methods=['POST'])
def chat():
    try:
        logger.info("Received POST request to /chat")
        logger.debug(f"Request data: {request.data}")
        if not request.is_json:
            logger.warning("Request is not JSON")
            return jsonify({"response": "Error: Request must be JSON"}), 400
        
        data = request.get_json()
        logger.debug(f"Parsed JSON: {data}")
        user_input = data.get('message', '').strip()
        if not user_input:
            logger.warning("No message provided")
            return jsonify({"response": "Error: No message provided"}), 400

        response = get_response(user_input)
        logger.info(f"Sending response: {response[:100]}...")  # Truncate for log
        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Error in /chat: {str(e)}")
        return jsonify({"response": f"Error: {str(e)}"}), 500

# Function to open browser
def open_browser():
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    logger.info("Starting Flask server")
    # Start the server in a separate thread and open the browser
    threading.Timer(1, open_browser).start()
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)