from flask import Flask, request, jsonify

app = Flask(__name__)

# Define the callback URL path and allow POST requests
@app.route('/callback', methods=['POST'])
def handle_callback():
    # 1. Grab the incoming data (assumes JSON format)
    incoming_data = request.get_json()
    
    # Print it to your local console so you can see it
    print(f"Received Callback Data: {incoming_data}")
    
    # 2. Process the data (Example logic)
    if not incoming_data:
        return jsonify({"status": "error", "message": "No data received"}), 400
        
    user_status = incoming_data.get("status", "unknown")
    
    # 3. Formulate the response payload back to the sender
    response_payload = {
        "status": "success",
        "message": f"Callback processed successfully. Status received: {user_status}"
    }
    
    # Return response payload and an HTTP 200 OK status
    return jsonify(response_payload), 200

if __name__ == '__main__':
    # Runs the server locally on port 5000
    app.run(port=5000, debug=True)
