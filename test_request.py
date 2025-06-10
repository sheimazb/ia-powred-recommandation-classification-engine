import requests
import json

def test_recommendation():
    url = "http://localhost:8000/recommend-solutions"
    headers = {"Content-Type": "application/json"}
    data = {
        "log_message": "ERROR: Complex processing failure sdf fd",
        "k": 5
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        print("Status Code:", response.status_code)
        print("\nSimilar Logs:")
        similar_logs = response.json()["similar_logs"]
        if similar_logs:
            for i, log in enumerate(similar_logs, 1):
                print(f"\nLog #{i}")
                print(f"Similarity: {log['similarity']}")
                print(f"Ticket ID: {log['ticket_id']}")
                print(f"Severity: {log['severity']}")
                print(f"Solution Title: {log['solution_title']}")
                print(f"Solution: {log['solution_content']}")
                print("-" * 50)
        else:
            print("No similar logs found")
    except requests.exceptions.RequestException as e:
        print("Error making request:", e)

if __name__ == "__main__":
    test_recommendation() 