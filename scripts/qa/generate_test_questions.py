
import json
import random

def generate():
    # 1. RAG Seeds (Hardcoded based on Schema/Known Data)
    rag_seeds = {
        "programmes": [
            "Diploma in Opticianry", "Foundation in Science", "Bachelor in Formulation Science", 
            "Bachelor of Pharmacy", "Diploma in Management", "Music Degree", "Engineering",
            "Doctor of Philosophy", "Master of Architecture", "Nursing"
        ],
        "facilities": [
            "Library", "Swimming Pool", "Gym", "Cafeteria", "Block A", "Auditorium", "Lab"
        ],
        "general_rag": [
            "Hostel fees", "Accomodation", "Bus schedule", "Academic Calendar", "Exam dates"
        ]
    }
    
    rag_templates = [
        "Tell me about {0}",
        "What is the fee for {0}?",
        "How long is the {0} course?",
        "Where is the {0} located?",
        "Contact details for {0}",
        "Is {0} available?",
        "Details regarding {0}"
    ]
    
    # 2. General Seeds (Non-DB)
    gen_seeds = [
        "Sky", "Ocean", "History of Rome", "Python programming", "How to cook pasta",
        "Capital of France", "Meaning of life", "Weather today", "Who is Einstein?",
        "Write a poem", "Tell me a joke", "What is AI?", "Best movies 2024"
    ]
    
    questions = []
    
    # Generate 100 RAG Questions
    print("Generating 100 RAG questions...")
    for i in range(100):
        category = random.choice(["programmes", "facilities", "general_rag"])
        topic = random.choice(rag_seeds[category])
        template = random.choice(rag_templates)
        q_text = template.format(topic)
        questions.append({
            "type": "RAG",
            "query": q_text
        })
        
    # Generate 100 General Questions
    print("Generating 100 General questions...")
    for i in range(100):
        topic = random.choice(gen_seeds)
        q_text = f"Tell me about {topic} " + str(i) # Add uniqueness
        questions.append({
            "type": "General",
            "query": q_text
        })
        
    random.shuffle(questions)
    
    with open("stress_test_questions.json", "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2)
        
    print(f"Generated {len(questions)} questions in stress_test_questions.json")

if __name__ == "__main__":
    generate()
