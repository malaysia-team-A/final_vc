import os
import re
import json
from google import genai # New SDK
from .language_engine import multilingual  # Multi-language support (Updated relative import)

class AIEngine:
    def __init__(self, model_name="gemma-3-27b-it"):
        """
        Initialize using the NEW Google Gen AI SDK (google-genai)
        """
        self.raw_model_name = model_name
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.client = None

        if self.api_key:
            try:
                # New SDK Client Initialization
                print(f"[INIT] Initializing Gemini AI ({self.raw_model_name}) via NEW Google Gen AI SDK...")
                self.client = genai.Client(api_key=self.api_key)
                
                # Normalize model name for new SDK (e.g., remove 'models/' prefix if present)
                # The new SDK typically expects 'gemini-1.5-flash'
                self.model_name = self.raw_model_name.replace("models/", "")
                
            except Exception as e:
                print(f"Gemini Init Failed: {e}")
                self.client = None
        else:
            print("[ERROR] GOOGLE_API_KEY not found.")

        # PROMPTS (Modified dynamically in process_message)
        
        self.qa_template = """You are Kai, a friendly university assistant.

Context:
{context}

Conversation History:
{conversation}

Question: {question}

Instructions:
1. BE CONCISE: Keep answers under 3 sentences unless detailed info is requested.
2. USE CONTEXT DATA: If Context contains student info (GPA, CGPA, grades, etc.), YOU MUST include those values in your answer. Never say "I don't have access" when data is in Context.
3. PLAIN TEXT ONLY: Do NOT use any markdown formatting like **bold**, *italics*, ##headers, or bullet points. Write in plain sentences only.
4. LANGUAGE: {lang_instruction}
5. Format: STRICT JSON ONLY.
   {{
      "text": "Answer in plain text...",
      "suggestions": ["Short Q1", "Short Q2", "Short Q3"]
   }}
"""

    def process_message(self, user_message: str, data_context: str = "", conversation_history=None, language: str = "en") -> dict:
        """
        Unified processing with multi-language parsing
        Returns JSON: { "response": str, "suggestions": list, "needs_context": bool, "search_term": str }
        """
        if not self.client:
            return {"response": "System Error: AI Model not initialized.", "suggestions": []}

        try:
            # 1. Prepare Conversation Text
            conversation_text = ""
            if conversation_history:
                recent = conversation_history[-6:] # Keep it short to save tokens
                segments = []
                for item in recent:
                    role = "User" if item.get("role") == "user" else "Model"
                    content = item.get('content', '')
                    # Clean previous JSON outputs from history to avoid confusion
                    try:
                        c_json = json.loads(content)
                        if isinstance(c_json, dict):
                            content = c_json.get('text', '')
                    except:
                        pass
                    segments.append(f"{role}: {content}")
                conversation_text = "\n".join(segments)

            # 2. Construct Prompt (One-Shot Decision)
            lang_instruction = multilingual.get_ai_language_instruction(language)
            
            if data_context:
                # PHASE 2: We have data, generate answer.
                prompt = self.qa_template.format(
                    context=data_context,
                    conversation=conversation_text,
                    question=user_message,
                    lang_instruction=lang_instruction
                )
            else:
                # PHASE 1: Intent Detection
                prompt = f"""You are Kai, a friendly university assistant. Analyze the user's input.

Current Conversation:
{conversation_text}

User Input: {user_message}

Instructions:
1. Check Intent:
   - GREETING/CHIT-CHAT (hi, hello, thanks, who are you?): needs_context: false. Answer directly and briefly.
   - MY INFO/DATA (my grades, my id, who am i, my GPA): needs_context: true, search_term: "self".
   - STUDENT SEARCH (search for student X): needs_context: true, search_term: "name or id".
   - GENERAL UNI INFO (fees, courses, location, hostel, schedule): needs_context: false (Answer from your knowledge). Only set true if specific stats are needed.

2. SMART SUGGESTIONS: Generate 2-3 helpful follow-up questions that:
   - Are specific and actionable (not generic like "Tell me more")
   - Relate to the current topic or naturally extend the conversation
   - Examples for student info: "What is my current semester?", "Show my enrolled courses"
   - Examples for general: "What are the hostel prices?", "When is the registration deadline?"

3. PLAIN TEXT ONLY: No markdown like **bold** or *italics*. Write in plain sentences.
4. LANGUAGE: {lang_instruction}

5. Output Format (JSON ONLY):
   {{
      "text": "Response text (if not needing context)...", 
      "suggestions": ["Specific Q1", "Specific Q2", "Specific Q3"],
      "needs_context": true/false,
      "search_term": "keyword or null"
   }}
"""
            # 3. Call API
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            raw_text = response.text.strip()
            
            # 4. Parse JSON
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                # Normalize output keys
                return {
                    "response": data.get("text", "I'm thinking..."),
                    "suggestions": data.get("suggestions", []),
                    "needs_context": data.get("needs_context", False),
                    "search_term": data.get("search_term", None)
                }

            else:
                # Fallback for plain text response
                return {
                    "response": raw_text, 
                    "suggestions": ["Menu", "Contact"],
                    "needs_context": False
                }

        except Exception as e:
            print(f"AI Error: {e}")
            return {"response": "I'm having trouble connecting right now.", "suggestions": []}

    # ... (Unified process_message method kept above) ...
    # Deprecated fallback methods removed for cleanliness.
