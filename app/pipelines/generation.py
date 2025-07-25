from typing import Dict, List
from groq import Groq
import json
import os

async def generation(generationInputs : Dict, rerankingPhase : List) -> Dict:
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": """ You are an HR scoring system that analyzes job fit and provides structured, concise, and user-friendly feedback based on a precomputed similarity score.

                    Instructions:
                    - Each job has a base similarity score already computed and given as `score` (0–100 scale).
                    - Do NOT modify or recalculate the `score`.

                    Match Relevance:
                    - Evaluate how relevant the user's certifications or work experience are to the job:
                        - Not Related → +0.00
                        - Somehow Related → +0.01 to +0.02
                        - Clearly Related → +0.03 to +0.04
                    - Store this value in a field called `boostWeight`.
                    - Do NOT add `boostWeight` to `score`.

                    Feedback Output:
                    For each job, return this JSON structure:
                    {
                        "id": "[jobUID]",
                        "score": [original score as float],
                    feedback": {
                    "match_summary": "[Short and neutral/optimistic summary. Do NOT reference other jobs (e.g., avoid 'Similar to job_1'). Each summary must stand on its own. Example: 'Entry-level, part-time food service role — could suit a flexible schedule.']",
                    "skill_note": "[Briefly describe key job skills. Suggest how the user might align better. Example: 'Requires food handling and customer service — consider adding 'customer service' if you’ve done similar work.']",
                    "extra_note": "[Always frame positively. If unrelated, say: 'No related certifications, but this could still be a solid entry point.' Keep it short and encouraging.]"
                    }

                        "boostWeight": [float between 0.00 and 0.04]
                    }

                    Notes:
                    - If the job is outside the user's main industry and no transferable skills are found, still provide helpful, respectful, and motivational feedback.
                    - Avoid negative or discouraging phrases like “doesn’t match,” “not qualified,” or “no relevant skills.”
                    - Use a tone that is constructive and open-ended, especially when the match is low or exploratory.

                    Final Output:
                    - Return a JSON array of all jobs.
                    - Sort the array in **descending order by `score`**.
                    - Do not return any explanation or comments outside the JSON.
            """
        },
        {
            "role": "user",
            "content": f"""
            content: User Profile:
            user_profile = {json.dumps(generationInputs, indent=2)}

            Job Listings:
            jobs =  {json.dumps(rerankingPhase, indent=2)}

                """
                }
                ],
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                response_format={"type":"json_object"}
            )
    #parsing the data to JSON format
    generationOutput = json.loads(chat_completion.choices[0].message.content)
    return generationOutput
