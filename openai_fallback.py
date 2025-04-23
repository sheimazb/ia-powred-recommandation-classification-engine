from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key="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")

def ask_openai_exception_type(log_message):
    """Use OpenAI to identify the specific exception type from a log message."""

    prompt = f"""
You are a specialized exception analyzer. Your task is to identify the specific exception type from a log message.

Examine this log message and tell me the exact exception class name (e.g., NullPointerException, ResourceAccessException, ProductNotFoundException, etc.).
If it's a custom exception, infer its proper name from the context.

Do not return general categories - I want the actual class name of the exception.
If you're creating a custom name, follow Java/C# naming conventions (PascalCase + 'Exception' suffix).

Log message: "{log_message}"

Return ONLY the exception class name, nothing else.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI_Error: {e}"

# Example usage
if __name__ == "__main__":
    spring_log = """
    message": "2025-04-17T17:27:01.009Z ERROR 1 --- [nio-8082-exec-3] c.s.s.service.ErrorSimulationService : DATABASE_ERROR: Failed to connect to database - Connection refused"
    """

    print("Exception type:", ask_openai_exception_type(spring_log))
