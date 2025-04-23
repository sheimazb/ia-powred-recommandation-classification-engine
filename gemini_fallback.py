import google.generativeai as genai
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure the Gemini API
genai.configure(api_key="AIzaSyCPuumw4XJKmaM0yz051sx8hrFAneN8YFE")

def extract_stack_trace(log_message):
    """Extract stack trace from log message."""
    if not log_message:
        return None
        
    lines = log_message.split('\n')
    stack_trace = []
    started = False
    
    for line in lines:
        # Detect start of stack trace
        if ('Exception:' in line or 'Error:' in line) and not started:
            started = True
            stack_trace.append(line.strip())
        # Collect stack trace lines
        elif started and ('at ' in line or 'Caused by:' in line or 'Suppressed:' in line):
            stack_trace.append(line.strip())
            
    return '\n'.join(stack_trace) if stack_trace else None

def analyze_stack_trace(log_message):
    """Analyze stack trace and provide comprehensive information using Gemini."""
    
    stack_trace = extract_stack_trace(log_message)
    if not stack_trace:
        return {
            "error": "No stack trace found in the log message",
            "stack_trace": None,
            "analysis": None
        }

    prompt = f"""
You are an expert Java/Spring Boot developer. Analyze this stack trace and provide a comprehensive analysis.
Break down your analysis into these parts:
1. Root Exception: The primary exception that was thrown
2. Cause: The underlying reason for the exception
3. Location: Where in the code the exception occurred (class, method, line)
4. Chain: If there are nested exceptions, list them in order
5. Recommendation: Brief suggestion on how to fix or investigate the issue

Stack trace:
{stack_trace}

Format your response as JSON with these exact keys:
{{
    "root_exception": "name of primary exception",
    "cause": "brief explanation of what caused it",
    "location": "where it occurred",
    "exception_chain": ["list of nested exceptions if any"],
    "recommendation": "how to fix or investigate"
}}
"""

    try:
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Generate response
        response = model.generate_content(prompt)
        
        # Log the response
        logger.info(f"Received stack trace analysis from Gemini API")
        
        return {
            "error": None,
            "stack_trace": stack_trace,
            "analysis": response.text
        }
    except Exception as e:
        logger.error(f"Error analyzing stack trace: {str(e)}", exc_info=True)
        return {
            "error": f"Analysis failed: {str(e)}",
            "stack_trace": stack_trace,
            "analysis": None
        }

def ask_gemini_exception_type(log_message):
    """Use Google's Gemini to identify the specific exception type from a log message."""
    
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
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Generate response
        response = model.generate_content(prompt)
        
        # Log the response
        logger.info(f"Received response from Gemini API: {response.text}")
        
        # Get the exception type from response
        exception_type = response.text.strip()
        
        # Ensure it follows naming convention
        if not (exception_type.endswith('Exception') or exception_type.endswith('Error')):
            exception_type += 'Exception'
            
        return exception_type
    except Exception as e:
        # Log the detailed error
        logger.error(f"Error in Gemini API call: {str(e)}", exc_info=True)
        return f"AnalysisFailedException: {str(e)}"

# Example usage
if __name__ == "__main__":
    test_logs = [
        """
        2024-03-17 10:15:23.456 ERROR [service-name,trace=abc123] --- [http-nio-8080-exec-1] c.e.s.UserService : Failed to process user request
        org.springframework.dao.DataAccessException: Could not update user profile
            at com.example.service.UserServiceImpl.updateProfile(UserServiceImpl.java:125)
            at com.example.controller.UserController.updateUser(UserController.java:87)
        Caused by: org.hibernate.StaleObjectStateException: Row was updated or deleted by another transaction
            at org.hibernate.persister.entity.AbstractEntityPersister.check(AbstractEntityPersister.java:2567)
            at org.hibernate.persister.entity.AbstractEntityPersister.update(AbstractEntityPersister.java:3390)
        """,
        
        """
        2024-03-17 11:20:45.789 ERROR [order-service] --- [async-executor-3] c.e.s.OrderProcessor : Order processing failed
        java.lang.NullPointerException: Cannot invoke "com.example.model.Order.getItems()" because "order" is null
            at com.example.service.OrderProcessor.processOrder(OrderProcessor.java:56)
            at com.example.service.OrderProcessor.lambda$processAsync$0(OrderProcessor.java:30)
        """
    ]
    
    for log in test_logs:
        print("\nTesting log analysis:")
        print("-" * 50)
        print(f"Log message:\n{log}\n")
        
        # Get exception type
        exception_type = ask_gemini_exception_type(log)
        print(f"Exception Type: {exception_type}\n")
        
        # Get comprehensive analysis
        analysis = analyze_stack_trace(log)
        print("Stack Trace Analysis:")
        print(analysis["analysis"] if analysis["analysis"] else analysis["error"]) 