import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Union, Optional, Tuple
import pandas as pd
from database_models import Log, Ticket, Solution
from sqlalchemy.orm import Session
import logging
import re
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LogRecommendationSystem:
    """
    Advanced log recommendation system that works with logs from any source or framework.
    Uses semantic and syntactic matching to find similar logs and potential solutions.
    """
    
    # Static patterns for log type identification
    LOG_PATTERNS = {
        # Frontend frameworks
        'react': [r'react(-dom)?\.', r'invalid hook call', r'cannot update a component'],
        'angular': [r'(angular|ng)\s?(error|warning)', r'ngError', r'Expression has changed after it was checked'],
        'vue': [r'vue\.', r'\[Vue warn\]', r'Avoid mutating a prop directly'],
        'javascript': [r'TypeError', r'ReferenceError', r'SyntaxError', r'Uncaught Error', r'Cannot read properties'],
        
        # Backend frameworks
        'django': [r'django\.', r'DoesNotExist', r'ImproperlyConfigured', r'OperationalError'],
        'flask': [r'flask\.', r'werkzeug\.', r'jinja2\.'],
        'spring': [r'org\.springframework', r'NullPointerException', r'IllegalArgumentException'],
        'rails': [r'ActionController', r'ActiveRecord', r'NameError'],
        'node': [r'node:', r'npm ERR!', r'Error: Cannot find module'],
        
        # Database related
        'sql': [r'sql\.', r'sqlalchemy\.', r'SELECT', r'INSERT', r'UPDATE', r'DELETE', r'syntax error'],
        'mongodb': [r'MongoError', r'MongoNetworkError'],
        
        # Infrastructure and deployment
        'docker': [r'docker:', r'container', r'image'],
        'kubernetes': [r'k8s\.', r'pod', r'namespace', r'deployment', r'kubectl'],
        'aws': [r'AWS\.', r'Error executing'],
        'nginx': [r'nginx:', r'failed \('],
        
        # General server errors
        'server': [r'timeout', r'memory', r'cpu', r'disk', r'space', r'resource', r'limit', r'server'],
        'network': [r'timeout', r'connection refused', r'network', r'unreachable', r'dns', r'resolve'],
        
        # .NET specific patterns
        'dotnet': [r'System\.[A-Za-z]+Exception', r'\.NET', r'Microsoft\.', r'\.dll', r'\.exe', r'\.pdb']
    }
    
    # Common log severity indicators
    SEVERITY_INDICATORS = {
        'CRITICAL': ['critical', 'fatal', 'emerg', 'alert', 'panic'],
        'ERROR': ['error', 'severe', 'err', 'failed', 'failure', 'exception'],
        'WARNING': ['warning', 'warn', 'attention'],
        'INFO': ['info', 'information', 'notice'],
        'DEBUG': ['debug', 'trace', 'verbose']
    }
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2', 
                 min_similarity_threshold: float = 0.55,  # Lowered from 0.65
                 min_term_ratio: float = 0.10):           # Lowered from 0.12
        """
        Initialize the enhanced recommendation system
        
        Args:
            model_name: The sentence transformer model to use for semantic embedding
            min_similarity_threshold: Minimum combined similarity score threshold (0.0-1.0)
            min_term_ratio: Minimum ratio of matching terms required
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.log_data = None
        self.embeddings = None
        self.min_similarity_threshold = min_similarity_threshold
        self.min_term_ratio = min_term_ratio
        self.error_indicators = self._build_error_indicators()
        self.log_processed_count = 0
        self.exception_type_weight = 0.25  # Weight for matching exception types
        self.debug_mode = False            # Enable to get detailed logging
        self.min_fallback_similarity = 0.45 # Minimum similarity for fallback results
        
    def _build_error_indicators(self) -> List[str]:
        """Build comprehensive list of error indicators for all types of logs"""
        indicators = [
            # Common error indicators across languages and frameworks
            'error', 'exception', 'failed', 'failure', 'warning', 'critical', 'fatal',
            'denied', 'invalid', 'unauthorized', 'forbidden', 'not found', 'timeout',
            'unavailable', 'unable to', 'cannot', "can't", 'could not', 'crash',
            'incorrect', 'unexpected', 'unhandled', 'rejected', 'violation',
            
            # Status and error codes
            'status', 'code', 'exit', 'return', 
            '400', '401', '403', '404', '422', '500', '502', '503', '504',
            
            # Stack trace indicators across languages
            'at ', 'caused by', 'trace:', 'stack trace:', 'traceback:', 
            'in file', 'in line', 'on line', 'at line',
            
            # JavaScript and frontend specific
            'undefined is not', 'null is not', 'cannot read property', 'is not a function',
            'is not defined', 'failed to load', 'syntax error',
            
            # Backend and database specific
            'unable to connect', 'connection refused', 'deadlock', 'timeout',
            'constraint violation', 'out of memory', 'permission denied',
            
            # .NET specific patterns
            'System.Exception', 'System.ApplicationException', 'System.ArgumentException',
            'System.NullReferenceException', 'System.InvalidOperationException',
            'System.IO.IOException', 'System.Data.SqlClient', 'required field',
            
            # Common file extensions with line numbers (indicating file locations in traces)
            '.js:', '.jsx:', '.ts:', '.tsx:', '.py:', '.java:', '.php:', '.rb:', '.go:', '.cs:', '.rs:', '.swift:',
            '.c:', '.cpp:', '.h:', '.html:', '.css:', '.sql:', '.json:', '.yml:', '.xml:',
            
            # Common infrastructure errors
            'host not found', 'no route to host', 'connection timed out',
            'disk full', 'no space left', 'too many open files',
            'certificate', 'authentication failed'
        ]
        
        # Add severity levels
        for severity_list in self.SEVERITY_INDICATORS.values():
            indicators.extend(severity_list)
        
        # Add framework-specific patterns
        for patterns in self.LOG_PATTERNS.values():
            indicators.extend([p.replace(r'\.', '.').replace('\\', '') for p in patterns])
            
        return list(set(indicators))  # Deduplicate
    
    def _normalize_log(self, log_text: str) -> str:
        """Normalize log text by removing variable parts like timestamps, IDs, and memory addresses"""
        if not log_text:
            return ""
            
        # Replace common variable patterns with placeholders
        normalized = log_text
        
        # Replace timestamps (various formats)
        normalized = re.sub(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[T ]\d{1,2}:\d{1,2}:\d{1,2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?', 
                           'TIMESTAMP', normalized)
        normalized = re.sub(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}(?: \d{1,2}:\d{1,2}:\d{1,2})?', 
                           'TIMESTAMP', normalized)
        normalized = re.sub(r'\d{1,2}:\d{1,2}:\d{1,2}(?:\.\d+)?', 
                           'TIME', normalized)
                           
        # Replace UUIDs, hashes, and IDs
        normalized = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', 
                           'UUID', normalized)
        normalized = re.sub(r'[0-9a-f]{32}', 'HASH', normalized)
        normalized = re.sub(r'[0-9a-f]{50}', 'HASH', normalized)
        normalized = re.sub(r'id=["\']\w+["\']', 'id=ID', normalized)
        
        # Replace memory addresses and line numbers (but preserve file names)
        normalized = re.sub(r'0x[0-9a-f]+', 'MEMORY_ADDR', normalized)
        normalized = re.sub(r'(\.(?:js|jsx|ts|tsx|py|java|rb|php|go|cs|rs|c|cpp|h))(:\d+)(:\d+)?', 
                           r'\1:LINE\3', normalized)
        
        # Replace specific variable values
        normalized = re.sub(r'value=["\'][^"\']*["\']', 'value=VALUE', normalized)
        normalized = re.sub(r'port \d+', 'port PORT', normalized)
        
        # Handle JSON-like structures (preserve keys but replace values)
        normalized = re.sub(r'("|\')[\w\s]+\1\s*:\s*("|\')[^"\']*\2', r'\1KEY\1: \2VALUE\2', normalized)
        
        # .NET specific normalizations
        normalized = re.sub(r'\'[^\']+\'', "'VALUE'", normalized)
        normalized = re.sub(r'"[^"]+"', '"VALUE"', normalized)
        normalized = re.sub(r'requiredField\w+', 'requiredField', normalized, flags=re.IGNORECASE)
        
        # Special handling for .NET inner exceptions
        normalized = re.sub(r'-{3,}>\s*$', '---> INNER_EXCEPTION', normalized)
        
        return normalized
        
    def _detect_log_type(self, log_text: str) -> List[str]:
        """Detect the likely sources and frameworks from a log message"""
        if not log_text:
            return []
            
        log_lower = log_text.lower()
        detected_types = []
        
        # Check for matches against our predefined patterns
        for framework, patterns in self.LOG_PATTERNS.items():
            if any(re.search(pattern, log_lower if framework != 'dotnet' else log_text) for pattern in patterns):
                detected_types.append(framework)
        
        # Special handling for .NET logs - case sensitive for System namespace
        if 'System.' in log_text and not 'dotnet' in detected_types:
            detected_types.append('dotnet')
            
        # Enhanced detection for specific exception patterns
        if re.search(r'Exception(\s*:|:?\s+in\s+)', log_text):
            if not any(t in detected_types for t in ['java', 'dotnet', 'spring']):
                # Find which language/framework this exception belongs to
                if 'at ' in log_text and ('(at' in log_text or '.java:' in log_text):
                    detected_types.append('java')
                elif 'at ' in log_text and ('.cs:' in log_text or 'System.' in log_text):
                    detected_types.append('dotnet')
                else:
                    detected_types.append('unknown_exception')
                
        return detected_types
        
    def _detect_severity(self, log_text: str) -> str: 
        """Detect the severity level of the log"""
        if not log_text:
            return "UNKNOWN"
            
        log_lower = log_text.lower()
        
        # First check for explicit severity markers
        explicit_markers = {
            'CRITICAL': [r'\[critical\]', r'critical:', r'fatal:', r'emergency:'],
            'ERROR': [r'\[error\]', r'error:', r'severe:', r'exception:'],
            'WARNING': [r'\[warn(ing)?\]', r'warning:', r'attention:'],
            'INFO': [r'\[info\]', r'info:', r'information:'],
            'DEBUG': [r'\[debug\]', r'debug:', r'trace:']
        }
        
        for severity, markers in explicit_markers.items():
            if any(re.search(marker, log_lower) for marker in markers):
                return severity
        
        # Then check for keyword indicators
        for severity, indicators in self.SEVERITY_INDICATORS.items():
            if any(indicator in log_lower for indicator in indicators):
                return severity
        
        # Special detection for exceptions - if we have any exception, it's probably an ERROR
        if 'exception' in log_lower or 'Exception' in log_text or 'Error' in log_text:
            return "ERROR"
                
        return "UNKNOWN"
    
    def _extract_error_code(self, log_text: str) -> Optional[str]:
        """Extract error codes from the log text"""
        if not log_text:
            return None
            
        # Common error code patterns
        patterns = [
            r'error(?:\s+code)?[\s:=]+([A-Z0-9_]{3,})',  # ERROR_CODE_FORMAT
            r'code[\s:=]+([A-Z0-9_]{3,})',               # Code: ERROR_CODE
            r'status[\s:=]+(\d{3})',                    # Status: 404
            r'(\d{3})(?:\s+\w+)?(?:\s+error)',           # 404 not found error
            r'exit code (\d+)',                          # Exit code 1
            r'HRESULT: 0x([0-9A-F]+)',                   # .NET HRESULT
            r'error code (0x[0-9A-F]+)',                 # Hex error codes
            r'error \((\d+)\)'                           # error (123)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, log_text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Special handling for HTTP status codes embedded in text
        status_match = re.search(r'\b(4\d{2}|5\d{2})\b', log_text)
        if status_match and any(term in log_text.lower() for term in ['http', 'status', 'response', 'request']):
            return status_match.group(1)
                
        return None
        
    def _extract_stack_trace(self, log_text: str) -> Optional[str]:
        """Extract stack trace from log if present"""
        if not log_text:
            return None
            
        # Common stack trace indicators
        trace_starters = [
            'at ', 'Traceback', 'Stack trace:', 'Caused by:', 
            'Exception in thread', 'Call stack:', 'System.Exception',
            'System.ApplicationException', '---> System.'
        ]
        
        # Check if any trace indicators exist
        trace_start = None
        for starter in trace_starters:
            pos = log_text.find(starter)
            if pos != -1:
                trace_start = pos
                break
        
        if trace_start is not None:
            # Extract from trace start to the end (or reasonable length)
            trace = log_text[trace_start:trace_start + 2000]  # Limit length
            return trace
            
        return None
    
    def _extract_exception_type(self, log_text: str) -> Optional[str]:
        """Extract exception type from log message"""
        if not log_text:
            return None
        
        # Various patterns for exception types
        patterns = [
            # .NET exceptions
            r'(System\.[A-Za-z0-9\.]+Exception)',
            
            # Java exceptions
            r'(java\.[a-z0-9\.]+Exception)',
            r'(org\.[a-z0-9\.]+Exception)',
            r'(com\.[a-z0-9\.]+Exception)',
            
            # General exceptions
            r'([A-Z][A-Za-z0-9]+Exception)',
            r'Exception: ([A-Z][A-Za-z0-9]+Error)',
            
            # JavaScript errors
            r'(TypeError|SyntaxError|ReferenceError|RangeError|URIError|EvalError)',
            
            # Python exceptions
            r'(ValueError|TypeError|KeyError|IndexError|AttributeError|ImportError|RuntimeError|NameError)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, log_text)
            if match:
                return match.group(1)
                
        return None
        
    def _get_inner_exception_types(self, log_text: str) -> List[str]:
        """Extract all exception types including inner exceptions"""
        if not log_text:
            return []
        
        # Find all exceptions in a .NET style trace with inner exceptions
        exceptions = []
        
        # First try to get the primary exception
        primary = self._extract_exception_type(log_text)
        if primary:
            exceptions.append(primary)
        
        # Look for inner exceptions (---> marker in .NET)
        inner_sections = re.findall(r'--->\s+([^\r\n]+)', log_text)
        for section in inner_sections:
            inner_exc = self._extract_exception_type(section)
            if inner_exc:
                exceptions.append(inner_exc)
        
        return exceptions
    
    def build_index_from_db(self, db: Session):
        """Build search index from logs in database with enhanced processing"""
        try:
            # First, check if we can query logs table
            total_logs = db.query(Log).count()
            logger.info(f"Total logs in database: {total_logs}")
            
            # Get all logs that have tickets
            logs = db.query(Log).join(Ticket).all()
            logger.info(f"Found {len(logs)} logs with tickets")
            
            if not logs:
                logger.warning("No logs found in database with tickets")
                return
                
            start_time = datetime.now()
            log_data = []
            skipped_logs = 0
            
            for log in logs:
                try:
                    # Process each log entry
                    log_message = log.description or ""
                    
                    # Add stack trace if available
                    if log.stack_trace:
                        log_message += " " + log.stack_trace
                        
                    # Skip if log message is empty after processing
                    if not log_message.strip():
                        skipped_logs += 1
                        continue
                        
                    # Extract key elements automatically
                    detected_types = self._detect_log_type(log_message)
                    severity = log.severity or self._detect_severity(log_message)
                    error_code = log.error_code or self._extract_error_code(log_message)
                    stack_trace = log.stack_trace or self._extract_stack_trace(log_message)
                    
                    # Extract exception type(s)
                    exception_types = []
                    if log.exception_type:
                        exception_types.append(log.exception_type)
                    else:
                        exception_types = self._get_inner_exception_types(log_message)
                    
                    # Create additional normalized version for better matching
                    normalized_message = self._normalize_log(log_message)
                    
                    # Add error info and custom message if available
                    error_info = []
                    if log.type:
                        error_info.append(log.type)
                    if error_code:
                        error_info.append(f"code: {error_code}")
                    if exception_types:
                        error_info.extend(exception_types)
                    
                    error_type = " ".join(error_info)
                    
                    # Include custom message
                    if log.custom_message:
                        log_message += " " + log.custom_message
                    
                    # Combine all information for embedding with emphasis on important parts
                    embedding_text = f"{error_type} {' '.join(exception_types) if exception_types else ''} {log_message}"
                    
                    # Process the log entry
                    log_entry = {
                        'log_id': log.id,
                        'log_message': log_message,
                        'normalized_message': normalized_message,
                        'error_type': error_type,
                        'error_code': error_code,
                        'exception_types': exception_types,
                        'detected_types': detected_types,
                        'embedding_text': embedding_text,
                        'ticket_id': log.ticket.id,
                        'solution_title': '',
                        'solution_content': 'No solution available yet',
                        'severity': severity
                    }
                    
                    # Add solution if available
                    if hasattr(log.ticket, 'solution') and log.ticket.solution:
                        log_entry.update({
                            'solution_title': log.ticket.solution.title or '',
                            'solution_content': log.ticket.solution.content or '',
                            'solution_author_user_id': str(log.ticket.solution.author_user_id) if log.ticket.solution.author_user_id is not None else ''
                        })
                    
                    log_data.append(log_entry)
                    
                except Exception as e:
                    logger.error(f"Error processing log {log.id}: {str(e)}")
                    skipped_logs += 1
            
            if not log_data:
                logger.warning("No valid log data found after processing")
                return
             
            self.log_processed_count = len(log_data)
            logger.info(f"Successfully processed {self.log_processed_count} logs, skipped {skipped_logs} logs")
            self.log_data = pd.DataFrame(log_data)
            
            # Create embeddings for all logs using the combined text
            embedding_texts = self.log_data['embedding_text'].tolist()
            logger.info(f"Creating embeddings for {len(embedding_texts)} logs...")
            self.embeddings = self.model.encode(embedding_texts, show_progress_bar=True)
            
            # Build search index
            dimension = self.embeddings.shape[1]
            logger.info(f"Building FAISS index with dimension {dimension}...")
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.embeddings.astype('float32'))
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Successfully built index with {len(embedding_texts)} logs in {duration:.2f} seconds")
            
            # Log framework distribution for analysis
            if not self.log_data.empty and 'detected_types' in self.log_data:
                framework_counts = {}
                for types in self.log_data['detected_types']:
                    for framework in types:
                        framework_counts[framework] = framework_counts.get(framework, 0) + 1
                framework_summary = ", ".join([f"{fw}: {count}" for fw, count in sorted(framework_counts.items(), key=lambda x: x[1], reverse=True)])
                logger.info(f"Framework distribution: {framework_summary}")
            
        except Exception as e:
            logger.error(f"Error building index: {str(e)}")
            raise
        
    def find_similar_logs(self, query_log: str, k: int = 20) -> Dict:
        """
        Find similar logs and their solutions
        
        Args:
            query_log: The log message to find similar logs for
            k: Maximum number of similar logs to return
            
        Returns:
            Dictionary with similar logs and a message
        """
        result = {
            "similar_logs": [],
            "query_analysis": {},
            "message": ""
        }
        
        # Handle empty index or database
        if self.index is None or self.log_data is None or len(self.log_data) == 0:
            result["message"] = "No logs available in the database"
            return result
            
        # Basic validation of query
        query_log = query_log.strip()
        if not query_log:
            result["message"] = "Empty log message provided"
            return result
        
        try:
            # Analyze and normalize input query
            query_lower = query_log.lower()
            
            # Detect log characteristics for better matching
            detected_types = self._detect_log_type(query_log)
            detected_severity = self._detect_severity(query_log)
            error_code = self._extract_error_code(query_log)
            exception_types = self._get_inner_exception_types(query_log)
            
            # Extract and clean meaningful terms
            query_terms = set([term.strip(',.;()[]{}"\'') for term in query_log.lower().split() 
                         if len(term.strip(',.;()[]{}"\'')) > 2])
            
            # Create normalized version for better matching
            normalized_query = self._normalize_log(query_log)
            
            # Add log analysis to results
            result["query_analysis"] = {
                "detected_types": detected_types,
                "severity": detected_severity,
                "error_code": error_code,
                "exception_types": exception_types,
                "term_count": len(query_terms),
                "normalized_query": normalized_query
            }
            
            # Debug logging if enabled
            if self.debug_mode:
                logger.info(f"Query analysis: {result['query_analysis']}")
            
            # Verify it's likely a log message
            contains_error_terms = any(indicator in query_lower for indicator in self.error_indicators)
            is_likely_log = bool(contains_error_terms or detected_types or error_code or exception_types or
                                detected_severity != "UNKNOWN")
            
            # NEW: Return early if not likely a log message with specific message
            if not is_likely_log:
                result["message"] = "Input does not appear to be an error log message"
                return result
            
            # Pre-construct regex patterns for exact exception matching - faster than iterating
            exception_regexes = []
            if exception_types:
                for ex_type in exception_types:
                    # Clean regex pattern to avoid regex errors
                    safe_pattern = re.escape(ex_type)
                    exception_regexes.append(re.compile(safe_pattern))
            
            # Convert input log to embedding
            query_embedding = self.model.encode([query_log])[0]
            
            # Search for similar logs - get more candidates initially for filtering
            k_search = min(len(self.log_data), max(k * 5, 100))  # Get more candidates for filtering
            distances, indices = self.index.search(
                query_embedding.reshape(1, -1).astype('float32'), 
                k_search
            )
            
            similar_logs = []
            
            # First pass to identify exact matches (if any)
            exact_matches = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.log_data):
                    log = self.log_data.iloc[idx]
                    
                    # Check for exact matching exception types
                    if exception_types and 'exception_types' in log and log['exception_types']:
                        if any(exc in exception_types for exc in log['exception_types']):
                            exact_matches.append((dist, idx, True))
                            continue
                    
                    # Check for exact matching error codes
                    if error_code and log['error_code'] and error_code == log['error_code']:
                        exact_matches.append((dist, idx, True))
                        continue
                        
                    # Add to normal processing
                    exact_matches.append((dist, idx, False))
            
            # Prioritize exact matches or process all
            match_list = exact_matches
            
            for dist, idx, is_exact_match in match_list:
                if idx < len(self.log_data):
                    log = self.log_data.iloc[idx]
                    
                    # Calculate base similarity score (vector similarity)
                    vector_similarity = 1 / (1 + dist)  # Convert distance to similarity score
                    
                    # Calculate term overlap using both original and normalized message
                    log_terms = set([term.strip(',.;()[]{}"\'') for term in log['log_message'].lower().split() 
                                 if len(term.strip(',.;()[]{}"\'')) > 2])
                    
                    # Calculate matching terms
                    matching_terms = query_terms.intersection(log_terms)
                    term_ratio = len(matching_terms) / max(len(query_terms), 1)
                    
                    # Check exact error code match (if both have error codes)
                    error_code_match = 0.0
                    if error_code and log['error_code'] and error_code == log['error_code']:
                        error_code_match = 0.3
                    
                    # Enhanced type matching
                    type_match = 0.0
                    if detected_types and log['detected_types']:
                        common_types = set(detected_types).intersection(set(log['detected_types']))
                        if common_types:
                            type_match = 0.2 * (len(common_types) / max(len(detected_types), 1))
                    
                    # Match severity levels
                    severity_match = 0.0
                    if detected_severity == log['severity'] and detected_severity != "UNKNOWN":
                        severity_match = 0.1
                    
                    # Normalized message similarity
                    normalized_match = 0.0
                    if normalized_query and log['normalized_message']:
                        # Check for shared normalized patterns
                        norm_query_terms = set(normalized_query.lower().split())
                        norm_log_terms = set(log['normalized_message'].lower().split())
                        norm_overlap = len(norm_query_terms.intersection(norm_log_terms))
                        normalized_match = 0.15 * (norm_overlap / max(len(norm_query_terms), 1))
                    
                    # Exception type matching - higher weight for this
                    exception_match = 0.0
                    if exception_types and 'exception_types' in log and log['exception_types']:
                        common_exceptions = set(exception_types).intersection(set(log['exception_types']))
                        if common_exceptions:
                            # High weight for exception matches - these are very specific
                            exception_match = self.exception_type_weight * (len(common_exceptions) / max(len(exception_types), 1))
                    
                    # Boost for exact matches
                    exact_match_boost = 0.2 if is_exact_match else 0.0
                    
                    # Final weighted similarity score
                    final_score = (
                        vector_similarity * 0.30 +      # Vector/semantic similarity - slightly reduced
                        term_ratio * 0.20 +             # Term overlap - slightly reduced
                        error_code_match +              # Error code exact match
                        type_match +                    # Framework/source type match
                        severity_match +                # Severity level match
                        normalized_match +              # Normalized pattern match
                        exception_match +               # Exception type match (NEW)
                        exact_match_boost               # Boost for exact matches (NEW)
                    )
                    
                    # Modified thresholds for more results - prioritize showing something
                    use_min_similarity = self.min_similarity_threshold * 0.9 if not similar_logs else self.min_similarity_threshold
                    use_min_term_ratio = self.min_term_ratio * 0.8 if not similar_logs else self.min_term_ratio
                    
                    # Lower thresholds for exact matches
                    if is_exact_match:
                        use_min_similarity *= 0.7
                        use_min_term_ratio *= 0.5
                    
                    # For dotnet logs, be more lenient with the term matching
                    if 'dotnet' in detected_types or any('System.' in ex_type for ex_type in exception_types):
                        use_min_term_ratio *= 0.75
                    
                    # Use our modified minimum thresholds to filter results
                    if final_score >= use_min_similarity and term_ratio >= use_min_term_ratio:
                        # Skip logs without solutions
                        if not log['solution_content'] or log['solution_content'] == 'No solution available yet':
                            continue
                            
                        match_details = []
                        if exception_match > 0:
                            match_details.append("matching exception types")
                        if error_code_match > 0:
                            match_details.append("exact error code")
                        if type_match > 0:
                            match_details.append("similar framework")
                        if normalized_match > term_ratio:
                            match_details.append("similar patterns")
                        
                        similar_logs.append({
                            'log_id': int(log['log_id']),
                            'log_message': log['log_message'],
                            'ticket_id': int(log['ticket_id']),
                            'error_type': log['error_type'],
                            'detected_types': log['detected_types'],
                            'exception_types': log['exception_types'] if 'exception_types' in log else [],
                            'solution_title': log['solution_title'],
                            'solution_content': log['solution_content'],
                            'solution_author_user_id': str(log['solution_author_user_id']) if 'solution_author_user_id' in log and log['solution_author_user_id'] is not None else '',
                            'severity': log['severity'],
                            'similarity': f"{final_score:.2%}",
                            'term_match': f"{term_ratio:.2%}",
                            'match_details': ", ".join(match_details) if match_details else "semantic similarity"
                        })
            
            # Sort by similarity score and limit results
            similar_logs.sort(key=lambda x: float(x['similarity'].rstrip('%')), reverse=True)
            similar_logs = similar_logs[:k]
            
            if not similar_logs:
                # Fall back to returning the closest matches by vector similarity if nothing else matched
                if len(indices[0]) > 0:
                    top_indices = indices[0][:min(3, len(indices[0]))]
                    for idx in top_indices:
                        if idx < len(self.log_data):
                            log = self.log_data.iloc[idx]
                            vector_similarity = 1 / (1 + distances[0][np.where(indices[0] == idx)[0][0]])
                            
                            # NEW: Only include fallback results if they meet a minimum absolute threshold
                            # Even for low confidence matches, we want some basic relevance
                            # Also skip logs without solutions
                            if vector_similarity < self.min_fallback_similarity or \
                               not log['solution_content'] or \
                               log['solution_content'] == 'No solution available yet':
                                continue
                                
                            similar_logs.append({
                                'log_id': int(log['log_id']),
                                'log_message': log['log_message'],
                                'ticket_id': int(log['ticket_id']),
                                'error_type': log['error_type'],
                                'detected_types': log['detected_types'],
                                'solution_title': log['solution_title'],
                                'solution_content': log['solution_content'],
                                'solution_author_user_id': str(log['solution_author_user_id']) if 'solution_author_user_id' in log and log['solution_author_user_id'] is not None else '',
                                'severity': log['severity'],
                                'similarity': f"{vector_similarity:.2%}",
                                'term_match': 'N/A',
                                'match_details': "closest semantic match (low confidence)"
                            })
                            
                    if similar_logs:
                        result["similar_logs"] = similar_logs
                        result["message"] = f"Found {len(similar_logs)} low confidence matches. Consider adjusting your query."
                    else:
                        result["message"] = "No sufficiently similar logs found. The input may not be an error log."
                else:
                    result["message"] = "No sufficiently similar logs found. The input may not be an error log."
                return result
            
            result["similar_logs"] = similar_logs
            result["message"] = f"Found {len(similar_logs)} relevant similar logs from a database of {self.log_processed_count} logs."
            return result
            
        except Exception as e:
            logger.error(f"Error finding similar logs: {str(e)}")
            result["message"] = f"Error processing query: {str(e)}"
            return result
    
    def set_similarity_threshold(self, threshold: float):
        """Update the minimum similarity threshold
        
        Args:
            threshold: New threshold value between 0.0 and 1.0
        """
        if 0.0 <= threshold <= 1.0:
            self.min_similarity_threshold = threshold
            logger.info(f"Updated similarity threshold to {threshold}")
        else:
            logger.error(f"Invalid threshold value: {threshold}. Must be between 0.0 and 1.0")
            
    def set_term_ratio_threshold(self, threshold: float):
        """Update the minimum term ratio threshold
        
        Args:
            threshold: New threshold value between 0.0 and 1.0
        """
        if 0.0 <= threshold <= 1.0:
            self.min_term_ratio = threshold
            logger.info(f"Updated term ratio threshold to {threshold}")
        else:
            logger.error(f"Invalid threshold value: {threshold}. Must be between 0.0 and 1.0")
            
    def set_fallback_similarity_threshold(self, threshold: float):
        """Update the minimum fallback similarity threshold
        
        Args:
            threshold: New threshold value between 0.0 and 1.0
        """
        if 0.0 <= threshold <= 1.0:
            self.min_fallback_similarity = threshold
            logger.info(f"Updated fallback similarity threshold to {threshold}")
        else:
            logger.error(f"Invalid threshold value: {threshold}. Must be between 0.0 and 1.0")
            
    def set_debug_mode(self, enabled: bool = True):
        """Enable or disable debug mode for detailed logging
        
        Args:
            enabled: Whether to enable debug mode
        """
        self.debug_mode = enabled
        logger.info(f"Debug mode {'enabled' if enabled else 'disabled'}")
        
    def debug_similar_logs(self, query_log: str, k: int = 5) -> Dict:
        """
        Find and debug similar logs with detailed matching information
        
        Args:
            query_log: The log message to find similar logs for
            k: Maximum number of similar logs to return
            
        Returns:
            Dictionary with similar logs, message, and detailed debug info
        """
        # Store previous debug setting and enable
        prev_debug = self.debug_mode
        self.debug_mode = True
        
        # Get standard results
        result = self.find_similar_logs(query_log, k)
        
        # Add detailed debug information
        result["debug_info"] = {
            "thresholds": {
                "similarity": self.min_similarity_threshold,
                "term_ratio": self.min_term_ratio
            },
            "query": {
                "length": len(query_log),
                "word_count": len(query_log.split()),
                "has_exception": "Exception" in query_log,
                "has_error_code": bool(self._extract_error_code(query_log)),
                "normalized": self._normalize_log(query_log)
            },
            "system_info": self.get_system_info()
        }
        
        # If we have matches, calculate detailed stats
        if result["similar_logs"]:
            similarity_scores = [float(log["similarity"].rstrip('%'))/100 for log in result["similar_logs"]]
            result["debug_info"]["matches"] = {
                "count": len(result["similar_logs"]),
                "avg_similarity": sum(similarity_scores) / len(similarity_scores),
                "max_similarity": max(similarity_scores),
                "min_similarity": min(similarity_scores),
            }
        
        # Restore previous debug setting
        self.debug_mode = prev_debug
        
        return result
    
    def get_system_info(self) -> Dict:
        """Get information about the recommendation system state"""
        return {
            "model_name": self.model.__class__.__name__,
            "logs_processed": self.log_processed_count if self.log_data is not None else 0,
            "index_built": self.index is not None,
            "similarity_threshold": self.min_similarity_threshold,
            "term_ratio_threshold": self.min_term_ratio,
            "fallback_similarity_threshold": self.min_fallback_similarity,
            "embedding_dimension": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "detected_frameworks": list(set(sum(self.log_data['detected_types'].tolist(), []))) if self.log_data is not None else [],
            "memory_usage_mb": self.embeddings.nbytes / 1024 / 1024 if self.embeddings is not None else 0
        }