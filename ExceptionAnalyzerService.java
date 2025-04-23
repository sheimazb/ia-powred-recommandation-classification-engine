package com.windlogs.tickets.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import lombok.extern.slf4j.Slf4j;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import java.util.Map;

@Slf4j
@Service
public class ExceptionAnalyzerService {

    @Value("${exception.analyzer.api.url:http://localhost:8000}")
    private String apiUrl;

    private final RestTemplate restTemplate;

    public ExceptionAnalyzerService(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    @Data
    private static class ExceptionAnalyzerRequest {
        private String message;
        private String model = "gemini"; // Default to Gemini model
    }

    @Data
    private static class StackTraceAnalysisRequest {
        private String message;
    }

    @Data
    private static class StackTraceAnalysisResponse {
        @JsonProperty("exception_type")
        private String exceptionType;
        @JsonProperty("stack_trace")
        private String stackTrace;
        private Map<String, Object> analysis;
        private String error;
    }

    @Data
    private static class ExceptionAnalyzerResponse {
        @JsonProperty("exception_type")
        private String exceptionType;
        private float score;
        private String matchedFrom;
        @JsonProperty("model_used")
        private String modelUsed;
    }

    public String analyzeException(String logMessage) {
        try {
            log.debug("Analyzing exception message: {}", logMessage.substring(0, Math.min(logMessage.length(), 100)));
            
            String endpoint = apiUrl + "/classify-log";
            
            ExceptionAnalyzerRequest request = new ExceptionAnalyzerRequest();
            request.setMessage(logMessage);

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            HttpEntity<ExceptionAnalyzerRequest> entity = new HttpEntity<>(request, headers);

            ResponseEntity<ExceptionAnalyzerResponse> responseEntity = restTemplate.exchange(
                endpoint,
                HttpMethod.POST,
                entity,
                ExceptionAnalyzerResponse.class
            );

            ExceptionAnalyzerResponse response = responseEntity.getBody();

            if (response != null && response.getExceptionType() != null) {
                log.info("Exception analyzed: {} (using {})", response.getExceptionType(), response.getModelUsed());
                return response.getExceptionType();
            }

            log.warn("Received null response from analyzer API");
            return "UnknownException";
            
        } catch (Exception e) {
            log.error("Failed to analyze exception: {}", e.getMessage(), e);
            return "AnalysisFailedException: " + e.getMessage();
        }
    }

    public StackTraceAnalysisResponse analyzeStackTrace(String logMessage) {
        try {
            log.debug("Analyzing stack trace for message: {}", logMessage.substring(0, Math.min(logMessage.length(), 100)));
            
            String endpoint = apiUrl + "/analyze-stack-trace";
            
            StackTraceAnalysisRequest request = new StackTraceAnalysisRequest();
            request.setMessage(logMessage);

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            HttpEntity<StackTraceAnalysisRequest> entity = new HttpEntity<>(request, headers);

            ResponseEntity<StackTraceAnalysisResponse> responseEntity = restTemplate.exchange(
                endpoint,
                HttpMethod.POST,
                entity,
                StackTraceAnalysisResponse.class
            );

            StackTraceAnalysisResponse response = responseEntity.getBody();
            
            if (response != null) {
                log.info("Stack trace analyzed successfully. Found exception type: {}", response.getExceptionType());
                return response;
            }

            log.warn("Received null response from stack trace analyzer API");
            StackTraceAnalysisResponse errorResponse = new StackTraceAnalysisResponse();
            errorResponse.setError("Received null response from API");
            return errorResponse;
            
        } catch (Exception e) {
            log.error("Failed to analyze stack trace", e);
            StackTraceAnalysisResponse errorResponse = new StackTraceAnalysisResponse();
            errorResponse.setError("Analysis failed: " + e.getMessage());
            return errorResponse;
        }
    }
} 