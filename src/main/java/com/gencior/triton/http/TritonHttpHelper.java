package com.gencior.triton.http;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

import javax.net.ssl.SSLContext;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.gencior.triton.config.TritonClientConfig;
import com.gencior.triton.exceptions.TritonInferException;

/**
 * Internal helper for executing HTTP requests against the Triton Inference Server REST API.
 * Handles base URL construction, timeout management, TLS configuration, and error mapping.
 *
 * @author sachachoumiloff
 * @since 1.1.0
 */
final class TritonHttpHelper {

    private static final Logger LOG = LoggerFactory.getLogger(TritonHttpHelper.class);

    private final TritonClientConfig config;
    private final HttpClient httpClient;
    private final String baseUrl;
    private final ObjectMapper objectMapper;
    private final Duration timeout;

    TritonHttpHelper(TritonClientConfig config) {
        this.config = config;
        this.objectMapper = new ObjectMapper();
        this.timeout = Duration.ofMillis(config.getDefaultTimeoutMs());
        this.baseUrl = buildBaseUrl(config);
        this.httpClient = buildHttpClient(config);
    }

    /**
     * Sends a GET request and returns the parsed JSON body.
     *
     * @param path the API path (e.g. "/v2/health/live")
     * @return the parsed JsonNode response body
     */
    JsonNode sendGet(String path) {
        return sendRequest(buildGet(path));
    }

    /**
     * Sends a GET request and returns the raw HTTP status code.
     * Does not throw on non-2xx responses.
     *
     * @param path the API path
     * @return the HTTP status code
     */
    int sendGetStatus(String path) {
        try {
            HttpRequest request = buildGet(path);
            long start = System.currentTimeMillis();
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            long elapsed = System.currentTimeMillis() - start;
            LOG.debug("GET {} -> {} ({}ms)", path, response.statusCode(), elapsed);
            return response.statusCode();
        } catch (Exception e) {
            LOG.debug("GET {} failed: {}", path, e.getMessage());
            return -1;
        }
    }

    /**
     * Sends a POST request with an empty body and returns the parsed JSON body.
     *
     * @param path the API path
     * @return the parsed JsonNode response body
     */
    JsonNode sendPostEmpty(String path) {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + path))
                .timeout(timeout)
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString("{}"))
                .build();
        return sendRequest(request);
    }

    /**
     * Sends a POST request with an empty body. Only checks for success (2xx).
     * Throws on error responses.
     *
     * @param path the API path
     */
    void sendPostEmptyVoid(String path) {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + path))
                .timeout(timeout)
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString("{}"))
                .build();
        try {
            long start = System.currentTimeMillis();
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            long elapsed = System.currentTimeMillis() - start;
            LOG.debug("POST {} -> {} ({}ms)", path, response.statusCode(), elapsed);
            checkResponse(response, path);
        } catch (TritonInferException e) {
            throw e;
        } catch (Exception e) {
            throw new TritonInferException("HTTP POST " + path + " failed: " + e.getMessage(), e);
        }
    }

    /**
     * Sends a POST request with a binary body (for inference with binary extension).
     *
     * @param path the API path
     * @param jsonHeader the JSON header portion
     * @param binaryData the concatenated binary tensor data
     * @return the raw HTTP response
     */
    HttpResponse<byte[]> sendPostBinary(String path, String jsonHeader, byte[] binaryData) {
        byte[] jsonBytes = jsonHeader.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        byte[] body = new byte[jsonBytes.length + binaryData.length];
        System.arraycopy(jsonBytes, 0, body, 0, jsonBytes.length);
        System.arraycopy(binaryData, 0, body, jsonBytes.length, binaryData.length);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + path))
                .timeout(timeout)
                .header("Content-Type", "application/octet-stream")
                .header("Inference-Header-Content-Length", String.valueOf(jsonBytes.length))
                .POST(HttpRequest.BodyPublishers.ofByteArray(body))
                .build();
        try {
            long start = System.currentTimeMillis();
            HttpResponse<byte[]> response = httpClient.send(request, HttpResponse.BodyHandlers.ofByteArray());
            long elapsed = System.currentTimeMillis() - start;
            LOG.debug("POST {} (binary) -> {} ({}ms)", path, response.statusCode(), elapsed);
            if (response.statusCode() >= 400) {
                String errorBody = new String(response.body(), java.nio.charset.StandardCharsets.UTF_8);
                throwTritonError(response.statusCode(), path, errorBody);
            }
            return response;
        } catch (TritonInferException e) {
            throw e;
        } catch (Exception e) {
            throw new TritonInferException("HTTP POST " + path + " failed: " + e.getMessage(), e);
        }
    }

    /**
     * Sends a POST request and returns the InputStream for SSE streaming.
     *
     * @param path the API path
     * @param jsonBody the JSON request body
     * @return the HTTP response with InputStream body
     */
    HttpResponse<java.io.InputStream> sendPostStream(String path, String jsonBody) {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + path))
                .timeout(timeout)
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                .build();
        try {
            HttpResponse<java.io.InputStream> response = httpClient.send(request,
                    HttpResponse.BodyHandlers.ofInputStream());
            if (response.statusCode() >= 400) {
                String errorBody = new String(response.body().readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
                throwTritonError(response.statusCode(), path, errorBody);
            }
            return response;
        } catch (TritonInferException e) {
            throw e;
        } catch (Exception e) {
            throw new TritonInferException("HTTP POST " + path + " (stream) failed: " + e.getMessage(), e);
        }
    }

    ObjectMapper getObjectMapper() {
        return objectMapper;
    }

    String getBaseUrl() {
        return baseUrl;
    }

    HttpClient getHttpClient() {
        return httpClient;
    }

    // ========== Internal ==========

    private HttpRequest buildGet(String path) {
        return HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + path))
                .timeout(timeout)
                .GET()
                .build();
    }

    private JsonNode sendRequest(HttpRequest request) {
        try {
            long start = System.currentTimeMillis();
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            long elapsed = System.currentTimeMillis() - start;
            LOG.debug("{} {} -> {} ({}ms)", request.method(), request.uri().getPath(),
                    response.statusCode(), elapsed);
            checkResponse(response, request.uri().getPath());
            String body = response.body();
            if (body == null || body.isBlank()) {
                return objectMapper.createObjectNode();
            }
            return objectMapper.readTree(body);
        } catch (TritonInferException e) {
            throw e;
        } catch (Exception e) {
            throw new TritonInferException("HTTP request to " + request.uri() + " failed: " + e.getMessage(), e);
        }
    }

    private void checkResponse(HttpResponse<String> response, String path) {
        if (response.statusCode() >= 400) {
            throwTritonError(response.statusCode(), path, response.body());
        }
    }

    private void throwTritonError(int statusCode, String path, String body) {
        String message = "HTTP " + statusCode + " on " + path;
        if (body != null && !body.isBlank()) {
            try {
                JsonNode errorJson = objectMapper.readTree(body);
                if (errorJson.has("error")) {
                    message = errorJson.get("error").asText();
                }
            } catch (Exception ignored) {
                message += ": " + body;
            }
        }
        throw new TritonInferException(message);
    }

    private static String buildBaseUrl(TritonClientConfig config) {
        String url = config.getUrl();
        String scheme = config.isTlsEnabled() ? "https" : "http";
        if (url.startsWith("http://") || url.startsWith("https://")) {
            return url;
        }
        return scheme + "://" + url;
    }

    private static HttpClient buildHttpClient(TritonClientConfig config) {
        HttpClient.Builder builder = HttpClient.newBuilder()
                .connectTimeout(Duration.ofMillis(config.getDefaultTimeoutMs()));

        if (config.isTlsEnabled()) {
            try {
                SSLContext sslContext = SslContextFactory.createSslContext(config);
                builder.sslContext(sslContext);
            } catch (Exception e) {
                throw new TritonInferException("Failed to create SSL context: " + e.getMessage(), e);
            }
        }
        return builder.build();
    }
}
