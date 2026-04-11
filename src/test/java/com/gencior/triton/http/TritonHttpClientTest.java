package com.gencior.triton.http;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.List;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import com.gencior.triton.config.TritonClientConfig;
import com.gencior.triton.core.pojo.TritonModelConfig;
import com.gencior.triton.core.pojo.TritonModelMetadata;
import com.gencior.triton.core.pojo.TritonModelStatistics;
import com.gencior.triton.core.pojo.TritonRepositoryIndex;
import com.gencior.triton.core.pojo.TritonServerMetadata;
import com.gencior.triton.exceptions.TritonInferException;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

/**
 * Unit tests for TritonHttpClient using a JDK built-in mock HTTP server.
 */
class TritonHttpClientTest {

    private static HttpServer mockServer;
    private static TritonHttpClient client;

    @BeforeAll
    static void setUp() throws IOException {
        mockServer = HttpServer.create(new InetSocketAddress(0), 0);
        registerHandlers(mockServer);
        mockServer.start();

        int port = mockServer.getAddress().getPort();
        TritonClientConfig config = new TritonClientConfig.Builder("localhost:" + port)
                .timeout(5000)
                .build();
        client = new TritonHttpClient(config);
    }

    @AfterAll
    static void tearDown() throws Exception {
        if (client != null) client.close();
        if (mockServer != null) mockServer.stop(0);
    }

    // ========== Health ==========

    @Test
    void isServerLive_returnsTrue() {
        assertTrue(client.isServerLive());
    }

    @Test
    void isServerReady_returnsTrue() {
        assertTrue(client.isServerReady());
    }

    // ========== Server Metadata ==========

    @Test
    void getServerMetadata_parsesCorrectly() {
        TritonServerMetadata meta = client.getServerMetadata();
        assertEquals("triton", meta.getName());
        assertEquals("2.42.0", meta.getVersion());
        assertEquals(2, meta.getExtensions().size());
        assertTrue(meta.supportsExtension("classification"));
    }

    // ========== Model Ready ==========

    @Test
    void isModelReady_returnsTrue() {
        assertTrue(client.isModelReady("identity_fp32"));
    }

    @Test
    void isModelReady_withVersion_returnsTrue() {
        assertTrue(client.isModelReady("identity_fp32", "1"));
    }

    @Test
    void isModelReady_unknownModel_returnsFalse() {
        assertFalse(client.isModelReady("nonexistent_model"));
    }

    // ========== Model Metadata ==========

    @Test
    void getModelMetadata_parsesCorrectly() {
        TritonModelMetadata meta = client.getModelMetadata("identity_fp32", "1");
        assertEquals("identity_fp32", meta.getName());
        assertEquals("python", meta.getPlatform());
        assertEquals(1, meta.getInputs().size());
        assertEquals("INPUT0", meta.getInputs().get(0).getName());
        assertEquals("FP32", meta.getInputs().get(0).getDatatype());
        assertEquals(1, meta.getOutputs().size());
        assertEquals("OUTPUT0", meta.getOutputs().get(0).getName());
    }

    // ========== Model Config ==========

    @Test
    void getModelConfig_parsesCorrectly() {
        TritonModelConfig config = client.getModelConfig("identity_fp32");
        assertEquals("identity_fp32", config.getName());
        assertEquals("python", config.getBackend());
        assertEquals(8, config.getMaxBatchSize());
    }

    @Test
    void getModelConfig_withVersion_parsesCorrectly() {
        TritonModelConfig config = client.getModelConfig("identity_fp32", "1");
        assertEquals("identity_fp32", config.getName());
    }

    // ========== Repository Index ==========

    @Test
    void getModelRepositoryIndex_parsesCorrectly() {
        TritonRepositoryIndex repoIndex = client.getModelRepositoryIndex();
        assertEquals(2, repoIndex.getModels().size());
        assertEquals("identity_fp32", repoIndex.getModels().get(0).getName());
        assertEquals("READY", repoIndex.getModels().get(0).getState());
    }

    // ========== Load / Unload ==========

    @Test
    void loadModel_success() {
        assertDoesNotThrow(() -> client.loadModel("sleeper"));
    }

    @Test
    void unLoadModel_success() {
        assertDoesNotThrow(() -> client.unLoadModel("sleeper"));
    }

    // ========== Statistics ==========

    @Test
    void getInferenceStatistics_parsesCorrectly() {
        List<TritonModelStatistics> stats = client.getInferenceStatistics("identity_fp32", "1");
        assertEquals(1, stats.size());
        assertEquals("identity_fp32", stats.get(0).getName());
        assertEquals(100, stats.get(0).getInferenceCount());
    }

    // ========== Close ==========

    @Test
    void close_doesNotThrow() {
        assertDoesNotThrow(() -> client.close());
    }

    // ========== Error Handling ==========

    @Test
    void getModelMetadata_notFound_throws() {
        assertThrows(TritonInferException.class, () -> client.getModelMetadata("nonexistent", "1"));
    }

    @Test
    void loadModel_error_throws() {
        assertThrows(TritonInferException.class, () -> client.loadModel("broken_model"));
    }

    @Test
    void getModelConfig_serverError_throws() {
        assertThrows(TritonInferException.class, () -> client.getModelConfig("error_model"));
    }

    @Test
    void unLoadModel_error_throws() {
        assertThrows(TritonInferException.class, () -> client.unLoadModel("error_model"));
    }

    // ========== Server Down ==========

    @Test
    void isServerLive_unreachableServer_returnsFalse() throws Exception {
        TritonClientConfig downConfig = new TritonClientConfig.Builder("localhost:1")
                .timeout(1000)
                .build();
        try (TritonHttpClient downClient = new TritonHttpClient(downConfig)) {
            assertFalse(downClient.isServerLive());
        }
    }

    @Test
    void isServerReady_unreachableServer_returnsFalse() throws Exception {
        TritonClientConfig downConfig = new TritonClientConfig.Builder("localhost:1")
                .timeout(1000)
                .build();
        try (TritonHttpClient downClient = new TritonHttpClient(downConfig)) {
            assertFalse(downClient.isServerReady());
        }
    }

    @Test
    void getServerMetadata_unreachableServer_throws() throws Exception {
        TritonClientConfig downConfig = new TritonClientConfig.Builder("localhost:1")
                .timeout(1000)
                .build();
        try (TritonHttpClient downClient = new TritonHttpClient(downConfig)) {
            assertThrows(TritonInferException.class, downClient::getServerMetadata);
        }
    }

    // ========== Null/Empty version handling ==========

    @Test
    void getModelMetadata_nullVersion_usesLatest() {
        TritonModelMetadata meta = client.getModelMetadata("identity_fp32", null);
        assertEquals("identity_fp32", meta.getName());
    }

    @Test
    void getModelMetadata_emptyVersion_usesLatest() {
        TritonModelMetadata meta = client.getModelMetadata("identity_fp32", "");
        assertEquals("identity_fp32", meta.getName());
    }

    @Test
    void getModelConfig_nullVersion_usesLatest() {
        TritonModelConfig config = client.getModelConfig("identity_fp32", null);
        assertEquals("identity_fp32", config.getName());
    }

    // ========== Mock Server Handlers ==========

    private static void registerHandlers(HttpServer server) {
        // Health
        server.createContext("/v2/health/live", ex -> respond(ex, 200, ""));
        server.createContext("/v2/health/ready", ex -> respond(ex, 200, ""));

        // Server metadata
        server.createContext("/v2", ex -> {
            if (ex.getRequestURI().getPath().equals("/v2")) {
                respond(ex, 200, """
                    {"name":"triton","version":"2.42.0","extensions":["classification","sequence"]}
                    """);
            } else {
                // Let other /v2/* handlers handle
                respond(ex, 404, """
                    {"error":"not found"}
                    """);
            }
        });

        // Model ready
        server.createContext("/v2/models/identity_fp32/ready", ex -> respond(ex, 200, ""));
        server.createContext("/v2/models/identity_fp32/versions/1/ready", ex -> respond(ex, 200, ""));
        server.createContext("/v2/models/nonexistent_model/ready", ex -> respond(ex, 404, """
                {"error":"model not found"}
                """));

        // Model metadata
        server.createContext("/v2/models/identity_fp32/versions/1", ex -> {
            String path = ex.getRequestURI().getPath();
            if (path.endsWith("/config")) {
                respond(ex, 200, """
                    {"name":"identity_fp32","platform":"","backend":"python","runtime":"",
                     "max_batch_size":8,"default_model_filename":"","cc_model_filenames":{},"metric_tags":{}}
                    """);
            } else if (path.endsWith("/stats")) {
                respond(ex, 200, """
                    {"model_stats":[{"name":"identity_fp32","version":"1","last_inference":0,
                     "inference_count":100,"execution_count":50,
                     "inference_stats":{"success":{"count":100,"ns":5000000},"fail":{"count":0,"ns":0},
                      "queue":{"count":100,"ns":200000},"compute_input":{"count":100,"ns":100000},
                      "compute_infer":{"count":100,"ns":4000000},"compute_output":{"count":100,"ns":100000},
                      "cache_hit":{"count":0,"ns":0},"cache_miss":{"count":0,"ns":0}},
                     "memory_usage":[],"response_stats":{}}]}
                    """);
            } else {
                respond(ex, 200, """
                    {"name":"identity_fp32","versions":["1"],"platform":"python",
                     "inputs":[{"name":"INPUT0","datatype":"FP32","shape":[-1,-1]}],
                     "outputs":[{"name":"OUTPUT0","datatype":"FP32","shape":[-1,-1]}]}
                    """);
            }
        });

        // Model config (without version)
        server.createContext("/v2/models/identity_fp32/config", ex ->
            respond(ex, 200, """
                {"name":"identity_fp32","platform":"","backend":"python","runtime":"",
                 "max_batch_size":8,"default_model_filename":"","cc_model_filenames":{},"metric_tags":{}}
                """));

        // Nonexistent model metadata
        server.createContext("/v2/models/nonexistent/versions/1", ex ->
            respond(ex, 404, """
                {"error":"Request for unknown model: nonexistent is not found"}
                """));

        // Repository index
        server.createContext("/v2/repository/index", ex ->
            respond(ex, 200, """
                [{"name":"identity_fp32","version":"1","state":"READY","reason":""},
                 {"name":"sleeper","version":"1","state":"READY","reason":""}]
                """));

        // Load / Unload
        server.createContext("/v2/repository/models/sleeper/load", ex -> respond(ex, 200, ""));
        server.createContext("/v2/repository/models/sleeper/unload", ex -> respond(ex, 200, ""));
        server.createContext("/v2/repository/models/broken_model/load", ex ->
            respond(ex, 400, """
                {"error":"failed to load model 'broken_model'"}
                """));

        // Error model (500 server error)
        server.createContext("/v2/models/error_model/config", ex ->
            respond(ex, 500, """
                {"error":"internal server error"}
                """));
        server.createContext("/v2/repository/models/error_model/unload", ex ->
            respond(ex, 500, """
                {"error":"internal server error during unload"}
                """));

        // identity_fp32 without version (for null/empty version tests)
        server.createContext("/v2/models/identity_fp32", ex -> {
            String path = ex.getRequestURI().getPath();
            if (path.equals("/v2/models/identity_fp32")) {
                respond(ex, 200, """
                    {"name":"identity_fp32","versions":["1"],"platform":"python",
                     "inputs":[{"name":"INPUT0","datatype":"FP32","shape":[-1,-1]}],
                     "outputs":[{"name":"OUTPUT0","datatype":"FP32","shape":[-1,-1]}]}
                    """);
            } else {
                respond(ex, 404, """
                    {"error":"not found"}
                    """);
            }
        });
    }

    private static void respond(HttpExchange exchange, int statusCode, String body) throws IOException {
        byte[] bytes = body.trim().getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json");
        exchange.sendResponseHeaders(statusCode, bytes.length == 0 ? -1 : bytes.length);
        if (bytes.length > 0) {
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(bytes);
            }
        }
        exchange.close();
    }
}
