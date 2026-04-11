package com.gencior.triton.http;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import com.gencior.triton.config.TritonClientConfig;
import com.gencior.triton.core.InferInput;
import com.gencior.triton.core.InferRequestedOutput;
import com.gencior.triton.core.InferResult;
import com.gencior.triton.core.InferStreamHandle;
import com.gencior.triton.core.InferStreamListener;
import com.gencior.triton.core.TritonDataType;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

/**
 * Unit tests for TritonHttpClient inference and streaming methods.
 */
class TritonHttpClientInferTest {

    private static HttpServer mockServer;
    private static TritonHttpClient client;

    @BeforeAll
    static void setUp() throws IOException {
        mockServer = HttpServer.create(new InetSocketAddress(0), 0);
        registerHandlers(mockServer);
        mockServer.start();

        int port = mockServer.getAddress().getPort();
        TritonClientConfig config = new TritonClientConfig.Builder("localhost:" + port)
                .timeout(10000)
                .build();
        client = new TritonHttpClient(config);
    }

    @AfterAll
    static void tearDown() throws Exception {
        if (client != null) client.close();
        if (mockServer != null) mockServer.stop(0);
    }

    // ========== Synchronous Inference ==========

    @Test
    void infer_fp32_shouldReturnCorrectOutput() {
        float[] inputData = {1.0f, 2.0f, 3.0f};
        InferInput input = new InferInput("INPUT0", new long[]{3}, TritonDataType.FP32);
        input.setData(inputData);

        InferResult result = client.infer("identity_fp32", List.of(input));

        assertNotNull(result);
        assertEquals("identity_fp32", result.getModelName());
        assertNotNull(result.getOutputNames());
        assertTrue(result.getOutputNames().contains("OUTPUT0"));

        float[] output = result.asFloatArray("OUTPUT0");
        assertArrayEquals(inputData, output, 1e-6f);
    }

    @Test
    void infer_int32_shouldReturnCorrectOutput() {
        int[] inputData = {10, 20, 30};
        InferInput input = new InferInput("INPUT0", new long[]{3}, TritonDataType.INT32);
        input.setData(inputData);

        InferResult result = client.infer("identity_int32", List.of(input));

        assertNotNull(result);
        int[] output = result.asIntArray("OUTPUT0");
        assertArrayEquals(inputData, output);
    }

    @Test
    void infer_withVersion_shouldWork() {
        float[] inputData = {5.0f};
        InferInput input = new InferInput("INPUT0", new long[]{1}, TritonDataType.FP32);
        input.setData(inputData);

        InferResult result = client.infer("identity_fp32", "1", List.of(input), null);

        assertNotNull(result);
        assertEquals("1", result.getModelVersion());
    }

    @Test
    void infer_withRequestedOutputs_shouldWork() {
        float[] inputData = {1.0f, 2.0f};
        InferInput input = new InferInput("INPUT0", new long[]{2}, TritonDataType.FP32);
        input.setData(inputData);

        InferResult result = client.infer("identity_fp32", "1", List.of(input),
                List.of(InferRequestedOutput.of("OUTPUT0")), null);

        assertNotNull(result);
        float[] output = result.asFloatArray("OUTPUT0");
        assertArrayEquals(inputData, output, 1e-6f);
    }

    // ========== Asynchronous Inference ==========

    @Test
    void inferAsync_shouldCompleteSuccessfully() throws Exception {
        float[] inputData = {4.0f, 5.0f, 6.0f};
        InferInput input = new InferInput("INPUT0", new long[]{3}, TritonDataType.FP32);
        input.setData(inputData);

        CompletableFuture<InferResult> future = client.inferAsync("identity_fp32", List.of(input));

        InferResult result = future.get(5, TimeUnit.SECONDS);
        assertNotNull(result);
        float[] output = result.asFloatArray("OUTPUT0");
        assertArrayEquals(inputData, output, 1e-6f);
    }

    @Test
    void inferAsync_multipleConcurrent_shouldAllComplete() throws Exception {
        List<CompletableFuture<InferResult>> futures = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            InferInput input = new InferInput("INPUT0", new long[]{1}, TritonDataType.FP32);
            input.setData(new float[]{(float) i});
            futures.add(client.inferAsync("identity_fp32", List.of(input)));
        }

        for (CompletableFuture<InferResult> future : futures) {
            InferResult result = future.get(5, TimeUnit.SECONDS);
            assertNotNull(result);
            assertNotNull(result.asFloatArray("OUTPUT0"));
        }
    }

    // ========== Streaming (SSE) ==========

    @Test
    void inferStream_shouldReceiveAllTokens() throws Exception {
        InferInput input = new InferInput("TEXT_INPUT", new long[]{1}, TritonDataType.BYTES);
        input.setData(new String[]{"hello world"});

        List<String> tokens = Collections.synchronizedList(new ArrayList<>());
        CountDownLatch latch = new CountDownLatch(1);
        AtomicReference<Throwable> error = new AtomicReference<>();

        InferStreamHandle handle = client.inferStream("streaming_echo", List.of(input),
                new InferStreamListener() {
                    @Override
                    public void onToken(InferResult result) {
                        tokens.add(result.asStringArray("TEXT_OUTPUT")[0]);
                    }

                    @Override
                    public void onComplete() {
                        latch.countDown();
                    }

                    @Override
                    public void onError(Throwable t) {
                        error.set(t);
                        latch.countDown();
                    }
                });

        assertTrue(latch.await(5, TimeUnit.SECONDS), "Stream should complete within 5s");
        assertNull(error.get(), "Stream should not have errors");
        assertEquals(3, tokens.size());
        assertEquals("token1", tokens.get(0));
        assertEquals("token2", tokens.get(1));
        assertEquals("token3", tokens.get(2));
        assertTrue(handle.isDone());
    }

    @Test
    void inferStream_cancel_shouldStop() throws Exception {
        InferInput input = new InferInput("TEXT_INPUT", new long[]{1}, TritonDataType.BYTES);
        input.setData(new String[]{"hello"});

        AtomicReference<InferStreamHandle> handleRef = new AtomicReference<>();
        CountDownLatch firstToken = new CountDownLatch(1);

        InferStreamHandle handle = client.inferStream("streaming_echo", List.of(input),
                new InferStreamListener() {
                    @Override
                    public void onToken(InferResult result) {
                        firstToken.countDown();
                        // Cancel after first token
                        InferStreamHandle h = handleRef.get();
                        if (h != null) h.cancel();
                    }
                });
        handleRef.set(handle);

        firstToken.await(5, TimeUnit.SECONDS);
        Thread.sleep(200);
        handle.cancel(); // safe to call multiple times
    }

    @Test
    void inferStreamPublisher_shouldReceiveAllTokens() throws Exception {
        InferInput input = new InferInput("TEXT_INPUT", new long[]{1}, TritonDataType.BYTES);
        input.setData(new String[]{"hello world"});

        List<String> tokens = Collections.synchronizedList(new ArrayList<>());
        CountDownLatch latch = new CountDownLatch(1);

        client.inferStreamPublisher("streaming_echo", List.of(input))
                .subscribe(new java.util.concurrent.Flow.Subscriber<InferResult>() {
                    @Override
                    public void onSubscribe(java.util.concurrent.Flow.Subscription subscription) {
                        subscription.request(Long.MAX_VALUE);
                    }

                    @Override
                    public void onNext(InferResult result) {
                        tokens.add(result.asStringArray("TEXT_OUTPUT")[0]);
                    }

                    @Override
                    public void onError(Throwable throwable) {
                        latch.countDown();
                    }

                    @Override
                    public void onComplete() {
                        latch.countDown();
                    }
                });

        assertTrue(latch.await(5, TimeUnit.SECONDS));
        assertEquals(3, tokens.size());
    }

    // ========== Mock Server Handlers ==========

    private static void registerHandlers(HttpServer server) {
        // Identity FP32 inference (binary extension)
        server.createContext("/v2/models/identity_fp32/infer", ex -> respondBinaryInfer(ex, "identity_fp32", "1", "FP32"));
        server.createContext("/v2/models/identity_fp32/versions/1/infer", ex -> respondBinaryInfer(ex, "identity_fp32", "1", "FP32"));

        // Identity INT32 inference
        server.createContext("/v2/models/identity_int32/infer", ex -> respondBinaryInfer(ex, "identity_int32", "1", "INT32"));

        // Streaming echo
        server.createContext("/v2/models/streaming_echo/generate_stream", ex -> respondSseStream(ex));
    }

    /**
     * Mock handler that echoes the binary input data back as output (identity model).
     */
    private static void respondBinaryInfer(HttpExchange exchange, String modelName, String modelVersion, String datatype) throws IOException {
        // Read the request body
        byte[] requestBody = exchange.getRequestBody().readAllBytes();

        // Parse the inference header content length
        String headerLenStr = exchange.getRequestHeaders().getFirst("Inference-Header-Content-Length");
        int jsonLen = headerLenStr != null ? Integer.parseInt(headerLenStr) : requestBody.length;

        // Extract binary input data (after JSON header)
        byte[] binaryInput = new byte[requestBody.length - jsonLen];
        if (binaryInput.length > 0) {
            System.arraycopy(requestBody, jsonLen, binaryInput, 0, binaryInput.length);
        }

        // Build response JSON header
        int elementSize = "FP32".equals(datatype) ? 4 : 4; // both FP32 and INT32 are 4 bytes
        int numElements = binaryInput.length / elementSize;
        String responseJson = String.format(
                "{\"model_name\":\"%s\",\"model_version\":\"%s\",\"id\":\"req-1\"," +
                "\"outputs\":[{\"name\":\"OUTPUT0\",\"datatype\":\"%s\",\"shape\":[%d]," +
                "\"parameters\":{\"binary_data_size\":%d}}]}",
                modelName, modelVersion, datatype, numElements, binaryInput.length);

        byte[] jsonBytes = responseJson.getBytes(StandardCharsets.UTF_8);
        byte[] responseBody = new byte[jsonBytes.length + binaryInput.length];
        System.arraycopy(jsonBytes, 0, responseBody, 0, jsonBytes.length);
        System.arraycopy(binaryInput, 0, responseBody, jsonBytes.length, binaryInput.length);

        exchange.getResponseHeaders().set("Content-Type", "application/octet-stream");
        exchange.getResponseHeaders().set("Inference-Header-Content-Length", String.valueOf(jsonBytes.length));
        exchange.sendResponseHeaders(200, responseBody.length);
        try (OutputStream os = exchange.getResponseBody()) {
            os.write(responseBody);
        }
        exchange.close();
    }

    /**
     * Mock SSE streaming handler that sends 3 tokens.
     */
    private static void respondSseStream(HttpExchange exchange) throws IOException {
        exchange.getRequestBody().readAllBytes(); // consume request body
        exchange.getResponseHeaders().set("Content-Type", "text/event-stream");
        exchange.sendResponseHeaders(200, 0);

        try (OutputStream os = exchange.getResponseBody()) {
            String[] tokens = {"token1", "token2", "token3"};
            for (String token : tokens) {
                // Generate endpoint uses flat key-value format
                String event = String.format(
                        "data: {\"model_name\":\"streaming_echo\",\"model_version\":\"1\"," +
                        "\"TEXT_OUTPUT\":\"%s\"}\n\n", token);
                os.write(event.getBytes(StandardCharsets.UTF_8));
                os.flush();
            }
        }
        exchange.close();
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
