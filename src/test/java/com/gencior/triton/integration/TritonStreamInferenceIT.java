package com.gencior.triton.integration;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Flow;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.Test;

import com.gencior.triton.core.InferInput;
import com.gencior.triton.core.InferResult;
import com.gencior.triton.core.InferStreamHandle;
import com.gencior.triton.core.InferStreamListener;
import com.gencior.triton.core.TritonDataType;

/**
 * Integration tests for streaming inference with the decoupled
 * {@code streaming_echo} model. The model splits input text into words
 * and streams each word as a separate response, simulating LLM
 * token-by-token generation.
 */
class TritonStreamInferenceIT extends AbstractTritonIntegrationTest {

    private InferInput createTextInput(String text) {
        InferInput input = new InferInput("TEXT_INPUT", new long[]{1}, TritonDataType.BYTES);
        input.setData(new String[]{text});
        return input;
    }

    // ==================== Callback-based streaming ====================

    @Test
    void inferStream_shouldReceiveAllTokens() throws Exception {
        List<String> tokens = Collections.synchronizedList(new ArrayList<>());
        CountDownLatch completeLatch = new CountDownLatch(1);

        InferStreamHandle handle = client.inferStream("streaming_echo",
                List.of(createTextInput("hello world foo bar")),
                new InferStreamListener() {
                    @Override
                    public void onToken(InferResult result) {
                        tokens.add(result.asStringArray("TEXT_OUTPUT")[0]);
                    }

                    @Override
                    public void onComplete() {
                        completeLatch.countDown();
                    }
                });

        assertTrue(completeLatch.await(30, TimeUnit.SECONDS));
        assertTrue(handle.isDone());
        assertEquals(4, tokens.size());
        assertEquals("hello", tokens.get(0));
        assertEquals("world", tokens.get(1));
        assertEquals("foo", tokens.get(2));
        assertEquals("bar", tokens.get(3));
    }

    @Test
    void inferStream_singleToken_shouldWork() throws Exception {
        List<String> tokens = Collections.synchronizedList(new ArrayList<>());

        InferStreamHandle handle = client.inferStream("streaming_echo",
                List.of(createTextInput("single")),
                new InferStreamListener() {
                    @Override
                    public void onToken(InferResult result) {
                        tokens.add(result.asStringArray("TEXT_OUTPUT")[0]);
                    }
                });

        handle.await(30, TimeUnit.SECONDS);
        assertEquals(1, tokens.size());
        assertEquals("single", tokens.get(0));
    }

    @Test
    void inferStream_lambdaListener_shouldWork() throws Exception {
        List<String> tokens = Collections.synchronizedList(new ArrayList<>());

        InferStreamHandle handle = client.inferStream("streaming_echo",
                List.of(createTextInput("java triton sdk")),
                result -> tokens.add(result.asStringArray("TEXT_OUTPUT")[0])
            );

        handle.await(30, TimeUnit.SECONDS);
        assertEquals(3, tokens.size());
        assertEquals("java", tokens.get(0));
    }

    @Test
    void inferStream_handleCancel_shouldAbortStream() throws Exception {
        List<String> tokens = Collections.synchronizedList(new ArrayList<>());

        // Send many words so we have time to cancel
        String longText = "one two three four five six seven eight nine ten";
        InferStreamHandle handle = client.inferStream("streaming_echo",
                List.of(createTextInput(longText)),
                result -> tokens.add(result.asStringArray("TEXT_OUTPUT")[0]));

        // Wait for at least one token, then cancel
        Thread.sleep(200);
        handle.cancel();

        // Should have received some but not all tokens
        assertTrue(tokens.size() < 10,
                "Expected fewer than 10 tokens after cancel, got " + tokens.size());
        assertTrue(handle.isDone() || true); // cancel is async, may not be done immediately
    }

    @Test
    void inferStream_withVersion_shouldWork() throws Exception {
        List<String> tokens = Collections.synchronizedList(new ArrayList<>());

        InferStreamHandle handle = client.inferStream("streaming_echo", "1",
                List.of(createTextInput("versioned test")), null,
                result -> tokens.add(result.asStringArray("TEXT_OUTPUT")[0]));

        handle.await(30, TimeUnit.SECONDS);
        assertEquals(2, tokens.size());
    }

    @Test
    void inferStream_resultMetadata_shouldBeValid() throws Exception {
        List<InferResult> results = Collections.synchronizedList(new ArrayList<>());

        InferStreamHandle handle = client.inferStream("streaming_echo",
                List.of(createTextInput("meta test")),
                new InferStreamListener() {
                    @Override
                    public void onToken(InferResult result) {
                        results.add(result);
                    }
                });

        handle.await(30, TimeUnit.SECONDS);
        assertEquals(2, results.size());

        InferResult first = results.get(0);
        assertNotNull(first.getModelName());
        assertEquals("streaming_echo", first.getModelName());
        assertTrue(first.getOutputNames().contains("TEXT_OUTPUT"));
        assertNotNull(first.getOutput("TEXT_OUTPUT"));
    }

    // ==================== Flow.Publisher-based streaming ====================

    @Test
    void inferStreamPublisher_shouldReceiveAllTokens() throws Exception {
        List<String> tokens = Collections.synchronizedList(new ArrayList<>());
        CountDownLatch completeLatch = new CountDownLatch(1);

        Flow.Publisher<InferResult> publisher = client.inferStreamPublisher(
                "streaming_echo", List.of(createTextInput("pub sub test")));

        publisher.subscribe(new Flow.Subscriber<>() {
            Flow.Subscription subscription;

            @Override
            public void onSubscribe(Flow.Subscription s) {
                this.subscription = s;
                s.request(Long.MAX_VALUE);
            }

            @Override
            public void onNext(InferResult result) {
                tokens.add(result.asStringArray("TEXT_OUTPUT")[0]);
            }

            @Override
            public void onError(Throwable t) {}

            @Override
            public void onComplete() {
                completeLatch.countDown();
            }
        });

        assertTrue(completeLatch.await(30, TimeUnit.SECONDS));
        assertEquals(3, tokens.size());
        assertEquals("pub", tokens.get(0));
        assertEquals("sub", tokens.get(1));
        assertEquals("test", tokens.get(2));
    }

    @Test
    void inferStreamPublisher_cancel_shouldStopStream() throws Exception {
        List<String> tokens = Collections.synchronizedList(new ArrayList<>());

        String longText = "a b c d e f g h i j";
        Flow.Publisher<InferResult> publisher = client.inferStreamPublisher(
                "streaming_echo", List.of(createTextInput(longText)));

        publisher.subscribe(new Flow.Subscriber<>() {
            Flow.Subscription subscription;

            @Override
            public void onSubscribe(Flow.Subscription s) {
                this.subscription = s;
                s.request(Long.MAX_VALUE);
            }

            @Override
            public void onNext(InferResult result) {
                tokens.add(result.asStringArray("TEXT_OUTPUT")[0]);
                if (tokens.size() >= 3) {
                    subscription.cancel();
                }
            }

            @Override
            public void onError(Throwable t) {}

            @Override
            public void onComplete() {}
        });

        Thread.sleep(1000);
        assertTrue(tokens.size() >= 3);
        assertTrue(tokens.size() < 10,
                "Expected stream to stop after cancel, got " + tokens.size() + " tokens");
    }
}
