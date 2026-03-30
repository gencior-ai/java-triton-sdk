package com.gencior.triton.grpc;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.when;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import org.junit.Before;
import org.junit.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import com.gencior.triton.config.TritonClientConfig;
import com.gencior.triton.core.InferInput;
import com.gencior.triton.core.InferStreamHandle;
import com.gencior.triton.core.InferStreamListener;
import com.gencior.triton.core.TritonDataType;
import com.gencior.triton.exceptions.TritonDataNotFoundException;
import com.gencior.triton.exceptions.TritonStreamException;

import inference.GRPCInferenceServiceGrpc.GRPCInferenceServiceBlockingStub;
import inference.GRPCInferenceServiceGrpc.GRPCInferenceServiceStub;
import inference.GrpcService.ModelInferRequest;
import inference.GrpcService.ModelInferResponse;
import inference.GrpcService.ModelStreamInferResponse;
import io.grpc.ManagedChannel;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.stub.StreamObserver;

@SuppressWarnings("unchecked")
public class TritonGrpcClientStreamTest {

    private TritonGrpcClient client;

    @Mock
    private TritonClientConfig config;

    @Mock
    private ManagedChannel channel;

    @Mock
    private GRPCInferenceServiceBlockingStub blockingStub;

    @Mock
    private GRPCInferenceServiceStub asyncStub;

    @Before
    public void setUp() {
        MockitoAnnotations.openMocks(this);
        when(config.getDefaultTimeoutMs()).thenReturn(1000L);
        when(blockingStub.withDeadlineAfter(1000L, TimeUnit.MILLISECONDS)).thenReturn(blockingStub);
        client = new TritonGrpcClient(config, channel, blockingStub, asyncStub);
    }

    private ModelStreamInferResponse buildStreamResponse(String token) {
        ModelInferResponse inferResponse = ModelInferResponse.newBuilder()
                .setModelName("test_model")
                .build();
        return ModelStreamInferResponse.newBuilder()
                .setInferResponse(inferResponse)
                .build();
    }

    private ModelStreamInferResponse buildErrorStreamResponse(String error) {
        return ModelStreamInferResponse.newBuilder()
                .setErrorMessage(error)
                .build();
    }

    // ==================== inferStream callback tests ====================

    @Test
    public void testInferStream_multipleTokens() throws Exception {
        List<String> receivedTokens = Collections.synchronizedList(new ArrayList<>());
        CountDownLatch completeLatch = new CountDownLatch(1);

        doAnswer(invocation -> {
            StreamObserver<ModelStreamInferResponse> responseObserver = invocation.getArgument(0);
            StreamObserver<ModelInferRequest> requestObserver = new NoOpStreamObserver<>();

            responseObserver.onNext(buildStreamResponse("token1"));
            responseObserver.onNext(buildStreamResponse("token2"));
            responseObserver.onNext(buildStreamResponse("token3"));
            responseObserver.onCompleted();

            return requestObserver;
        }).when(asyncStub).modelStreamInfer(any(StreamObserver.class));

        InferInput input = new InferInput("INPUT0", new long[]{1}, TritonDataType.FP32);
        input.setData(new float[]{1.0f});

        InferStreamHandle handle = client.inferStream("test_model", List.of(input),
                new InferStreamListener() {
                    @Override
                    public void onToken(com.gencior.triton.core.InferResult result) {
                        receivedTokens.add(result.getModelName());
                    }

                    @Override
                    public void onComplete() {
                        completeLatch.countDown();
                    }
                });

        assertTrue(completeLatch.await(5, TimeUnit.SECONDS));
        assertEquals(3, receivedTokens.size());
        assertNotNull(handle);
    }

    @Test
    public void testInferStream_errorMessage() throws Exception {
        AtomicReference<Throwable> receivedError = new AtomicReference<>();
        CountDownLatch errorLatch = new CountDownLatch(1);

        doAnswer(invocation -> {
            StreamObserver<ModelStreamInferResponse> responseObserver = invocation.getArgument(0);
            StreamObserver<ModelInferRequest> requestObserver = new NoOpStreamObserver<>();

            responseObserver.onNext(buildErrorStreamResponse("Model execution failed"));

            return requestObserver;
        }).when(asyncStub).modelStreamInfer(any(StreamObserver.class));

        InferInput input = new InferInput("INPUT0", new long[]{1}, TritonDataType.FP32);
        input.setData(new float[]{1.0f});

        InferStreamHandle handle = client.inferStream("test_model", List.of(input),
                new InferStreamListener() {
                    @Override
                    public void onToken(com.gencior.triton.core.InferResult result) {}

                    @Override
                    public void onError(Throwable t) {
                        receivedError.set(t);
                        errorLatch.countDown();
                    }
                });

        assertTrue(errorLatch.await(5, TimeUnit.SECONDS));
        assertTrue(receivedError.get() instanceof TritonStreamException);
        assertTrue(receivedError.get().getMessage().contains("Model execution failed"));
    }

    @Test
    public void testInferStream_grpcError() throws Exception {
        AtomicReference<Throwable> receivedError = new AtomicReference<>();
        CountDownLatch errorLatch = new CountDownLatch(1);

        doAnswer(invocation -> {
            StreamObserver<ModelStreamInferResponse> responseObserver = invocation.getArgument(0);
            StreamObserver<ModelInferRequest> requestObserver = new NoOpStreamObserver<>();

            responseObserver.onError(new StatusRuntimeException(Status.UNAVAILABLE));

            return requestObserver;
        }).when(asyncStub).modelStreamInfer(any(StreamObserver.class));

        InferInput input = new InferInput("INPUT0", new long[]{1}, TritonDataType.FP32);
        input.setData(new float[]{1.0f});

        client.inferStream("test_model", List.of(input),
                new InferStreamListener() {
                    @Override
                    public void onToken(com.gencior.triton.core.InferResult result) {}

                    @Override
                    public void onError(Throwable t) {
                        receivedError.set(t);
                        errorLatch.countDown();
                    }
                });

        assertTrue(errorLatch.await(5, TimeUnit.SECONDS));
        assertTrue(receivedError.get() instanceof StatusRuntimeException);
    }

    @Test
    public void testInferStream_handleAwait() throws Exception {
        doAnswer(invocation -> {
            StreamObserver<ModelStreamInferResponse> responseObserver = invocation.getArgument(0);
            StreamObserver<ModelInferRequest> requestObserver = new NoOpStreamObserver<>();

            responseObserver.onNext(buildStreamResponse("token"));
            responseObserver.onCompleted();

            return requestObserver;
        }).when(asyncStub).modelStreamInfer(any(StreamObserver.class));

        InferInput input = new InferInput("INPUT0", new long[]{1}, TritonDataType.FP32);
        input.setData(new float[]{1.0f});

        InferStreamHandle handle = client.inferStream("test_model", List.of(input),
                result -> {});

        handle.await(5, TimeUnit.SECONDS);
        assertTrue(handle.isDone());
    }

    @Test
    public void testInferStream_handleAwaitOnError() throws Exception {
        doAnswer(invocation -> {
            StreamObserver<ModelStreamInferResponse> responseObserver = invocation.getArgument(0);
            StreamObserver<ModelInferRequest> requestObserver = new NoOpStreamObserver<>();

            responseObserver.onNext(buildErrorStreamResponse("fail"));

            return requestObserver;
        }).when(asyncStub).modelStreamInfer(any(StreamObserver.class));

        InferInput input = new InferInput("INPUT0", new long[]{1}, TritonDataType.FP32);
        input.setData(new float[]{1.0f});

        InferStreamHandle handle = client.inferStream("test_model", List.of(input),
                new InferStreamListener() {
                    @Override
                    public void onToken(com.gencior.triton.core.InferResult result) {}
                    @Override
                    public void onError(Throwable t) {}
                });

        try {
            handle.await(5, TimeUnit.SECONDS);
        } catch (ExecutionException e) {
            assertTrue(e.getCause() instanceof TritonStreamException);
        }
    }

    @Test(expected = TritonDataNotFoundException.class)
    public void testInferStream_missingData() {
        InferInput input = new InferInput("INPUT0", new long[]{1}, TritonDataType.FP32);
        client.inferStream("test_model", List.of(input), result -> {});
    }

    @Test
    public void testInferStream_lambdaListener() throws Exception {
        AtomicBoolean tokenReceived = new AtomicBoolean(false);

        doAnswer(invocation -> {
            StreamObserver<ModelStreamInferResponse> responseObserver = invocation.getArgument(0);
            StreamObserver<ModelInferRequest> requestObserver = new NoOpStreamObserver<>();

            responseObserver.onNext(buildStreamResponse("token"));
            responseObserver.onCompleted();

            return requestObserver;
        }).when(asyncStub).modelStreamInfer(any(StreamObserver.class));

        InferInput input = new InferInput("INPUT0", new long[]{1}, TritonDataType.FP32);
        input.setData(new float[]{1.0f});

        InferStreamHandle handle = client.inferStream("test_model", List.of(input),
                result -> tokenReceived.set(true));

        handle.await(5, TimeUnit.SECONDS);
        assertTrue(tokenReceived.get());
    }

    // ==================== inferStreamPublisher tests ====================

    @Test
    public void testInferStreamPublisher_success() throws Exception {
        List<String> receivedModels = Collections.synchronizedList(new ArrayList<>());
        CountDownLatch completeLatch = new CountDownLatch(1);

        doAnswer(invocation -> {
            StreamObserver<ModelStreamInferResponse> responseObserver = invocation.getArgument(0);
            StreamObserver<ModelInferRequest> requestObserver = new NoOpStreamObserver<>();

            responseObserver.onNext(buildStreamResponse("t1"));
            responseObserver.onNext(buildStreamResponse("t2"));
            responseObserver.onCompleted();

            return requestObserver;
        }).when(asyncStub).modelStreamInfer(any(StreamObserver.class));

        InferInput input = new InferInput("INPUT0", new long[]{1}, TritonDataType.FP32);
        input.setData(new float[]{1.0f});

        java.util.concurrent.Flow.Publisher<com.gencior.triton.core.InferResult> publisher =
                client.inferStreamPublisher("test_model", List.of(input));

        publisher.subscribe(new java.util.concurrent.Flow.Subscriber<>() {
            java.util.concurrent.Flow.Subscription subscription;

            @Override
            public void onSubscribe(java.util.concurrent.Flow.Subscription s) {
                this.subscription = s;
                s.request(Long.MAX_VALUE);
            }

            @Override
            public void onNext(com.gencior.triton.core.InferResult result) {
                receivedModels.add(result.getModelName());
            }

            @Override
            public void onError(Throwable t) {}

            @Override
            public void onComplete() {
                completeLatch.countDown();
            }
        });

        assertTrue(completeLatch.await(5, TimeUnit.SECONDS));
        assertEquals(2, receivedModels.size());
    }

    @Test
    public void testInferStreamPublisher_errorPropagated() throws Exception {
        AtomicReference<Throwable> receivedError = new AtomicReference<>();
        CountDownLatch errorLatch = new CountDownLatch(1);

        doAnswer(invocation -> {
            StreamObserver<ModelStreamInferResponse> responseObserver = invocation.getArgument(0);
            StreamObserver<ModelInferRequest> requestObserver = new NoOpStreamObserver<>();

            responseObserver.onError(new StatusRuntimeException(Status.INTERNAL));

            return requestObserver;
        }).when(asyncStub).modelStreamInfer(any(StreamObserver.class));

        InferInput input = new InferInput("INPUT0", new long[]{1}, TritonDataType.FP32);
        input.setData(new float[]{1.0f});

        client.inferStreamPublisher("test_model", List.of(input))
                .subscribe(new java.util.concurrent.Flow.Subscriber<>() {
                    @Override
                    public void onSubscribe(java.util.concurrent.Flow.Subscription s) {
                        s.request(Long.MAX_VALUE);
                    }

                    @Override
                    public void onNext(com.gencior.triton.core.InferResult result) {}

                    @Override
                    public void onError(Throwable t) {
                        receivedError.set(t);
                        errorLatch.countDown();
                    }

                    @Override
                    public void onComplete() {}
                });

        assertTrue(errorLatch.await(5, TimeUnit.SECONDS));
        assertTrue(receivedError.get() instanceof StatusRuntimeException);
    }

    private static class NoOpStreamObserver<T> implements StreamObserver<T> {
        @Override public void onNext(T value) {}
        @Override public void onError(Throwable t) {}
        @Override public void onCompleted() {}
    }
}
