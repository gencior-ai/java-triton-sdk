package com.gencior.triton.grpc;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import org.junit.Before;
import org.junit.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import com.gencior.triton.config.TritonClientConfig;
import com.gencior.triton.core.InferInput;
import com.gencior.triton.core.InferRequestedOutput;
import com.gencior.triton.core.InferResult;
import com.gencior.triton.core.TritonDataType;

import inference.GRPCInferenceServiceGrpc.GRPCInferenceServiceBlockingStub;
import inference.GRPCInferenceServiceGrpc.GRPCInferenceServiceStub;
import inference.GrpcService.ModelInferRequest;
import inference.GrpcService.ModelInferResponse;
import io.grpc.ManagedChannel;
import io.grpc.stub.StreamObserver;

@SuppressWarnings("unchecked")
public class TritonGrpcClientRequestedOutputTest {

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

    // ==================== infer with requested outputs ====================

    @Test
    public void testInfer_withRequestedOutputs_sendsOutputsInRequest() {
        InferInput input = new InferInput("input0", new long[]{1}, TritonDataType.FP32);
        input.setData(new float[]{1.0f});

        ModelInferResponse responseProto = ModelInferResponse.newBuilder()
                .setModelName("test_model")
                .build();

        when(blockingStub.modelInfer(any(ModelInferRequest.class))).thenReturn(responseProto);

        List<InferRequestedOutput> outputs = List.of(
                InferRequestedOutput.of("embeddings"),
                InferRequestedOutput.of("logits")
        );

        InferResult result = client.infer("test_model", null, List.of(input), outputs, null);

        assertNotNull(result);

        ArgumentCaptor<ModelInferRequest> captor = ArgumentCaptor.forClass(ModelInferRequest.class);
        verify(blockingStub).modelInfer(captor.capture());

        ModelInferRequest capturedRequest = captor.getValue();
        assertEquals(2, capturedRequest.getOutputsCount());
        assertEquals("embeddings", capturedRequest.getOutputs(0).getName());
        assertEquals("logits", capturedRequest.getOutputs(1).getName());
    }

    @Test
    public void testInfer_withoutRequestedOutputs_sendsNoOutputsField() {
        InferInput input = new InferInput("input0", new long[]{1}, TritonDataType.FP32);
        input.setData(new float[]{1.0f});

        ModelInferResponse responseProto = ModelInferResponse.newBuilder()
                .setModelName("test_model")
                .build();

        when(blockingStub.modelInfer(any(ModelInferRequest.class))).thenReturn(responseProto);

        InferResult result = client.infer("test_model", List.of(input));

        assertNotNull(result);

        ArgumentCaptor<ModelInferRequest> captor = ArgumentCaptor.forClass(ModelInferRequest.class);
        verify(blockingStub).modelInfer(captor.capture());

        assertEquals(0, captor.getValue().getOutputsCount());
    }

    @Test
    public void testInfer_withOutputParameters_sendsParametersInRequest() {
        InferInput input = new InferInput("input0", new long[]{1}, TritonDataType.FP32);
        input.setData(new float[]{1.0f});

        ModelInferResponse responseProto = ModelInferResponse.newBuilder()
                .setModelName("test_model")
                .build();

        when(blockingStub.modelInfer(any(ModelInferRequest.class))).thenReturn(responseProto);

        InferRequestedOutput output = new InferRequestedOutput.Builder("classification")
                .addParameter("classification", 3L)
                .addParameter("binary_data", true)
                .build();

        client.infer("test_model", null, List.of(input), List.of(output), null);

        ArgumentCaptor<ModelInferRequest> captor = ArgumentCaptor.forClass(ModelInferRequest.class);
        verify(blockingStub).modelInfer(captor.capture());

        ModelInferRequest.InferRequestedOutputTensor capturedOutput = captor.getValue().getOutputs(0);
        assertEquals("classification", capturedOutput.getName());
        assertEquals(2, capturedOutput.getParametersCount());
        assertEquals(3L, capturedOutput.getParametersMap().get("classification").getInt64Param());
        assertTrue(capturedOutput.getParametersMap().get("binary_data").getBoolParam());
    }

    @Test
    public void testInfer_withEmptyOutputList_sendsNoOutputsField() {
        InferInput input = new InferInput("input0", new long[]{1}, TritonDataType.FP32);
        input.setData(new float[]{1.0f});

        ModelInferResponse responseProto = ModelInferResponse.newBuilder().setModelName("m").build();
        when(blockingStub.modelInfer(any(ModelInferRequest.class))).thenReturn(responseProto);

        client.infer("test_model", null, List.of(input), List.of(), null);

        ArgumentCaptor<ModelInferRequest> captor = ArgumentCaptor.forClass(ModelInferRequest.class);
        verify(blockingStub).modelInfer(captor.capture());

        assertEquals(0, captor.getValue().getOutputsCount());
    }

    @Test
    public void testInfer_existingOverload_stillWorks() {
        InferInput input = new InferInput("input0", new long[]{1}, TritonDataType.INT32);
        input.setData(new int[]{42});

        ModelInferResponse responseProto = ModelInferResponse.newBuilder()
                .setModelName("test_model")
                .build();

        when(blockingStub.modelInfer(any(ModelInferRequest.class))).thenReturn(responseProto);

        InferResult result = client.infer("test_model", null, List.of(input), null);

        assertNotNull(result);
        assertEquals("test_model", result.getModelName());
    }

    // ==================== inferAsync with requested outputs ====================

    @Test
    public void testInferAsync_withRequestedOutputs() throws Exception {
        InferInput input = new InferInput("input0", new long[]{1}, TritonDataType.FP32);
        input.setData(new float[]{1.0f});

        ModelInferResponse responseProto = ModelInferResponse.newBuilder()
                .setModelName("test_model")
                .build();

        doAnswer(invocation -> {
            StreamObserver<ModelInferResponse> observer = invocation.getArgument(1);
            observer.onNext(responseProto);
            return null;
        }).when(asyncStub).modelInfer(any(ModelInferRequest.class), any(StreamObserver.class));

        List<InferRequestedOutput> outputs = List.of(InferRequestedOutput.of("output_only"));

        CompletableFuture<InferResult> future = client.inferAsync(
                "test_model", null, List.of(input), outputs, null);

        InferResult result = future.get(5, TimeUnit.SECONDS);
        assertNotNull(result);

        ArgumentCaptor<ModelInferRequest> captor = ArgumentCaptor.forClass(ModelInferRequest.class);
        verify(asyncStub).modelInfer(captor.capture(), any(StreamObserver.class));

        assertEquals(1, captor.getValue().getOutputsCount());
        assertEquals("output_only", captor.getValue().getOutputs(0).getName());
    }

    // ==================== InferRequestedOutput unit tests ====================

    @Test
    public void testInferRequestedOutput_of() {
        InferRequestedOutput output = InferRequestedOutput.of("embeddings");
        assertEquals("embeddings", output.getName());
        assertTrue(output.getParameters().isEmpty());
        assertEquals(false, output.hasParameters());
    }

    @Test
    public void testInferRequestedOutput_builder() {
        InferRequestedOutput output = new InferRequestedOutput.Builder("output0")
                .addParameter("key_str", "value")
                .addParameter("key_long", 42L)
                .addParameter("key_bool", true)
                .addParameter("key_double", 3.14)
                .build();

        assertEquals("output0", output.getName());
        assertTrue(output.hasParameters());
        assertEquals(4, output.getParameters().size());
        assertEquals("value", output.getParameters().get("key_str"));
        assertEquals(42L, output.getParameters().get("key_long"));
        assertEquals(true, output.getParameters().get("key_bool"));
        assertEquals(3.14, output.getParameters().get("key_double"));
    }

    @Test(expected = NullPointerException.class)
    public void testInferRequestedOutput_nullName_throws() {
        InferRequestedOutput.of(null);
    }

    @Test(expected = NullPointerException.class)
    public void testInferRequestedOutput_builderNullName_throws() {
        new InferRequestedOutput.Builder(null);
    }
}
