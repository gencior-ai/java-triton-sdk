package com.gencior.grpc;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import org.junit.Before;
import org.junit.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import com.gencior.config.TritonClientConfig;
import com.gencior.core.InferInput;
import com.gencior.core.InferResult;
import com.gencior.core.TritonDataType;
import com.gencior.core.pojo.TritonModelConfig;
import com.gencior.core.pojo.TritonModelMetadata;
import com.gencior.core.pojo.TritonModelStatistics;
import com.gencior.core.pojo.TritonRepositoryIndex;
import com.gencior.core.pojo.TritonServerMetadata;
import com.gencior.exceptions.TritonDataNotFoundException;
import com.gencior.grpc.TritonGrpcClient;

import inference.GRPCInferenceServiceGrpc.GRPCInferenceServiceBlockingStub;
import inference.GRPCInferenceServiceGrpc.GRPCInferenceServiceStub;
import inference.GrpcService.ModelConfigResponse;
import inference.GrpcService.ModelInferRequest;
import inference.GrpcService.ModelInferResponse;
import inference.GrpcService.ModelMetadataResponse;
import inference.GrpcService.ModelStatistics;
import inference.GrpcService.ModelStatisticsResponse;
import inference.GrpcService.RepositoryIndexResponse;
import inference.GrpcService.RepositoryModelLoadResponse;
import inference.GrpcService.RepositoryModelUnloadResponse;
import inference.GrpcService.ServerLiveResponse;
import inference.GrpcService.ServerMetadataResponse;
import inference.GrpcService.ServerReadyResponse;
import inference.ModelConfigOuterClass.ModelConfig;
import io.grpc.ManagedChannel;
import io.grpc.stub.StreamObserver;

public class TritonGrpcClientTest {

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
        when(blockingStub.withDeadlineAfter(1000L, java.util.concurrent.TimeUnit.MILLISECONDS)).thenReturn(blockingStub);

        client = new TritonGrpcClient(config, channel, blockingStub, asyncStub);
    }

    @Test
    public void testIsServerLive_True() {
        ServerLiveResponse response = ServerLiveResponse.newBuilder().setLive(true).build();
        when(blockingStub.serverLive(any())).thenReturn(response);

        assertTrue(client.isServerLive());
    }

    @Test
    public void testIsServerLive_False() {
        ServerLiveResponse response = ServerLiveResponse.newBuilder().setLive(false).build();
        when(blockingStub.serverLive(any())).thenReturn(response);

        assertFalse(client.isServerLive());
    }

    @Test
    public void testIsServerReady_True() {
        ServerReadyResponse response = ServerReadyResponse.newBuilder().setReady(true).build();
        when(blockingStub.serverReady(any())).thenReturn(response);

        assertTrue(client.isServerReady());
    }

    @Test
    public void testGetServerMetadata() {
        ServerMetadataResponse proto = ServerMetadataResponse.newBuilder()
                .setName("triton_server_test")
                .setVersion("2.34.0")
                .addExtensions("grpc_extension")
                .build();
        when(blockingStub.serverMetadata(any())).thenReturn(proto);

        TritonServerMetadata metadata = client.getServerMetadata();

        assertNotNull(metadata);
        assertEquals("triton_server_test", metadata.getName());
        assertEquals("2.34.0", metadata.getVersion());
        assertEquals(1, metadata.getExtensions().size());
    }

    @Test
    public void testGetModelMetadata() {
        ModelMetadataResponse proto = ModelMetadataResponse.newBuilder()
                .setName("ensemble_model")
                .addVersions("1")
                .addVersions("2")
                .setPlatform("ensemble")
                .build();
        when(blockingStub.modelMetadata(any())).thenReturn(proto);

        TritonModelMetadata metadata = client.getModelMetadata("ensemble_model", "1");

        assertNotNull(metadata);
        assertEquals("ensemble_model", metadata.getName());
        assertEquals(2, metadata.getVersions().size());
        assertEquals("ensemble", metadata.getPlatform());
    }

    @Test
    public void testGetModelConfig() {
        ModelConfig configProto = ModelConfig.newBuilder()
                .setName("resnet50")
                .setPlatform("onnxruntime_onnx")
                .setMaxBatchSize(16)
                .build();
        ModelConfigResponse response = ModelConfigResponse.newBuilder()
                .setConfig(configProto)
                .build();
        when(blockingStub.modelConfig(any())).thenReturn(response);

        TritonModelConfig modelConfig = client.getModelConfig("resnet50");

        assertNotNull(modelConfig);
        assertEquals("resnet50", modelConfig.getName());
        assertEquals(16, modelConfig.getMaxBatchSize());
        assertEquals("onnxruntime_onnx", modelConfig.getPlatform());
    }

    @Test
    public void testGetModelRepositoryIndex() {
        RepositoryIndexResponse proto = RepositoryIndexResponse.newBuilder()
                .addModels(RepositoryIndexResponse.ModelIndex.newBuilder().setName("model_a").setState("READY").build())
                .addModels(RepositoryIndexResponse.ModelIndex.newBuilder().setName("model_b").setState("UNAVAILABLE").build())
                .build();
        when(blockingStub.repositoryIndex(any())).thenReturn(proto);

        TritonRepositoryIndex index = client.getModelRepositoryIndex();

        assertNotNull(index);
        assertEquals(2, index.getModels().size());
        assertEquals("model_a", index.getModels().get(0).getName());
    }

    @Test
    public void testLoadModel() {
        RepositoryModelLoadResponse proto = RepositoryModelLoadResponse.newBuilder().build();
        when(blockingStub.repositoryModelLoad(any())).thenReturn(proto);

        client.loadModel("test_model");

        verify(blockingStub, times(1)).repositoryModelLoad(any());
    }

    @Test
    public void testUnLoadModel() {
        RepositoryModelUnloadResponse proto = RepositoryModelUnloadResponse.newBuilder().build();
        when(blockingStub.repositoryModelUnload(any())).thenReturn(proto);

        client.unLoadModel("test_model");

        verify(blockingStub, times(1)).repositoryModelUnload(any());
    }

    @Test
    public void testGetInferenceStatistics() {
        ModelStatistics statsProto = ModelStatistics.newBuilder()
                .setName("resnet")
                .setVersion("1")
                .setInferenceCount(100)
                .setExecutionCount(10)
                .build();
        ModelStatisticsResponse response = ModelStatisticsResponse.newBuilder()
                .addModelStats(statsProto)
                .build();
        when(blockingStub.modelStatistics(any())).thenReturn(response);

        List<TritonModelStatistics> stats = client.getInferenceStatistics("resnet", "1");

        assertNotNull(stats);
        assertEquals(1, stats.size());
        assertEquals("resnet", stats.get(0).getName());
        assertEquals(100L, stats.get(0).getInferenceCount());
        assertEquals(10.0, stats.get(0).getBatchingEfficiency(), 0.001);
    }

    @Test
    public void testInfer_Sync_Success() {
        InferInput input = new InferInput("input0", new long[]{1}, TritonDataType.INT32);
        input.setData(new int[]{42});

        ModelInferResponse responseProto = ModelInferResponse.newBuilder()
                .setModelName("test_model")
                .setId("req_123")
                .build();

        when(blockingStub.modelInfer(any(ModelInferRequest.class))).thenReturn(responseProto);

        InferResult result = client.infer("test_model", List.of(input));

        assertNotNull(result);
        assertEquals("test_model", result.getModelName());
        assertEquals("req_123", result.getRequestId());
        verify(blockingStub, times(1)).modelInfer(any(ModelInferRequest.class));
    }

    @Test(expected = TritonDataNotFoundException.class)
    public void testInfer_MissingData_ThrowsException() {
        InferInput input = new InferInput("input0", new long[]{1}, TritonDataType.INT32);
        client.infer("test_model", List.of(input));
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testInferAsync_Success() throws Exception {
        InferInput input = new InferInput("input0", new long[]{1}, TritonDataType.INT32);
        input.setData(new int[]{42});

        ModelInferResponse responseProto = ModelInferResponse.newBuilder()
                .setModelName("test_model_async")
                .build();

        doAnswer(invocation -> {
            StreamObserver<ModelInferResponse> observer = invocation.getArgument(1);
            observer.onNext(responseProto);
            return null;
        }).when(asyncStub).modelInfer(any(ModelInferRequest.class), any(StreamObserver.class));

        CompletableFuture<InferResult> future = client.inferAsync("test_model_async", List.of(input));
        InferResult result = future.get(5, TimeUnit.SECONDS);

        assertNotNull(result);
        assertEquals("test_model_async", result.getModelName());
        assertTrue(future.isDone());
    }

    @Test
    public void testClose() throws Exception {
        when(channel.shutdown()).thenReturn(channel);
        when(channel.isShutdown()).thenReturn(false);

        client.close();

        verify(channel).shutdown();
        verify(channel).awaitTermination(5, TimeUnit.SECONDS);
    }
}
