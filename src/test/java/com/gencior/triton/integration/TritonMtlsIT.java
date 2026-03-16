package com.gencior.triton.integration;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.time.Duration;
import java.util.List;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.testcontainers.containers.BindMode;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.wait.strategy.Wait;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import com.gencior.triton.config.TritonClientConfig;
import com.gencior.triton.core.InferInput;
import com.gencior.triton.core.InferResult;
import com.gencior.triton.core.TritonDataType;
import com.gencior.triton.grpc.TritonGrpcClient;

/**
 * Integration tests for mutual TLS (mTLS).
 *
 * <p>The Triton container is started with {@code --grpc-use-ssl-mutual=1}
 * so the gRPC endpoint requires the client to present a valid certificate
 * signed by the trusted CA.
 */
@Testcontainers
class TritonMtlsIT {

    private static final String TRITON_IMAGE = "nvcr.io/nvidia/tritonserver:25.02-py3";

    private static final String PROJECT_ROOT = new File(System.getProperty("user.dir"))
            .getAbsoluteFile()
            .getAbsolutePath();

    private static final String MODELS_PATH = PROJECT_ROOT + "/dev/models_cpu";
    private static final String CERTS_PATH = PROJECT_ROOT + "/dev/certs";

    private static final File CA_CERT = new File(CERTS_PATH, "ca.pem");
    private static final File CLIENT_CERT = new File(CERTS_PATH, "client.pem");
    private static final File CLIENT_KEY = new File(CERTS_PATH, "client.key");
    private static final File WRONG_CA_CERT = new File(CERTS_PATH, "wrong-ca.pem");

    @Container
    protected static final GenericContainer<?> tritonContainer = new GenericContainer<>(TRITON_IMAGE)
            .withFileSystemBind(MODELS_PATH, "/models", BindMode.READ_ONLY)
            .withFileSystemBind(CERTS_PATH, "/certs", BindMode.READ_ONLY)
            .withExposedPorts(8000, 8001)
            .withCommand(
                    "tritonserver",
                    "--model-repository=/models",
                    "--model-control-mode=explicit",
                    "--load-model=identity_fp32",
                    "--grpc-use-ssl=1",
                    "--grpc-use-ssl-mutual=1",
                    "--grpc-server-cert=/certs/server.pem",
                    "--grpc-server-key=/certs/server.key",
                    "--grpc-root-cert=/certs/ca.pem",
                    "--log-verbose=1"
            )
            .withCreateContainerCmdModifier(cmd ->
                    cmd.getHostConfig().withShmSize(2147483648L)
            )
            .waitingFor(
                    Wait.forHttp("/v2/health/ready")
                            .forPort(8000)
                            .withStartupTimeout(Duration.ofSeconds(120))
            );

    private static TritonGrpcClient mtlsClient;

    @BeforeAll
    static void initClient() {
        String grpcTarget = tritonContainer.getHost() + ":" + tritonContainer.getMappedPort(8001);
        TritonClientConfig config = new TritonClientConfig.Builder(grpcTarget)
                .timeout(30000)
                .tlsEnabled(true)
                .trustCertFile(CA_CERT)
                .clientCertFile(CLIENT_CERT)
                .clientKeyFile(CLIENT_KEY)
                .build();
        mtlsClient = new TritonGrpcClient(config);
    }

    @AfterAll
    static void closeClient() throws Exception {
        if (mtlsClient != null) {
            mtlsClient.close();
        }
    }

    // ==================== Successful mTLS ====================

    @Test
    void mtls_serverLive_shouldWork() {
        assertTrue(mtlsClient.isServerLive());
    }

    @Test
    void mtls_serverReady_shouldWork() {
        assertTrue(mtlsClient.isServerReady());
    }

    @Test
    void mtls_infer_shouldWork() {
        float[] inputData = {10.0f, 20.0f, 30.0f};
        InferInput input = new InferInput("INPUT0", new long[]{3}, TritonDataType.FP32);
        input.setData(inputData);

        InferResult result = mtlsClient.infer("identity_fp32", List.of(input));

        assertNotNull(result);
        assertArrayEquals(inputData, result.asFloatArray("OUTPUT0"), 1e-6f);
    }

    @Test
    void mtls_getServerMetadata_shouldWork() {
        var metadata = mtlsClient.getServerMetadata();
        assertNotNull(metadata);
        assertNotNull(metadata.getName());
    }

    // ==================== Edge Cases ====================

    @Test
    void mtls_withoutClientCert_shouldFailHandshake() throws Exception {
        String grpcTarget = tritonContainer.getHost() + ":" + tritonContainer.getMappedPort(8001);
        TritonClientConfig config = new TritonClientConfig.Builder(grpcTarget)
                .timeout(5000)
                .tlsEnabled(true)
                .trustCertFile(CA_CERT)
                .build();

        try (TritonGrpcClient client = new TritonGrpcClient(config)) {
            assertThrows(io.grpc.StatusRuntimeException.class, client::isServerLive);
        }
    }

    @Test
    void mtls_wrongCA_shouldFailHandshake() throws Exception {
        String grpcTarget = tritonContainer.getHost() + ":" + tritonContainer.getMappedPort(8001);
        TritonClientConfig config = new TritonClientConfig.Builder(grpcTarget)
                .timeout(5000)
                .tlsEnabled(true)
                .trustCertFile(WRONG_CA_CERT)
                .clientCertFile(CLIENT_CERT)
                .clientKeyFile(CLIENT_KEY)
                .build();

        try (TritonGrpcClient client = new TritonGrpcClient(config)) {
            assertThrows(io.grpc.StatusRuntimeException.class, client::isServerLive);
        }
    }

    @Test
    void mtls_plaintextClient_shouldFail() throws Exception {
        String grpcTarget = tritonContainer.getHost() + ":" + tritonContainer.getMappedPort(8001);
        TritonClientConfig config = new TritonClientConfig.Builder(grpcTarget)
                .timeout(5000)
                .build();

        try (TritonGrpcClient client = new TritonGrpcClient(config)) {
            assertThrows(io.grpc.StatusRuntimeException.class, client::isServerLive);
        }
    }

    @Test
    void mtls_tlsWithoutClientCertOnMtlsServer_shouldFail() throws Exception {
        String grpcTarget = tritonContainer.getHost() + ":" + tritonContainer.getMappedPort(8001);
        TritonClientConfig config = new TritonClientConfig.Builder(grpcTarget)
                .timeout(5000)
                .tlsEnabled(true)
                .trustCertFile(CA_CERT)
                .build();

        try (TritonGrpcClient client = new TritonGrpcClient(config)) {
            assertThrows(io.grpc.StatusRuntimeException.class, client::isServerLive);
        }
    }
}
