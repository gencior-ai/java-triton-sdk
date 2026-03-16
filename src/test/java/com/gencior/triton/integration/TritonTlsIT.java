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
 * Integration tests for one-way TLS (server certificate verification).
 *
 * <p>The Triton container is started with {@code --grpc-use-ssl} so the gRPC
 * endpoint requires TLS. The client must trust the server certificate via the
 * root CA.
 */
@Testcontainers
class TritonTlsIT {

    private static final String TRITON_IMAGE = "nvcr.io/nvidia/tritonserver:25.02-py3";

    private static final String PROJECT_ROOT = new File(System.getProperty("user.dir"))
            .getAbsoluteFile()
            .getAbsolutePath();

    private static final String MODELS_PATH = PROJECT_ROOT + "/dev/models_cpu";
    private static final String CERTS_PATH = PROJECT_ROOT + "/dev/certs";

    private static final File CA_CERT = new File(CERTS_PATH, "ca.pem");
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
                    "--grpc-server-cert=/certs/server.pem",
                    "--grpc-server-key=/certs/server.key",
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

    private static TritonGrpcClient tlsClient;

    @BeforeAll
    static void initClient() {
        String grpcTarget = tritonContainer.getHost() + ":" + tritonContainer.getMappedPort(8001);
        TritonClientConfig config = new TritonClientConfig.Builder(grpcTarget)
                .timeout(30000)
                .tlsEnabled(true)
                .trustCertFile(CA_CERT)
                .build();
        tlsClient = new TritonGrpcClient(config);
    }

    @AfterAll
    static void closeClient() throws Exception {
        if (tlsClient != null) {
            tlsClient.close();
        }
    }

    // ==================== Successful TLS ====================

    @Test
    void tls_serverLive_shouldWork() {
        assertTrue(tlsClient.isServerLive());
    }

    @Test
    void tls_serverReady_shouldWork() {
        assertTrue(tlsClient.isServerReady());
    }

    @Test
    void tls_infer_shouldWork() {
        float[] inputData = {1.0f, 2.0f, 3.0f};
        InferInput input = new InferInput("INPUT0", new long[]{3}, TritonDataType.FP32);
        input.setData(inputData);

        InferResult result = tlsClient.infer("identity_fp32", List.of(input));

        assertNotNull(result);
        assertArrayEquals(inputData, result.asFloatArray("OUTPUT0"), 1e-6f);
    }

    @Test
    void tls_getServerMetadata_shouldWork() {
        var metadata = tlsClient.getServerMetadata();
        assertNotNull(metadata);
        assertNotNull(metadata.getName());
        assertNotNull(metadata.getVersion());
    }

    // ==================== TLS with JVM default truststore ====================

    @Test
    void tls_withoutTrustCert_shouldFailOnSelfSigned() throws Exception {
        String grpcTarget = tritonContainer.getHost() + ":" + tritonContainer.getMappedPort(8001);
        TritonClientConfig config = new TritonClientConfig.Builder(grpcTarget)
                .timeout(5000)
                .tlsEnabled(true)
                .build();

        try (TritonGrpcClient client = new TritonGrpcClient(config)) {
            assertThrows(io.grpc.StatusRuntimeException.class, client::isServerLive);
        }
    }

    // ==================== Edge Cases ====================

    @Test
    void tls_wrongCA_shouldFailHandshake() throws Exception {
        String grpcTarget = tritonContainer.getHost() + ":" + tritonContainer.getMappedPort(8001);
        TritonClientConfig config = new TritonClientConfig.Builder(grpcTarget)
                .timeout(5000)
                .tlsEnabled(true)
                .trustCertFile(WRONG_CA_CERT)
                .build();

        try (TritonGrpcClient client = new TritonGrpcClient(config)) {
            assertThrows(io.grpc.StatusRuntimeException.class, client::isServerLive);
        }
    }

    @Test
    void tls_plaintextClientToTlsServer_shouldFail() throws Exception {
        String grpcTarget = tritonContainer.getHost() + ":" + tritonContainer.getMappedPort(8001);
        TritonClientConfig config = new TritonClientConfig.Builder(grpcTarget)
                .timeout(5000)
                .build();

        try (TritonGrpcClient client = new TritonGrpcClient(config)) {
            assertThrows(io.grpc.StatusRuntimeException.class, client::isServerLive);
        }
    }
}
