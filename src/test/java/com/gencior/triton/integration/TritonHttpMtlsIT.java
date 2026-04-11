package com.gencior.triton.integration;

import static org.junit.jupiter.api.Assertions.*;

import java.io.File;
import java.time.Duration;
import java.util.List;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.testcontainers.containers.BindMode;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.Network;
import org.testcontainers.containers.wait.strategy.Wait;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import com.gencior.triton.config.TritonClientConfig;
import com.gencior.triton.core.InferInput;
import com.gencior.triton.core.InferResult;
import com.gencior.triton.core.TritonDataType;
import com.gencior.triton.core.pojo.TritonServerMetadata;
import com.gencior.triton.http.TritonHttpClient;

/**
 * HTTP integration tests for mutual TLS (mTLS).
 *
 * <p>Architecture: Client --HTTPS+mTLS--> nginx (TLS termination) --HTTP--> Triton.
 * The nginx container is configured with {@code ssl_verify_client on}, requiring
 * the client to present a certificate signed by the trusted CA.</p>
 */
@Testcontainers
class TritonHttpMtlsIT {

    private static final String TRITON_IMAGE = "nvcr.io/nvidia/tritonserver:25.02-py3";
    private static final String NGINX_IMAGE = "nginx:1.27-alpine";

    private static final String PROJECT_ROOT = new File(System.getProperty("user.dir"))
            .getAbsoluteFile().getAbsolutePath();
    private static final String MODELS_PATH = PROJECT_ROOT + "/dev/models_cpu";
    private static final String CERTS_PATH = PROJECT_ROOT + "/dev/certs";
    private static final String NGINX_CONF_PATH = PROJECT_ROOT + "/dev/nginx/mtls.conf";

    private static final File CA_CERT = new File(CERTS_PATH, "ca.pem");
    private static final File CLIENT_CERT = new File(CERTS_PATH, "client.pem");
    private static final File CLIENT_KEY = new File(CERTS_PATH, "client.key");
    private static final File WRONG_CA_CERT = new File(CERTS_PATH, "wrong-ca.pem");

    private static final Network network = Network.newNetwork();

    @Container
    private static final GenericContainer<?> tritonContainer = new GenericContainer<>(TRITON_IMAGE)
            .withNetwork(network)
            .withNetworkAliases("triton")
            .withFileSystemBind(MODELS_PATH, "/models", BindMode.READ_ONLY)
            .withExposedPorts(8000)
            .withCommand(
                    "tritonserver",
                    "--model-repository=/models",
                    "--model-control-mode=explicit",
                    "--load-model=identity_fp32",
                    "--log-verbose=1"
            )
            .withCreateContainerCmdModifier(cmd ->
                    cmd.getHostConfig().withShmSize(2147483648L))
            .waitingFor(Wait.forHttp("/v2/health/ready")
                    .forPort(8000)
                    .withStartupTimeout(Duration.ofSeconds(120)));

    @Container
    private static final GenericContainer<?> nginxContainer = new GenericContainer<>(NGINX_IMAGE)
            .withNetwork(network)
            .withFileSystemBind(NGINX_CONF_PATH, "/etc/nginx/conf.d/default.conf", BindMode.READ_ONLY)
            .withFileSystemBind(CERTS_PATH, "/certs", BindMode.READ_ONLY)
            .withExposedPorts(443)
            .dependsOn(tritonContainer)
            .waitingFor(Wait.forListeningPort()
                    .withStartupTimeout(Duration.ofSeconds(30)));

    private static TritonHttpClient mtlsClient;

    @BeforeAll
    static void initClient() {
        String httpsTarget = nginxContainer.getHost() + ":" + nginxContainer.getMappedPort(443);
        TritonClientConfig config = new TritonClientConfig.Builder(httpsTarget)
                .timeout(30000)
                .tlsEnabled(true)
                .trustCertFile(CA_CERT)
                .clientCertFile(CLIENT_CERT)
                .clientKeyFile(CLIENT_KEY)
                .build();
        mtlsClient = new TritonHttpClient(config);
    }

    @AfterAll
    static void closeClient() throws Exception {
        if (mtlsClient != null) mtlsClient.close();
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
        TritonServerMetadata metadata = mtlsClient.getServerMetadata();
        assertNotNull(metadata);
        assertNotNull(metadata.getName());
    }

    // ==================== Edge Cases ====================

    @Test
    void mtls_withoutClientCert_shouldFail() throws Exception {
        String httpsTarget = nginxContainer.getHost() + ":" + nginxContainer.getMappedPort(443);
        TritonClientConfig config = new TritonClientConfig.Builder(httpsTarget)
                .timeout(5000)
                .tlsEnabled(true)
                .trustCertFile(CA_CERT)
                // No client cert → nginx rejects (ssl_verify_client on)
                .build();

        try (TritonHttpClient client = new TritonHttpClient(config)) {
            assertFalse(client.isServerLive());
        }
    }

    @Test
    void mtls_wrongCA_shouldFail() throws Exception {
        String httpsTarget = nginxContainer.getHost() + ":" + nginxContainer.getMappedPort(443);
        TritonClientConfig config = new TritonClientConfig.Builder(httpsTarget)
                .timeout(5000)
                .tlsEnabled(true)
                .trustCertFile(WRONG_CA_CERT)
                .clientCertFile(CLIENT_CERT)
                .clientKeyFile(CLIENT_KEY)
                .build();

        try (TritonHttpClient client = new TritonHttpClient(config)) {
            assertFalse(client.isServerLive());
        }
    }

    @Test
    void mtls_plaintextClient_shouldFail() throws Exception {
        String httpsTarget = nginxContainer.getHost() + ":" + nginxContainer.getMappedPort(443);
        TritonClientConfig config = new TritonClientConfig.Builder(httpsTarget)
                .timeout(5000)
                .build();

        try (TritonHttpClient client = new TritonHttpClient(config)) {
            assertFalse(client.isServerLive());
        }
    }

    @Test
    void mtls_tlsWithoutClientCertOnMtlsServer_shouldFail() throws Exception {
        String httpsTarget = nginxContainer.getHost() + ":" + nginxContainer.getMappedPort(443);
        TritonClientConfig config = new TritonClientConfig.Builder(httpsTarget)
                .timeout(5000)
                .tlsEnabled(true)
                .trustCertFile(CA_CERT)
                .build();

        try (TritonHttpClient client = new TritonHttpClient(config)) {
            // Trust cert is correct but no client cert → nginx rejects
            assertFalse(client.isServerLive());
        }
    }
}
