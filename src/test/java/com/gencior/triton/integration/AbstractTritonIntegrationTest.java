package com.gencior.triton.integration;

import java.io.File;
import java.time.Duration;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.testcontainers.containers.BindMode;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.wait.strategy.Wait;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import com.gencior.triton.config.TritonClientConfig;
import com.gencior.triton.grpc.TritonGrpcClient;

@Testcontainers
public abstract class AbstractTritonIntegrationTest {

    private static final String TRITON_IMAGE = "nvcr.io/nvidia/tritonserver:25.02-py3";

    @Container
    protected static final GenericContainer<?> tritonContainer;

    protected static TritonGrpcClient client;

    static {
        String projectRoot = new File(System.getProperty("user.dir"))
                .getAbsoluteFile()
                .getAbsolutePath();

        String modelsPath = projectRoot + "/dev/models_cpu";

        tritonContainer = new GenericContainer<>(TRITON_IMAGE)
                .withFileSystemBind(modelsPath, "/models", BindMode.READ_ONLY)
                .withExposedPorts(8000, 8001)
                .withCommand(
                        "tritonserver",
                        "--model-repository=/models",
                        "--model-control-mode=explicit",
                        "--load-model=identity_fp32",
                        "--load-model=identity_int32",
                        "--load-model=identity_string",
                        "--load-model=adder",
                        "--load-model=sleeper",
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
    }

    @BeforeAll
    static void initClient() {
        String grpcTarget = tritonContainer.getHost() + ":" + tritonContainer.getMappedPort(8001);
        TritonClientConfig config = new TritonClientConfig.Builder(grpcTarget)
                .timeout(30000)
                .build();
        client = new TritonGrpcClient(config);
    }

    @AfterAll
    static void closeClient() throws Exception {
        if (client != null) {
            client.close();
        }
    }
}
