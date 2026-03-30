package com.gencior.triton.config;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class TritonClientConfigTest {

    @TempDir
    Path tempDir;

    private File createDummyFile(String name) throws IOException {
        Path file = tempDir.resolve(name);
        Files.writeString(file, "dummy-content");
        return file.toFile();
    }

    // --- Default config (plaintext) ---

    @Test
    void build_default_shouldBePlaintextWithNullCerts() {
        TritonClientConfig config = new TritonClientConfig.Builder("localhost:8001").build();

        assertFalse(config.isTlsEnabled());
        assertNull(config.getTrustCertFile());
        assertNull(config.getClientCertFile());
        assertNull(config.getClientKeyFile());
        assertEquals("localhost:8001", config.getUrl());
        assertEquals(60000, config.getDefaultTimeoutMs());
    }

    // --- TLS enabled (one-way, JVM truststore) ---

    @Test
    void build_tlsEnabledNoTrustCert_shouldSucceed() {
        TritonClientConfig config = new TritonClientConfig.Builder("localhost:8001")
                .tlsEnabled(true)
                .build();

        assertTrue(config.isTlsEnabled());
        assertNull(config.getTrustCertFile());
    }

    // --- TLS enabled with trust cert ---

    @Test
    void build_tlsEnabledWithTrustCert_shouldSucceed() throws IOException {
        File caCert = createDummyFile("ca.pem");

        TritonClientConfig config = new TritonClientConfig.Builder("localhost:8001")
                .tlsEnabled(true)
                .trustCertFile(caCert)
                .build();

        assertTrue(config.isTlsEnabled());
        assertEquals(caCert, config.getTrustCertFile());
    }

    // --- mTLS (all certs provided) ---

    @Test
    void build_mtls_shouldSucceed() throws IOException {
        File caCert = createDummyFile("ca.pem");
        File clientCert = createDummyFile("client.crt");
        File clientKey = createDummyFile("client.key");

        TritonClientConfig config = new TritonClientConfig.Builder("localhost:8001")
                .tlsEnabled(true)
                .trustCertFile(caCert)
                .clientCertFile(clientCert)
                .clientKeyFile(clientKey)
                .build();

        assertTrue(config.isTlsEnabled());
        assertNotNull(config.getTrustCertFile());
        assertNotNull(config.getClientCertFile());
        assertNotNull(config.getClientKeyFile());
    }

    // --- Validation errors ---

    @Test
    void build_trustCertWithoutTls_shouldThrow() throws IOException {
        File caCert = createDummyFile("ca.pem");

        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () ->
                new TritonClientConfig.Builder("localhost:8001")
                        .trustCertFile(caCert)
                        .build()
        );
        assertTrue(ex.getMessage().contains("TLS must be enabled"));
    }

    @Test
    void build_clientCertWithoutTls_shouldThrow() throws IOException {
        File clientCert = createDummyFile("client.crt");
        File clientKey = createDummyFile("client.key");

        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () ->
                new TritonClientConfig.Builder("localhost:8001")
                        .clientCertFile(clientCert)
                        .clientKeyFile(clientKey)
                        .build()
        );
        assertTrue(ex.getMessage().contains("TLS must be enabled"));
    }

    @Test
    void build_clientCertWithoutKey_shouldThrow() {
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () ->
                new TritonClientConfig.Builder("localhost:8001")
                        .tlsEnabled(true)
                        .clientCertFile(new File("dummy"))
                        .build()
        );
        assertTrue(ex.getMessage().contains("Both clientCertFile and clientKeyFile"));
    }

    @Test
    void build_clientKeyWithoutCert_shouldThrow() {
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () ->
                new TritonClientConfig.Builder("localhost:8001")
                        .tlsEnabled(true)
                        .clientKeyFile(new File("dummy"))
                        .build()
        );
        assertTrue(ex.getMessage().contains("Both clientCertFile and clientKeyFile"));
    }

    @Test
    void build_nonexistentTrustCert_shouldThrow() {
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () ->
                new TritonClientConfig.Builder("localhost:8001")
                        .tlsEnabled(true)
                        .trustCertFile(new File("/nonexistent/ca.pem"))
                        .build()
        );
        assertTrue(ex.getMessage().contains("not found or not readable"));
    }

    @Test
    void build_nonexistentClientCerts_shouldThrow() throws IOException {
        File clientCert = createDummyFile("client.crt");

        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () ->
                new TritonClientConfig.Builder("localhost:8001")
                        .tlsEnabled(true)
                        .clientCertFile(clientCert)
                        .clientKeyFile(new File("/nonexistent/client.key"))
                        .build()
        );
        assertTrue(ex.getMessage().contains("not found or not readable"));
    }

    // --- Existing validations still work ---

    @Test
    void build_nullUrl_shouldThrow() {
        assertThrows(IllegalArgumentException.class, () ->
                new TritonClientConfig.Builder(null).build()
        );
    }

    @Test
    void build_zeroTimeout_shouldThrow() {
        assertThrows(IllegalArgumentException.class, () ->
                new TritonClientConfig.Builder("localhost:8001")
                        .timeout(0)
                        .build()
        );
    }

    // --- Builder method chaining ---

    @Test
    void build_allOptions_shouldReturnCorrectValues() throws IOException {
        File caCert = createDummyFile("ca.pem");
        File clientCert = createDummyFile("client.crt");
        File clientKey = createDummyFile("client.key");

        TritonClientConfig config = new TritonClientConfig.Builder("myserver:443")
                .timeout(5000)
                .maxInboundMessageSize(1024)
                .tlsEnabled(true)
                .trustCertFile(caCert)
                .clientCertFile(clientCert)
                .clientKeyFile(clientKey)
                .build();

        assertEquals("myserver:443", config.getUrl());
        assertEquals(5000, config.getDefaultTimeoutMs());
        assertEquals(1024, config.getMaxInboundMessageSize());
        assertTrue(config.isTlsEnabled());
        assertEquals(caCert, config.getTrustCertFile());
        assertEquals(clientCert, config.getClientCertFile());
        assertEquals(clientKey, config.getClientKeyFile());
    }
}
