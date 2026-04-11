package com.gencior.triton.integration;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

import com.gencior.triton.core.pojo.TritonServerMetadata;

/**
 * HTTP integration tests for server health, readiness and metadata.
 * Mirrors TritonServerIT (gRPC) for full parity.
 */
class TritonHttpServerIT extends AbstractTritonHttpIntegrationTest {

    @Test
    void isServerLive_shouldReturnTrue() {
        assertTrue(client.isServerLive());
    }

    @Test
    void isServerReady_shouldReturnTrue() {
        assertTrue(client.isServerReady());
    }

    @Test
    void getServerMetadata_shouldReturnValidMetadata() {
        TritonServerMetadata metadata = client.getServerMetadata();

        assertNotNull(metadata);
        assertNotNull(metadata.getName());
        assertNotNull(metadata.getVersion());
        assertFalse(metadata.getName().isEmpty());
        assertFalse(metadata.getVersion().isEmpty());
        assertNotNull(metadata.getExtensions());
        assertFalse(metadata.getExtensions().isEmpty());
    }
}
