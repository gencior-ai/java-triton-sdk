package com.gencior.triton.integration;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.gencior.triton.core.pojo.TritonServerMetadata;

import org.junit.jupiter.api.Test;

class TritonServerIT extends AbstractTritonIntegrationTest {

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
