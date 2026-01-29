package com.gencior.core.pojo;

import java.util.Collections;
import java.util.List;

import inference.GrpcService;

/**
 * Encapsulates metadata information about a Triton Inference Server instance.
 *
 * <p>This class represents server-level metadata such as the server name, version, and supported
 * extensions. This information is retrieved from the Triton Inference Server via the gRPC API.
 *
 * <p>This is an immutable object that wraps the gRPC message {@code ServerMetadataResponse}.
 *
 * @author sachachoumiloff
 * @since 1.0.0
 */
public final class TritonServerMetadata {

    private final String name;
    private final String version;
    private final List<String> extensions;

    private TritonServerMetadata(String name, String version, List<String> extensions) {
        this.name = name;
        this.version = version;
        this.extensions = extensions != null ? List.copyOf(extensions) : Collections.emptyList();
    }

    public static TritonServerMetadata fromProto(GrpcService.ServerMetadataResponse response) {
        return new TritonServerMetadata(
                response.getName(),
                response.getVersion(),
                response.getExtensionsList()
        );
    }

    /**
     * Checks whether the server supports a specific extension.
     *
     * @param extension the name of the extension to check
     * @return {@code true} if the server supports the extension, {@code false} otherwise
     */
    public boolean supportsExtension(String extension) {
        return extensions.contains(extension);
    }

    /**
     * Returns the name of the Triton server.
     *
     * @return the server name
     */
    public String getName() {
        return name;
    }

    /**
     * Returns the version of the Triton server.
     *
     * @return the server version string
     */
    public String getVersion() {
        return version;
    }

    /**
     * Returns an unmodifiable list of extensions supported by the server.
     *
     * <p>The returned list is immutable and cannot be modified.
     *
     * @return an immutable list of supported extension names
     */
    public List<String> getExtensions() {
        return extensions;
    }

    @Override
    public String toString() {
        return String.format("TritonServerMetadata[name=%s, version=%s, extensions=%s]",
                name, version, extensions);
    }
}
