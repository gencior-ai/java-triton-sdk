package com.gencior.triton.exceptions;

/**
 * Exception thrown when a streaming inference call receives an error
 * from the Triton server via the {@code error_message} field of
 * {@code ModelStreamInferResponse}.
 *
 * @author sachachoumiloff
 * @since 1.0.0
 */
public class TritonStreamException extends TritonInferException {

    public TritonStreamException(String errorMessage) {
        super("Triton streaming error: " + errorMessage);
    }

    public TritonStreamException(String errorMessage, Throwable cause) {
        super("Triton streaming error: " + errorMessage, cause);
    }
}
