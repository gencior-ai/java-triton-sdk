package com.gencior.exceptions;

/**
 *
 * @author sachachoumiloff
 */
public class TritonInferException extends RuntimeException {
    public TritonInferException(String message) {
        super(message);
    }

    public TritonInferException(String message, Throwable cause) {
        super(message, cause);
    }
}
