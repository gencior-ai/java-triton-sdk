package com.gencior.triton.exceptions;

/**
 *
 * @author sachachoumiloff
 */
public class TritonDataNotFoundException extends TritonInferException {
    public TritonDataNotFoundException(String inputName) {
        super("No data or raw content found for input: " + inputName + ". Did you call setData()?");
    }
}
