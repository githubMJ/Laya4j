package com.laya4j.core;

/**
 * Laya4j 统一异常
 */
public class LayaException extends RuntimeException {
    public LayaException(String message) {
        super(message);
    }

    public LayaException(String message, Throwable cause) {
        super(message, cause);
    }
}