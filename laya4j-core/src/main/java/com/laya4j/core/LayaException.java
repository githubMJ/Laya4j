package com.laya4j.core;

/**
 * Laya4j 统一异常根类型。
 *
 * <p>所有由本库主动抛出的运行时异常都继承自该类,便于调用方用一个
 * {@code catch (LayaException e)} 统一捕获库相关的错误,同时通过
 * 具体子类区分错误发生的阶段,做更精细的编程式处理:
 * <ul>
 *   <li>{@link ModelLoadException}  —— 模型/tokenizer 加载阶段失败</li>
 *   <li>{@link InferenceException}  —— ONNX 前向推理阶段失败</li>
 *   <li>{@link ModelFetchException} —— 模型权重下载/解析阶段失败</li>
 * </ul>
 * 不属于以上三类的配置/校验类错误(例如 {@link Question} 构造参数非法)
 * 会直接抛出本类或 {@link IllegalStateException} 等标准 JDK 异常。
 */
public class LayaException extends RuntimeException {

    /**
     * @param message 错误描述
     */
    public LayaException(String message) {
        super(message);
    }

    /**
     * @param message 错误描述
     * @param cause   根因异常(例如底层 ONNX Runtime / IO 异常)
     */
    public LayaException(String message, Throwable cause) {
        super(message, cause);
    }
}
