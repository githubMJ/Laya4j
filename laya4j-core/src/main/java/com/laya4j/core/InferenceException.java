package com.laya4j.core;

/**
 * ONNX 前向推理执行失败时抛出。
 *
 * <p>典型触发场景:
 * <ul>
 *   <li>输入张量形状/类型与模型期望不匹配</li>
 *   <li>ONNX Runtime 在 {@code session.run()} 过程中抛出底层异常</li>
 * </ul>
 *
 * <p>与 {@link ModelLoadException}(模型已成功加载,但某一次调用推理失败)
 * 相区分——加载失败通常意味着模型不可用需要重新初始化,而推理失败可能是
 * 偶发的单次请求问题,调用方可以选择跳过该请求或重试。
 */
public class InferenceException extends LayaException {

    /** @param message 错误描述 */
    public InferenceException(String message) {
        super(message);
    }

    /**
     * @param message 错误描述
     * @param cause   根因异常(例如 {@code OrtException})
     */
    public InferenceException(String message, Throwable cause) {
        super(message, cause);
    }
}
