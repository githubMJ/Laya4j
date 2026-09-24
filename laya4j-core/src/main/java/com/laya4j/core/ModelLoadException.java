package com.laya4j.core;

/**
 * 模型或 tokenizer 加载失败时抛出。
 *
 * <p>典型触发场景:
 * <ul>
 *   <li>ONNX 权重文件不存在或路径错误</li>
 *   <li>tokenizer 目录缺失必要文件({@code tokenizer.json} 等)</li>
 *   <li>ONNX Runtime 创建 Session 失败(文件损坏、算子不受支持等)</li>
 * </ul>
 *
 * <p>与 {@link InferenceException}(推理阶段失败)、
 * {@link ModelFetchException}(下载阶段失败)相区分,
 * 便于调用方针对"模型是否可用"做专门的初始化重试/告警逻辑。
 */
public class ModelLoadException extends LayaException {

    /** @param message 错误描述 */
    public ModelLoadException(String message) {
        super(message);
    }

    /**
     * @param message 错误描述
     * @param cause   根因异常(例如 {@code OrtException}、{@code IOException})
     */
    public ModelLoadException(String message, Throwable cause) {
        super(message, cause);
    }
}
