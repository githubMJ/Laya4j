package com.laya4j.core;

/**
 * 从远程仓库(如 HuggingFace Hub)解析或下载模型权重失败时抛出。
 *
 * <p>典型触发场景:
 * <ul>
 *   <li>网络请求返回非 2xx/206 状态码(例如仓库路径错误导致 404)</li>
 *   <li>本地 {@code models/} 目录与缓存目录均未找到可用的 tokenizer 文件</li>
 *   <li>断点续传过程中的 IO 错误</li>
 * </ul>
 *
 * <p>与 {@link ModelLoadException} 相区分:本异常发生在"把权重文件
 * 准备到本地磁盘"这一步,而 {@link ModelLoadException} 发生在
 * "把本地磁盘上已存在的权重加载进内存"这一步——分开有助于调用方判断
 * 应该重试下载,还是应该检查本地文件是否损坏。
 */
public class ModelFetchException extends LayaException {

    /** @param message 错误描述 */
    public ModelFetchException(String message) {
        super(message);
    }

    /**
     * @param message 错误描述
     * @param cause   根因异常(例如 {@code IOException})
     */
    public ModelFetchException(String message, Throwable cause) {
        super(message, cause);
    }
}
