package com.laya4j.model;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;

import java.nio.file.Paths;
import java.util.Arrays;

/**
 * 对照 Python 端的 build_sequence 输出,定位 Java 端差异
 *
 * 把 4 个测试用例的 Java ids / markers 跟 Python baseline 对比
 */
public class AlignDiagnose {

    // Python 端实测基线(同 ONNX 模型 + 同输入)
    static final String[][] PY_IDS = {
        // Case 1: 我要求立即退款,这是第三次投诉了
        // Case 2: Please cancel my subscription immediately
        // Case 3: 服务挂了快两小时了,严重影响业务
        // Case 4: Where is my order #1234? It has been 5 days.
        // 由外部脚本生成,这里先留空
    };
    static final int[][] PY_MARKERS = {};

    public static void main(String[] args) throws Exception {
        HuggingFaceTokenizer t = HuggingFaceTokenizer.builder()
                .optTokenizerPath(Paths.get(
                    "/Users/aidenma/Documents/UGit/Python_Project/Laya4j/models/tokenizer/tokenizer.json"))
                .optAddSpecialTokens(false)
                .build();

        String[] inputs = {
            "我要求立即退款,这是第三次投诉了",
            "Please cancel my subscription immediately",
            "服务挂了快两小时了,严重影响业务",
            "Where is my order #1234? It has been 5 days."
        };

        String ins = "用户是否要求退款?";

        for (String text : inputs) {
            String headText = "noul question: " + ins;
            long[] head = t.encode(headText).getIds();

            // false/true description
            long[] optFalse = t.encode(" false: no, the statement does not hold").getIds();
            long[] optTrue  = t.encode(" true:  yes, the statement holds").getIds();

            // state
            long[] st = t.encode(text).getIds();

            System.out.println("=== " + text);
            System.out.println("  head_ids    =" + Arrays.toString(head));
            System.out.println("  opt[false]  =" + Arrays.toString(optFalse));
            System.out.println("  opt[true]   =" + Arrays.toString(optTrue));
            System.out.println("  state       =" + Arrays.toString(st));
        }
    }
}