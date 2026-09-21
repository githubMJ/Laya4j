package com.laya4j.model;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import com.laya4j.core.Question;

import java.nio.file.Paths;
import java.util.*;

/**
 * 用真实 SequenceBuilder 跑出 4 个 case 的 ids/markers,跟 Python 对照
 */
public class DumpJavaIds {

    static long[] PYTHON_CASE_2_IDS = {2, 552, 6311, 2872, 235292, 64095, 17557, 24893, 186612, 235336,
                                       1, 4, 1566, 235292, 793, 235269, 573, 6218, 1721, 780, 3385,
                                       4, 1382, 235292, 7778, 235269, 573, 6218, 12723,
                                       1, 5651, 12459, 970, 20670, 7544, 1};
    static int[] PYTHON_CASE_2_MARKERS = {11, 21};

    public static void main(String[] args) throws Exception {
        HuggingFaceTokenizer t = HuggingFaceTokenizer.builder()
                .optTokenizerPath(Paths.get(
                    "/Users/aidenma/Documents/UGit/Python_Project/Laya4j/models/tokenizer/tokenizer.json"))
                .optAddSpecialTokens(false)
                .build();

        // 模拟 SequenceBuilder
        SequenceBuilder builder = new SequenceBuilder(t, 1024, 256);

        // Case 2 偏差最大
        Question q = Question.noul("refund", "用户是否要求退款?").build();
        String text = "Please cancel my subscription immediately";

        // 通过 reflection 拿 SequenceBuilder.build
        java.lang.reflect.Method buildMethod = SequenceBuilder.class.getDeclaredMethod("build", String.class, Question.class);
        buildMethod.setAccessible(true);
        Object bs = buildMethod.invoke(builder, text, q);

        java.lang.reflect.Method idsM = bs.getClass().getDeclaredMethod("ids");
        idsM.setAccessible(true);
        java.lang.reflect.Method markersM = bs.getClass().getDeclaredMethod("markers");
        markersM.setAccessible(true);

        long[] ids = (long[]) idsM.invoke(bs);
        int[] markers = (int[]) markersM.invoke(bs);

        System.out.println("Java ids   = " + Arrays.toString(ids));
        System.out.println("Java markers= " + Arrays.toString(markers));
        System.out.println("Py   ids    = " + Arrays.toString(PYTHON_CASE_2_IDS));
        System.out.println("Py   markers= " + Arrays.toString(PYTHON_CASE_2_MARKERS));
        System.out.println("ids len match? " + (ids.length == PYTHON_CASE_2_IDS.length));

        // 找出第一个差异
        int n = Math.min(ids.length, PYTHON_CASE_2_IDS.length);
        int diffIdx = -1;
        for (int i = 0; i < n; i++) {
            if (ids[i] != PYTHON_CASE_2_IDS[i]) {
                diffIdx = i;
                break;
            }
        }
        if (diffIdx >= 0) {
            System.out.printf("首个差异在 idx=%d: java=%d, py=%d%n", diffIdx, ids[diffIdx], PYTHON_CASE_2_IDS[diffIdx]);
        } else {
            System.out.println("ids 完全一致!");
        }
    }
}