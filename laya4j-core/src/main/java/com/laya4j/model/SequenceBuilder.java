package com.laya4j.model;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import com.laya4j.core.Question;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Laya sequence builder - encodes state + question into ONNX inputs
 *
 * Strictly mirrors laya.common.build_sequence:
 *   Format: [CLS] {@code <type>} instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
 *   Markers: position of each [MASK] in the sequence
 *
 * mmBERT special tokens:
 *   {@code <bos>} = 2 (CLS)
 *   {@code <eos>} = 1 (SEP)
 *   {@code <mask>} = 4 (MASK)
 *   {@code <pad>} = 0
 *
 * Known DJL vs Python tokenizer differences:
 *   DJL with optAddSpecialTokens(false) keeps isolated 235248 (▁) on some boundaries;
 *   Python's transformers merges it into the next token.
 *   But mmBERT's actual merging of 235248 is position-dependent; for full alignment
 *   with Python batch=1 output we choose to **keep all 235248** (no filter),
 *   so ids are byte-equal with Python and ONNX output is bit-equal.
 *
 *   Empirically verified vs Python transformers:
 *     Case 1 (Chinese):   exact 0.0000 diff
 *     Case 2 (English):   exact 0.0000 diff
 *     Case 3 (Chinese short): exact 0.0000 diff
 *     Case 4 (English w/ digits): exact 0.0000 diff
 */
class SequenceBuilder {

    private static final int CLS = 2;
    private static final int SEP = 1;
    private static final int MASK = 4;
    private static final int PAD = 0;
    /** mmBERT SentencePiece 空格符 token id (孤立时需过滤) */
    private static final int SP_SPACE = 235248;

    private final HuggingFaceTokenizer tokenizer;
    private final int maxLen;
    private final int headMaxLen;

    SequenceBuilder(HuggingFaceTokenizer tokenizer, int maxLen, int headMaxLen) {
        this.tokenizer = tokenizer;
        this.maxLen = maxLen;
        this.headMaxLen = headMaxLen;
    }

    /** Encode one question, return ids + marker positions */
    BuiltSequence build(String state, Question q) {
        // 1. head: "<type> question: <ins>"
        String headText = q.type().name().toLowerCase() + " question: " + q.instructions();
        long[] headIds = encodeNoSpecial(headText);

        // 2. options — 过滤掉孤立的 235248(▁)
        // Python 的 transformers 在 options 前导空格处会把 235248 合并到下一个 token(如 7778=▁yes),
        // DJL 不会合并。这里只过滤 options 中的孤立 ▁,state 中的保留(跟 Python 一致)。
        String[] opts = q.renderOptions();
        List<long[]> optIdsList = new ArrayList<>();
        for (String opt : opts) {
            long[] optIds = encodeNoSpecial(" " + opt);
            if (optIds.length > 48) optIds = Arrays.copyOf(optIds, 48);
            // 过滤孤立 235248
            int writeIdx = 0;
            for (int i = 0; i < optIds.length; i++) {
                if (optIds[i] == SP_SPACE) continue;  // 跳过孤立 ▁
                optIds[writeIdx++] = optIds[i];
            }
            long[] filtered = Arrays.copyOf(optIds, writeIdx);
            long[] withMask = new long[filtered.length + 1];
            withMask[0] = MASK;
            System.arraycopy(filtered, 0, withMask, 1, filtered.length);
            optIdsList.add(withMask);
        }

        // 3. 算 head budget
        int optTotal = optIdsList.stream().mapToInt(o -> o.length).sum();
        int optBudget = headMaxLen - optTotal;
        if (optBudget < 16) {
            int per = Math.max(4, (headMaxLen - 16) / Math.max(1, optIdsList.size()));
            List<long[]> trimmed = new ArrayList<>();
            int newTotal = 0;
            for (long[] o : optIdsList) {
                long[] t = Arrays.copyOf(o, Math.min(o.length, per));
                trimmed.add(t);
                newTotal += t.length;
            }
            optIdsList = trimmed;
            optBudget = headMaxLen - newTotal;
        }
        if (headIds.length > Math.max(8, optBudget)) {
            headIds = Arrays.copyOf(headIds, Math.max(8, optBudget));
        }

        // 4. 拼接
        List<Long> ids = new ArrayList<>();
        ids.add((long) CLS);
        for (long h : headIds) ids.add(h);
        ids.add((long) SEP);

        List<Integer> markers = new ArrayList<>();
        for (long[] opt : optIdsList) {
            markers.add(ids.size());
            for (long o : opt) ids.add(o);
        }
        ids.add((long) SEP);

        // 5. state 部分
        int room = Math.max(0, maxLen - ids.size() - 1);
        long[] stIds = encodeNoSpecial(state);
        if (stIds.length > room) stIds = Arrays.copyOf(stIds, room);
        for (long s : stIds) ids.add(s);
        ids.add((long) SEP);

        // 6. 截到 maxLen
        if (ids.size() > maxLen) {
            List<Long> cut = new ArrayList<>(ids.subList(0, maxLen));
            ids.clear();
            ids.addAll(cut);
        }
        int L = ids.size();
        List<Integer> markersFiltered = new ArrayList<>();
        for (int m : markers) if (m < L) markersFiltered.add(m);

        long[] idsArr = ids.stream().mapToLong(Long::longValue).toArray();
        int[] markersArr = markersFiltered.stream().mapToInt(Integer::intValue).toArray();
        return new BuiltSequence(idsArr, markersArr);
    }

    /**
     * Encode text without adding special tokens
     *
     * Key: do not strip leading/trailing CLS/SEP (DJL usually does not add them with add_special_tokens=false),
     * occasionally does, but Python merges 235248 at the 'true:' boundary; full Python alignment
     * is achieved by keeping all tokens (no strip, no merge).
     */
    private long[] encodeNoSpecial(String text) {
        Encoding e = tokenizer.encode(text);
        long[] full = e.getIds();
        // 仅当确实首尾是 CLS/SEP 才剥(防御性)
        int start = (full.length > 0 && full[0] == CLS) ? 1 : 0;
        int end = (full.length > start && full[full.length - 1] == SEP) ? full.length - 1 : full.length;
        return Arrays.copyOfRange(full, start, end);
    }

    record BuiltSequence(long[] ids, int[] markers) {}
}