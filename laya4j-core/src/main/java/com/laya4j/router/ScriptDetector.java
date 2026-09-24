package com.laya4j.router;

import com.laya4j.core.LayaException;

import java.util.HashMap;
import java.util.Map;

/**
 * Unicode 脚本检测器 - 复刻 laya.lang 的功能
 *
 * Inputs:任意文本
 * Outputs:
 *   - dominantScript: dominant script (han / kana / hangul / latin / cyrillic / arabic)
 *   - fractions: per-script fractions
 *   - language: detected language code (en/de/es/fr/ja/zh/ko/ru/ar/...)
 *   - isEnglish: whether detected as English
 */
public class ScriptDetector {

    public record DetectionResult(
            String dominantScript,
            double latinFraction,
            double nonLatinFraction,
            String language,
            boolean isEnglish
    ) {}

    private static final String[] EN_HINTS = {"the", "is", "are", "please", "my", "i", "you", "we", "this", "that", "have", "has", "been"};
    private static final String[] DE_HINTS = {"der", "die", "das", "ich", "sie", "bitte", "meine", "mein", "und", "ist", "nicht", "haben"};
    private static final String[] ES_HINTS = {"por", "favor", "gracias", "mi", "pero", "que", "está", "buenos", "tiene"};
    private static final String[] FR_HINTS = {"le", "la", "merci", "mon", "ma", "je", "tu", "vous", "nous", "êtes"};
    private static final String[] PT_HINTS = {"por", "favor", "obrigado", "meu", "minha", "está"};
    private static final String[] IT_HINTS = {"il", "la", "grazie", "mio", "mia", "per", "favore"};

    public DetectionResult detect(String text) {
        if (text == null || text.isEmpty()) {
            return new DetectionResult("latin", 1.0, 0.0, "en", true);
        }

        int total = 0;
        int latinCount = 0, hanCount = 0, kanaCount = 0, hangulCount = 0;
        int cyrillicCount = 0, arabicCount = 0;

        for (int i = 0; i < text.length(); ) {
            int cp = text.codePointAt(i);
            int charCount = Character.charCount(cp);
            i += charCount;
            if (Character.isWhitespace(cp) || Character.isDigit(cp) || Character.isISOControl(cp)) continue;
            if (isPunctuation(cp)) continue;
            total++;

            Character.UnicodeScript script = Character.UnicodeScript.of(cp);
            if (script == Character.UnicodeScript.LATIN) latinCount++;
            else if (script == Character.UnicodeScript.HAN) hanCount++;
            else if (script == Character.UnicodeScript.HIRAGANA || script == Character.UnicodeScript.KATAKANA) kanaCount++;
            else if (script == Character.UnicodeScript.HANGUL) hangulCount++;
            else if (script == Character.UnicodeScript.CYRILLIC) cyrillicCount++;
            else if (script == Character.UnicodeScript.ARABIC) arabicCount++;
        }
        if (total == 0) return new DetectionResult("latin", 0, 0, "en", true);

        double latinFrac = (double) latinCount / total;
        double hanFrac = (double) hanCount / total;
        double kanaFrac = (double) kanaCount / total;
        double hangulFrac = (double) hangulCount / total;
        double cyrillicFrac = (double) cyrillicCount / total;
        double arabicFrac = (double) arabicCount / total;

        // 主导脚本
        String dominant = "latin";
        double maxFrac = latinFrac;
        if (hanFrac > maxFrac)      { maxFrac = hanFrac;      dominant = "han"; }
        if (kanaFrac > maxFrac)     { maxFrac = kanaFrac;     dominant = "kana"; }
        if (hangulFrac > maxFrac)   { maxFrac = hangulFrac;   dominant = "hangul"; }
        if (cyrillicFrac > maxFrac) { maxFrac = cyrillicFrac; dominant = "cyrillic"; }
        if (arabicFrac > maxFrac)   { maxFrac = arabicFrac;   dominant = "arabic"; }

        // 路由决策
        String lang = "?";
        boolean isEn = false;
        if (latinFrac >= 0.7) {
            lang = detectLatinLang(text);
            isEn = lang.equals("en");
        } else if (kanaFrac >= 0.3)     lang = "ja";
        else if (hanFrac >= 0.3)      lang = "zh";
        else if (hangulFrac >= 0.3)     lang = "ko";
        else if (cyrillicFrac >= 0.3)   lang = "ru";
        else if (arabicFrac >= 0.3)     lang = "ar";
        else if (lang.equals("?"))      lang = "unknown";

        return new DetectionResult(dominant, latinFrac, 1.0 - latinFrac, lang, isEn);
    }

    /** Latin 文本的语种识别(基于高频词) */
    private String detectLatinLang(String text) {
        String lower = " " + text.toLowerCase() + " ";
        int[] scores = new int[6]; // en, de, es, fr, pt, it
        score(lower, EN_HINTS, scores, 0);
        score(lower, DE_HINTS, scores, 1);
        score(lower, ES_HINTS, scores, 2);
        score(lower, FR_HINTS, scores, 3);
        score(lower, PT_HINTS, scores, 4);
        score(lower, IT_HINTS, scores, 5);
        int best = 0;
        for (int i = 1; i < scores.length; i++) if (scores[i] > scores[best]) best = i;
        String[] langs = {"en", "de", "es", "fr", "pt", "it"};
        return scores[best] > 0 ? langs[best] : "en";
    }

    private void score(String text, String[] words, int[] scores, int idx) {
        for (String w : words) {
            if (text.contains(" " + w + " ") || text.contains(" " + w + ",")
                    || text.contains(" " + w + ".") || text.contains(" " + w + "?")) {
                scores[idx]++;
            }
        }
    }

    private static boolean isPunctuation(int cp) {
        Character.UnicodeScript s = Character.UnicodeScript.of(cp);
        if (s != Character.UnicodeScript.COMMON) return false;
        return cp < 0x80 && !Character.isLetterOrDigit(cp) && !Character.isWhitespace(cp);
    }
}