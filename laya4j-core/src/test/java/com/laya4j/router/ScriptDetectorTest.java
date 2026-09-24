package com.laya4j.router;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ScriptDetectorTest {

    private final ScriptDetector detector = new ScriptDetector();

    @Test
    void detectsEnglish() {
        ScriptDetector.DetectionResult r = detector.detect(
                "Please refund my duplicate order immediately, this is urgent");
        assertEquals("latin", r.dominantScript());
        assertTrue(r.latinFraction() > 0.9);
        assertEquals("en", r.language());
        assertTrue(r.isEnglish());
    }

    @Test
    void detectsChinese() {
        ScriptDetector.DetectionResult r = detector.detect(
                "我昨天被重复扣款两次,要求立刻退款");
        assertEquals("han", r.dominantScript());
        assertEquals("zh", r.language());
        assertFalse(r.isEnglish());
    }

    @Test
    void detectsJapaneseKana() {
        ScriptDetector.DetectionResult r = detector.detect(
                "注文 #1234 の重複請求をすぐに返金してください");
        assertEquals("kana", r.dominantScript());
        assertEquals("ja", r.language());
    }

    @Test
    void detectsKorean() {
        ScriptDetector.DetectionResult r = detector.detect(
                "주문 #1234 중복 청구 환불해 주세요");
        assertEquals("hangul", r.dominantScript());
        assertEquals("ko", r.language());
    }

    @Test
    void detectsSpanish() {
        ScriptDetector.DetectionResult r = detector.detect(
                "Por favor, reembolse mi pedido facturado dos veces");
        assertEquals("latin", r.dominantScript());
        assertEquals("es", r.language());
        assertFalse(r.isEnglish());
    }

    @Test
    void detectsGerman() {
        ScriptDetector.DetectionResult r = detector.detect(
                "Bitte erstatten Sie meine doppelte Bestellung");
        assertEquals("latin", r.dominantScript());
        assertEquals("de", r.language());
        assertFalse(r.isEnglish());
    }

    @Test
    void detectsRussian() {
        ScriptDetector.DetectionResult r = detector.detect(
                "Привет, я требую возврат средств за заказ");
        assertEquals("cyrillic", r.dominantScript());
        assertEquals("ru", r.language());
    }

    @Test
    void detectsArabic() {
        ScriptDetector.DetectionResult r = detector.detect(
                "مرحبا، أريد استرداد طلبي رقم 1234");
        assertEquals("arabic", r.dominantScript());
        assertEquals("ar", r.language());
    }

    @Test
    void emptyStringDefaultsToEnglish() {
        ScriptDetector.DetectionResult r = detector.detect("");
        assertEquals("en", r.language());
        assertTrue(r.isEnglish());
    }

    @Test
    void nullStringDefaultsToEnglish() {
        ScriptDetector.DetectionResult r = detector.detect(null);
        assertEquals("en", r.language());
    }

    @Test
    void fractionsSumToOne() {
        ScriptDetector.DetectionResult r = detector.detect(
                "Hello world 你好世界");
        assertEquals(1.0, r.latinFraction() + r.nonLatinFraction(), 1e-9);
    }
}