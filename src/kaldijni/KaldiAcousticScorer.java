package kaldijni;

import java.io.IOException;
import java.nio.file.Path;

public class KaldiAcousticScorer {

    static {
        try {
            String osName = System.getProperty("os.name");
            if (osName.contains("Windows")) {
                throw new IllegalStateException("Windows is not supported yet");
            } else if (osName.contains("Linux")) {
                NativeUtils.loadLibraryFromJar("/resources/kaldi-suskun.so");
            } else {
                throw new IllegalStateException("There is no library for OS = " + osName);
            }
        } catch (IOException e1) {
            e1.printStackTrace();
        }
    }

    private long nativeHandle;

    public static KaldiAcousticScorer load(Path modelFile) {
        KaldiAcousticScorer scorer = new KaldiAcousticScorer();
        scorer.nativeHandle = scorer.initialize(modelFile.toFile().getAbsolutePath());
        return scorer;
    }

    native long initialize(String fileName);


}
