package kaldijni;

import java.nio.file.Path;
import java.nio.file.Paths;

public class KaldiAcousticScorer {

    static {
        String osName = System.getProperty("os.name");
        if (osName.contains("Windows")) {
            throw new IllegalStateException("Windows is not supported yet");
        } else if (osName.contains("Linux")) {
            System.loadLibrary("kaldi-jni");
        } else {
            throw new IllegalStateException("There is no library for OS = " + osName);
        }
    }

    private long nativeHandle;

    public static KaldiAcousticScorer load(Path modelFile) {
        KaldiAcousticScorer scorer = new KaldiAcousticScorer();
        scorer.nativeHandle = scorer.initialize(modelFile.toFile().getAbsolutePath());
        return scorer;
    }

    // Native methods.

    /**
     * Initializes with the given model file.
     * @param fileName  model file
     * @return native class pointer address as a long number.
     */
    native long initialize(String fileName);

    native int inputDimension();

    native int outputDimension();

    native int ivectorDimension();

    native String modelInfo();

    public static void main(String[] args) {
        Path p = Paths.get("models/final.mdl");
        KaldiAcousticScorer scorer = KaldiAcousticScorer.load(p);
        System.out.println("scorer.modelInfo() = " + scorer.modelInfo());
    }
}
