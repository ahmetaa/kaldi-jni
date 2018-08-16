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
     *
     * @param fileName model file
     * @return native class pointer address as a long number.
     */
    private native long initialize(String fileName);

    /**
     * returns underlying NN model input dimension.
     */
    int inputDimension() {
        return inputDimension(nativeHandle);
    }

    private native int inputDimension(long nativeHandle);

    /**
     * returns underlying NN model output dimension.
     */
    int outputDimension() {
        return outputDimension(nativeHandle);
    }

    private native int outputDimension(long nativeHandle);

    /**
     * returns i-vector dimension.
     */
    int ivectorDimension() {
        return ivectorDimension(nativeHandle);
    }

    native int ivectorDimension(long nativeHandle);

    /**
     * returns minimum required right context amount.
     */
    int rightContext() {
        return rightContext(nativeHandle);
    }

    native int rightContext(long nativeHandle);

    /**
     * returns minimum required left context amount.
     */
    int leftContext() {
        return leftContext(nativeHandle);
    }

    native int leftContext(long nativeHandle);

    /**
     * Returns a string that contains detailed information on NN model.
     */
    String modelInfo() {
        return modelInfo(nativeHandle);
    }

    native String modelInfo(long nativeHandle);

    public static void main(String[] args) {
        Path p = Paths.get("models/final.mdl");
        KaldiAcousticScorer scorer = KaldiAcousticScorer.load(p);

        System.out.println("Input dim      = " + scorer.inputDimension());
        System.out.println("Output dim     = " + scorer.outputDimension());
        System.out.println("Ivector dim    = " + scorer.ivectorDimension());
        System.out.println("Left Context   = " + scorer.leftContext());
        System.out.println("Right Conetext = " + scorer.rightContext());
        System.out.println("scorer.modelInfo() = " + scorer.modelInfo());
    }
}
