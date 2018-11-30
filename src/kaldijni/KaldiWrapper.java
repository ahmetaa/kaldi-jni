package kaldijni;

import java.nio.file.Path;
import java.nio.file.Paths;

public class KaldiWrapper {

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

    public static KaldiWrapper load(Path modelFile, Path fstPath, Path wordSymsPath) {
        KaldiWrapper wrapper = new KaldiWrapper();
        wrapper.nativeHandle = wrapper.initialize(
                modelFile.toFile().getAbsolutePath(),
                fstPath.toFile().getAbsolutePath(),
                wordSymsPath.toFile().getAbsolutePath());
        return wrapper;
    }

    // Native methods and public accessors.

    /**
     * Initializes with the given model file.
     *
     * @return native class pointer address as a long number.
     */
    private native long initialize(String modelPath, String fstPath, String wordSymsPath);

    public void decode(
            Path outputPath,
            String utteranceId,
            float[] features) {
        decode(nativeHandle, outputPath.toFile().getAbsolutePath(), utteranceId, features);
    }

    private native void decode(
            long nativeHandle,
            String outputPath,
            String utteranceId,
            float[] features);

    /**
     * Returns a string that contains detailed information on NN model.
     */
    public String modelInfo() {
        return modelInfo(nativeHandle);
    }

    private native String modelInfo(long nativeHandle);

    public static void main(String[] args) {
        Path root = Paths.get("model-root");
        Path model = root.resolve("final.mdl");
        Path fst = root.resolve("fst/babel-train/HCLG.fst");
        Path symbols = root.resolve("fst/babel-train/words.txt");

        KaldiWrapper wrapper = KaldiWrapper.load(model, fst, symbols);

        Path out = Paths.get("foo");
        wrapper.decode(out, "bar", new float[]{});

        // System.out.println("modelInfo() = " + wrapper.modelInfo());
    }
}
