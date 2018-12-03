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
            float[] features,
            int frameCount,
            int dimension) {
        decode(
                nativeHandle,
                outputPath.toFile().getAbsolutePath(),
                utteranceId,
                features,
                frameCount,
                dimension);
    }

    private native void decode(
            long nativeHandle,
            String outputPath,
            String utteranceId,
            float[] features,
            int frameCount,
            int dimension);

    private void decodeWithFeatureFile(
            Path outputPath,
            Path featureFile) {
        decodeWithFeatureFile(
                nativeHandle,
                "ark:" + outputPath.toFile().getAbsolutePath(),
                "ark:" + featureFile.toFile().getAbsolutePath());
    }

    private native void decodeWithFeatureFile(
            long nativeHandle,
            String outputPath,
            String featureFile);

    /**
     * Returns a string that contains detailed information on NN model.
     */
    public String modelInfo() {
        return modelInfo(nativeHandle);
    }

    private native String modelInfo(long nativeHandle);

    public static void main(String[] args) {
        Path root = /* Put root here */Paths.get("");
        Path model = root.resolve("final.mdl");
        Path fst = root.resolve("fst/babel-train/HCLG.fst");
        Path symbols = root.resolve("fst/babel-train/words.txt");

        KaldiWrapper wrapper = KaldiWrapper.load(model, fst, symbols);

        Path featurePath = Paths.get("test/data/raw_mfcc_tmp.01.ark");
/*
        Iterable<SpeechData> it = KaldiIO.iterableKaldiFeatureLoader(featurePath.toFile(), 8000);
        SpeechData data = it.iterator().next();
        float[] features = toVector(data.getDataAsMatrix());
        int frameCount = data.vectorCount();
        int dimension = data.get(0).size();
*/

        Path out = Paths.get("foo");
        wrapper.decodeWithFeatureFile(out, featurePath);

        // System.out.println("modelInfo() = " + wrapper.modelInfo());
    }

    private static float[] toVector(float[][] arr2d) {
        int vecCount = arr2d.length;
        int dimension = arr2d[0].length;
        float[] res = new float[vecCount * dimension];
        for (int i = 0; i < vecCount; i++) {
            System.arraycopy(arr2d[i], 0, res, i * dimension, dimension);
        }
        return res;
    }
}
