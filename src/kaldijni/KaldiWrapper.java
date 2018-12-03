package kaldijni;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;


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

    public static void main(String[] args) throws IOException {
        Path root = /* Put root here */Paths.get("/media/ahmetaa/depo/projects/kaldi-models/model/babel-no-ivector");
        Path model = root.resolve("final.mdl");
        Path fst = root.resolve("fst/babel-train/HCLG.fst");
        Path symbols = root.resolve("fst/babel-train/words.txt");

        KaldiWrapper wrapper = KaldiWrapper.load(model, fst, symbols);

        Path wav  = Paths.get("test/wav/wav1-8khz.wav");
        Path featurePath = Paths.get("test/mfcc/wav1-8khz.mfcc.ark");

        generateMfcc(wav, featurePath);

        Path out = Paths.get("foo");
        Log.info("Start.");
        wrapper.decodeWithFeatureFile(out, featurePath);
        //wrapper.decodeTest(out, featurePath, model);
        Log.info("End.");

        // System.out.println("modelInfo() = " + wrapper.modelInfo());
    }

    private static SpeechData generateMfcc(Path wav, Path out) throws IOException {
        FloatData allInput = new WavFileChannelReader(wav.toFile()).getAllSamples();
        ShiftedFrameGenerator generator = ShiftedFrameGenerator.forTime(8000, 25, 10, true);
        List<FloatData> frames = generator.generateAllFrames(allInput);
        Preprocessor preprocessor = Preprocessor.KALDI_8KHZ
            .ditherMultiplier(0) // for determinism, we remove dither.
            .windower(WindowFunction.Type.HAMMING)
            .build();
        List<Preprocessor.Result> results = preprocessor.processAllFloat(frames);
        Spectrogram spectrogram = new Spectrogram.Builder(preprocessor).build();

        MelFilterBank filter = MelFilterBank.KALDI_8KHZ
            .filterAmount(40)
            .minimumFrequency(40)
            .maximumFrequency(3800)
            .build();

        MelCepstrum cosineTransform = MelCepstrum
            .builder(40, 40)
            .applyLiftering(22)
            .kaldiStyle(true)
            .build();

        List<FloatData> features = results
            .stream()
            .map(p -> cosineTransform.process(filter.process(spectrogram.process(p.data))))
            .collect(Collectors.toList());
        SpeechData sp = new SpeechData(Segment.fromWavFile(wav,0), features);

        KaldiIO.writeBinaryKaldiFeatures(out, sp);
        return sp;
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
