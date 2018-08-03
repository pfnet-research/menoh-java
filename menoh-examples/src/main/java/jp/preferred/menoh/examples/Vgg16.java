package jp.preferred.menoh.examples;

import com.sun.jna.NativeLibrary;
import jp.preferred.menoh.*;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.*;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.FloatBuffer;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import javax.imageio.ImageIO;

/**
 * <p>Classify an input image by using VGG16, a CNN-based pre-trained model.</p> See the
 * <a href="https://pfnet-research.github.io/menoh/md_tutorial.html">Menoh tutorial</a> for the details.</p>
 */
public class Vgg16 {
    public static void main(String[] args) throws Exception {
        if (args.length <= 0) {
            System.err.println("You must specify a filename of the input image in the argument.");
            System.exit(1);
        }

        System.err.println("Working Directory = " + System.getProperty("user.dir"));
        if (Boolean.getBoolean("jna.debug_load")) {
            NativeLibrary.getProcess(); // a trick to initialize `NativeLibrary` explicitly in this place
            System.err.println("jna.library.path: " + System.getProperty("jna.library.path"));
            System.err.println("jna.platform.library.path: " + System.getProperty("jna.platform.library.path"));
        }

        final String conv11InName = "140326425860192";
        final String fc6OutName = "140326200777584";
        final String softmaxOutName = "140326200803680";

        final int batchSize = 1;
        final int channelNum = 3;
        final int height = 224;
        final int width = 224;

        final String inputImagePath = args[0];
        final String onnxModelPath = getResourceFilePath("data/VGG16.onnx");
        final String synsetWordsPath = getResourceFilePath("data/synset_words.txt");

        // Read and pre-process an input
        final BufferedImage image = ImageIO.read(new File(inputImagePath));
        if (image == null) {
            throw new Exception("Invalid input image path: " + inputImagePath);
        }
        final float[] imageData = renderToNctw(cropAndResize(image, width, height));

        // Build a ModelRunner
        try (
                // Note: You must `close()` the runner and builder to free the native memory explicitly
                ModelRunnerBuilder builder = ModelRunner
                        // Load ONNX model data
                        .fromOnnxFile(onnxModelPath)

                        // Define input profile (name, dtype, dims) and output profile (name, dtype)
                        // Menoh calculates dims of outputs automatically at build time
                        .addInputProfile(conv11InName, DType.FLOAT, new int[] {batchSize, channelNum, height, width})
                        .addOutputProfile(fc6OutName, DType.FLOAT)
                        .addOutputProfile(softmaxOutName, DType.FLOAT)

                        // Configure backend
                        .backendName("mkldnn")
                        .backendConfig("");
                 ModelRunner runner = builder.build()
        ) {
            // The builder can be deleted explicitly after building a model runner
            builder.close();

            // Run the inference
            runner.run(conv11InName, imageData);

            // Get output of `fc6` (the first hidden FC layer after CNNs in VGG16)
            final Variable fc6Out = runner.variable(fc6OutName);

            final FloatBuffer fc6OutFloatBuf = fc6Out.buffer().asFloatBuffer();
            for (int i = 0; i < 10; i++) {
                System.out.print(Float.toString(fc6OutFloatBuf.get(i)) + " ");
            }
            System.out.println();

            // Get output of `softmax` (the output layer in VGG16)
            final Variable softmaxOut = runner.variable(softmaxOutName);

            final int[] softmaxOutDims = softmaxOut.dims();
            final float[] scores = new float[softmaxOutDims[1]];

            // Note: use `get()` instead of `array()` because it is a direct buffer
            softmaxOut.buffer().asFloatBuffer().get(scores);

            final int topK = 5;
            final List<String> categories = loadCategories(synsetWordsPath);

            System.out.println("top " + topK + "  categories are");

            final List<Score> topKIndices = extractTopKIndices(scores, 0, softmaxOutDims[1], topK);
            for (Score s : topKIndices) {
                int ki = s.index;
                System.out.println("index: " + ki + ", score: " + scores[ki] + ", category: " + categories.get(ki));
            }
        }
    }

    private static String getResourceFilePath(String name) throws IOException, URISyntaxException {
        // ref. https://stackoverflow.com/a/17870390/1014818
        final URL url = Vgg16.class.getClassLoader().getResource(name);
        if (url != null) {
            return Paths.get(url.toURI()).toFile().getCanonicalPath();
        } else {
            throw new FileNotFoundException("The specified resource not found: " + name);
        }
    }

    private static BufferedImage resizeImage(BufferedImage image, int width, int height) {
        final BufferedImage destImage = new BufferedImage(width, height, image.getType());
        final Graphics2D g = destImage.createGraphics();
        g.drawImage(image, 0, 0, width, height, null);
        g.dispose();

        return destImage;
    }

    /**
     * Crop and resize an input image to fit the size required by the VGG16 model.
     */
    private static BufferedImage cropAndResize(BufferedImage image, int width, int height) {
        final int shortEdge = Math.min(image.getWidth(), image.getHeight());
        final int x0 = (image.getWidth() - shortEdge) / 2;
        final int y0 = (image.getHeight() - shortEdge) / 2;
        final int width0 = shortEdge;
        final int height0 = shortEdge;

        BufferedImage cropped = image.getSubimage(x0, y0, width0, height0);
        BufferedImage resized = resizeImage(cropped, width, height);

        return resized;
    }

    /**
     * Convert a HWC (Height x Width x Channels) format image into a NCHW (N x Channels x Height x Width) format.
     */
    private static float[] renderToNctw(BufferedImage image) {
        final int height = image.getHeight();
        final int width = image.getWidth();
        final int channelSize = 3;
        final float[] data = new float[channelSize * image.getWidth() * image.getHeight()];

        final Raster raster = image.getData();
        final int[] buf = new int[channelSize];

        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                raster.getPixel(x, y, buf);
                data[0 * (height * width) + y * width + x] = buf[2];
                data[1 * (height * width) + y * width + x] = buf[1];
                data[2 * (height * width) + y * width + x] = buf[0];
            }
        }

        return data;
    }

    static class Score {
        final int index;
        final float score;

        public Score(int index, float score) {
            this.index = index;
            this.score = score;
        }
    }

    private static List<Score> extractTopKIndices(float[] scores, int offset, int length, int k) {
        final List<Score> q = new ArrayList<>();
        for (int i = offset; i < offset + length; i++) {
            q.add(new Score(i, scores[i]));
        }

        Collections.sort(
                q,
                new Comparator<Score>() {
                    @Override
                    public int compare(Score obj1, Score obj2) {
                        if (obj1.score < obj2.score) {
                            return 1;
                        } else if (obj1.score > obj2.score) {
                            return -1;
                        }
                        return 0;
                    }
                }
        );

        return q.subList(0, 5);
    }

    private static List<String> loadCategories(String synsetWordsPath) throws IOException {
        final List<String> ret = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(synsetWordsPath))) {
            String line;
            while ((line = br.readLine()) != null) {
                ret.add(line);
            }
        }

        return ret;
    }
}
