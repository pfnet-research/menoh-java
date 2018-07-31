package jp.preferred.menoh.examples;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import jp.preferred.menoh.*;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.*;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import javax.imageio.ImageIO;

public class Vgg16 {
    public static void main(String[] args) throws Exception {
        System.out.println("Working Directory = " + System.getProperty("user.dir"));

        String conv11InName = "140326425860192";
        String fc6OutName = "140326200777584";
        String softmaxOutName = "140326200803680";

        int batchSize = 1;
        int channelNum = 3;
        int height = 224;
        int width = 224;

        if (args.length <= 0) {
            System.err.println("You must specify a filename of the input image in the argument.");
            System.exit(1);
        }

        final String inputImagePath = args[0];
        final String onnxModelPath = getResourceFilePath("data/VGG16.onnx");
        final String synsetWordsPath = getResourceFilePath("data/synset_words.txt");

        BufferedImage image = ImageIO.read(new File(inputImagePath));

        if (image == null) {
            throw new Exception("Invalid input image path: " + inputImagePath);
        }

        image = cropAndResize(image, width, height);

        float[] imageData = renderToNctw(image);

        try (
                // Load ONNX model data
                ModelData modelData = ModelData.makeFromOnnx(onnxModelPath);

                // Define input profile (name, dtype, dims) and output profile (name, dtype)
                // dims of output is automatically calculated later
                VariableProfileTableBuilder vptBuilder = makeVariableProfileTableBuilder(
                        conv11InName, batchSize, channelNum, height, width, fc6OutName, softmaxOutName);

                // Build VariableProfileTable and get variable dims (if needed)
                VariableProfileTable vpt = vptBuilder.build(modelData);

                // Make ModelBuilder and attach extenal memory buffer
                // Variables which are not attached external memory buffer here are attached
                // internal memory buffers which are automatically allocated
                ModelBuilder modelBuilder = makeModelBuilder(vpt, imageData, conv11InName);

                // Build model and run inference
                Model model = buildAndRunModel(modelBuilder, modelData)
        ) {
            // Get buffer pointer of output
            ByteBuffer fc6OutputBuff = model.variable(fc6OutName).buffer();
            ByteBuffer softmaxOutputBuff = model.variable(softmaxOutName).buffer();

            // Get output
            FloatBuffer fc6OutputFloatBuff = fc6OutputBuff.asFloatBuffer();
            for (int i = 0; i < 10; i++) {
                System.out.print(Float.toString(fc6OutputFloatBuff.get(i)) + " ");
            }
            System.out.println();

            String[] categories = loadCategoryList(synsetWordsPath);
            int topK = 5;

            // Note: softmaxOutputBuff.array() is not available because it is a direct buffer
            int[] softmaxDims = vpt.variableProfile(softmaxOutName).dims();
            float[] scoreArray = new float[softmaxDims[1]];
            softmaxOutputBuff.asFloatBuffer().get(scoreArray);

            int[] topKIndexList = extractTopKIndexList(
                    scoreArray,
                    0,
                    softmaxDims[1],
                    topK);

            System.out.println("top " + topK + "  categories are");

            for (int ki : topKIndexList) {
                System.out.println(ki + " " + scoreArray[ki] + "  categories are" + categories[ki]);
            }
        }
    }


    private static String getResourceFilePath(String name) throws IOException, URISyntaxException {
        // ref. https://stackoverflow.com/a/17870390/1014818
        URL url = Vgg16.class.getClassLoader().getResource(name);
        if (url != null) {
            return Paths.get(url.toURI()).toFile().getCanonicalPath();
        } else {
            throw new FileNotFoundException("The specified resource not found: " + name);
        }
    }

    private static BufferedImage resizeImage(BufferedImage image, int width, int height) {
        BufferedImage destImage = new BufferedImage(width, height, image.getType());
        Graphics2D g = destImage.createGraphics();
        g.drawImage(image, 0, 0, width, height, null);
        g.dispose();
        return destImage;
    }

    private static BufferedImage cropAndResize(BufferedImage image, int width, int height) {
        int shortEdge = Math.min(image.getWidth(), image.getHeight());
        int x0 = (image.getWidth() - shortEdge) / 2;
        int y0 = (image.getHeight() - shortEdge) / 2;
        int width0 = shortEdge;
        int height0 = shortEdge;
        BufferedImage cropped = image.getSubimage(x0, y0, width0, height0);
        BufferedImage resized = resizeImage(cropped, width, height);
        return resized;
    }


    private static float[] renderToNctw(BufferedImage image) {
        int channels = 3;

        float[] data = new float[channels * image.getWidth() * image.getHeight()];

        Raster raster = image.getData();
        int[] temp = new int[channels];

        for (int y = 0; y < image.getHeight(); ++y) {
            for (int x = 0; x < image.getWidth(); ++x) {
                raster.getPixel(x, y, temp);
                data[0 * (image.getHeight() * image.getWidth()) + y * image.getWidth() + x] = temp[2];
                data[1 * (image.getHeight() * image.getWidth()) + y * image.getWidth() + x] = temp[1];
                data[2 * (image.getHeight() * image.getWidth()) + y * image.getWidth() + x] = temp[0];
            }
        }

        return data;
    }

    private static VariableProfileTableBuilder makeVariableProfileTableBuilder(
            String conv11InName,
            int batchSize,
            int channelNum,
            int height,
            int width,
            String fc6OutName,
            String softmaxOutName) throws MenohException {
        VariableProfileTableBuilder vptBuilder = VariableProfileTableBuilder.make();
        vptBuilder.addInputProfile(conv11InName, DType.FLOAT,
                new int[]{batchSize, channelNum, height, width});

        vptBuilder.addOutputProfile(fc6OutName, DType.FLOAT);
        vptBuilder.addOutputProfile(softmaxOutName, DType.FLOAT);

        return vptBuilder;
    }

    private static ModelBuilder makeModelBuilder(
            VariableProfileTable vpt, float[] imageData, String conv11InName) throws MenohException {
        ModelBuilder modelBuilder = ModelBuilder.make(vpt);
        modelBuilder.attach(conv11InName, imageData);

        return modelBuilder;
    }

    public static Model buildAndRunModel(ModelBuilder modelBuilder, ModelData modelData) {
        final Model model = modelBuilder.build(modelData, "mkldnn", "");
        model.run();

        // you can delete modelData explicitly after building a model
        modelData.close();

        return model;
    }

    static class ScoreIndex {
        float score;
        int index;
    }

    private static int[] extractTopKIndexList(float[] floatArray, int first, int last, int k) {
        List<ScoreIndex> q = new ArrayList<>(0);

        for (int i = first; i != last; i++) {
            ScoreIndex sc = new ScoreIndex();
            sc.score = floatArray[i];
            sc.index = i;
            q.add(sc);
        }

        Collections.sort(
                q,
                new Comparator<ScoreIndex>() {
                    @Override
                    public int compare(ScoreIndex obj1, ScoreIndex obj2) {
                        if (obj1.score < obj2.score) {
                            return 1;
                        } else if (obj1.score > obj2.score) {
                            return -1;
                        }
                        return 0;
                    }
                }
        );

        int[] tops = new int[k];

        for (int i = 0; i < k; i++) {
            tops[i] = q.get(i).index;
        }

        return tops;
    }

    private static String[] loadCategoryList(String synsetWordsPath) throws IOException {
        List<String> ret = new ArrayList<>(0);
        FileReader fr = new FileReader(synsetWordsPath);
        BufferedReader br = new BufferedReader(fr);

        String str = br.readLine();
        while (str != null) {
            ret.add(str);
            str = br.readLine();
        }

        br.close();

        return ret.toArray(new String[0]);
    }
}
