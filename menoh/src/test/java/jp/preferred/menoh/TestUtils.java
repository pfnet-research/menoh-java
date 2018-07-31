package jp.preferred.menoh;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.file.Paths;

public class TestUtils {
    /**
     * Convert resource name into file path.
     */
    public static String getResourceFilePath(String name) throws IOException, URISyntaxException {
        // ref. https://stackoverflow.com/a/17870390/1014818
        URL url = TestUtils.class.getClassLoader().getResource(name);
        if (url != null) {
            return Paths.get(url.toURI()).toFile().getCanonicalPath();
        } else {
            throw new FileNotFoundException("The specified resource not found: " + name);
        }
    }

    /**
     * make {@link VariableProfileTableBuilder} for the {@link ModelData} made from `and_op.onnx`.
     */
    public static VariableProfileTableBuilder makeVptBuilderForAndModel(int[] inputDims) {
        return makeVptBuilderForAndModel("input", inputDims, "output");
    }

    /**
     * make {@link VariableProfileTableBuilder} for the {@link ModelData} made from `and_op.onnx`.
     */
    public static VariableProfileTableBuilder makeVptBuilderForAndModel(
            String inputProfileName,
            int[] inputDims,
            String outputProfileName) {
        final VariableProfileTableBuilder builder = VariableProfileTableBuilder.make();
        builder.addInputProfile(inputProfileName, DType.FLOAT, inputDims);
        builder.addOutputProfile(outputProfileName, DType.FLOAT);

        return builder;
    }

    /**
     * make {@link ModelBuilder} and attach the specified input.
     */
    public static ModelBuilder makeModelBuilderWithInput(
            VariableProfileTable vpt, String inputName, ByteBuffer inputData) {
        final ModelBuilder builder = ModelBuilder.make(vpt);
        builder.attach(inputName, inputData);

        return builder;
    }

    /**
     * make {@link ModelBuilder} and attach the specified input.
     */
    public static ModelBuilder makeModelBuilderWithInput(
            VariableProfileTable vpt, String inputName, float[] inputData) {
        final ModelBuilder builder = ModelBuilder.make(vpt);
        builder.attach(inputName, inputData);

        return builder;
    }
}
