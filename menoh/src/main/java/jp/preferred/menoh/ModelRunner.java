package jp.preferred.menoh;

import java.nio.ByteBuffer;
import java.util.HashMap;

/**
 * A convenient wrapper for building and running {@link Model}.
 */
public class ModelRunner implements AutoCloseable {
    private final ModelRunnerBuilder builder;

    private final Model model;

    private static final String DEFAULT_BACKEND_NAME = "mkldnn";

    private static final String DEFAULT_BACKEND_CONFIG = "";

    ModelRunner(ModelRunnerBuilder builder, Model model) {
        this.builder = builder;
        this.model = model;
    }

    @Override
    public void close() {
        model.close();
        builder.close();
    }

    /**
     * Returns the underlying {@link Model}.
     */
    public Model model() {
        return this.model;
    }

    public static ModelRunnerBuilder fromOnnxFile(String path) {
        ModelData modelData = null;
        VariableProfileTableBuilder vptBuilder = null;
        try {
            modelData = ModelData.fromOnnxFile(path);
            vptBuilder = VariableProfileTable.builder();

            return new ModelRunnerBuilder(
                    modelData,
                    vptBuilder,
                    DEFAULT_BACKEND_NAME,
                    DEFAULT_BACKEND_CONFIG,
                    new HashMap<String, ByteBuffer>());
        } catch (Throwable t) {
            if (modelData != null) {
                modelData.close();
            }
            if (vptBuilder != null) {
                vptBuilder.close();
            }

            throw t;
        }
    }

    public void run() {
        model.run();
    }
}
