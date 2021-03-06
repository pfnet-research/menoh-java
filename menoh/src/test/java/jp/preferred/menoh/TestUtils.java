package jp.preferred.menoh;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
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
}
