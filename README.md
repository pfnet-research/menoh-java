# menoh-java
*menoh-java* enables you to build your own Deep Neural Network (DNN) application with a few lines of code in Java.

This is a Java binding for [Menoh](https://github.com/pfnet-research/menoh/) DNN inference library, which supports [ONNX](http://onnx.ai/) model format.

## Getting Started

### Add a dependency
Using Gradle:

```groovy
dependencies {
  implementation 'jp.preferred.menoh:menoh:0.1.0'
}
```

Using Maven:

```xml
<dependency>
    <groupId>jp.preferred.menoh</groupId>
    <artifactId>menoh</artifactId>
    <version>0.1.0</version>
</dependency>
```

### Install native libraries
menoh-java requires [Menoh Core](https://github.com/pfnet-research/menoh/) 1.1.1 or later and its dependent native shared libraries to maximize the utilization of hardware resources. You need to [install](https://github.com/pfnet-research/menoh/blob/master/README.md#installation-using-package-manager-or-binary-packages) them to the JVM classpath or the system library path before running.

## Examples
Please see [menoh-examples](menoh-examples) directory in this repository.

## Usage
menoh-java provides two types of APIs: high-level and low-level. The high-level API is a simple wrapper of the low-level API. You should choose the high-level API in most cases because it manages lifecycle of the low-level objects on behalf of you.

### High-level API
`ModelRunner` is a high-level API of menoh-java. What you only need to do is to configure `ModelRunnerBuilder` by using your ONNX model, and `build()` a runner object.

Note that you must `close()` both the runner and its builder objects explicitly because it frees the memory for objects in the native heap which will not be garbage collected by the JVM.

```java
import jp.preferred.menoh.ModelRunner;
import jp.preferred.menoh.ModelRunnerBuilder;

try (
    ModelRunnerBuilder builder = ModelRunner
        // Load ONNX model data
        .fromOnnxFile(onnxModelPath)

        // Define input profile (name, dtype, dims) and output profile (name, dtype)
        // Menoh calculates dims of outputs automatically at build time
        .addInputProfile(conv11InName, DType.FLOAT, new int[] {batchSize, channelNum, height, width})
        .addOutputName(fc6OutName)
        .addOutputName(softmaxOutName)

        // Configure backend
        .backendName("mkldnn")
        .backendConfig("");
    ModelRunner runner = builder.build()
) {
    // The builder can be deleted explicitly after building a model runner
    builder.close();
    ...
```

Once you create the `ModelRunner`, you can `run()` the model with input data again and again:

```java
    // Run the inference
    runner.run(conv11InName, imageData);

    final Variable softmaxOut = runner.variable(softmaxOutName);

    final int[] softmaxOutDims = softmaxOut.dims();
    final ByteBuffer softmaxOutBuf = softmaxOut.buffer();

    // Note: use `get()` instead of `array()` because it is a direct buffer
    final float[] scores = new float[softmaxOutDims[1]];
    softmaxOutBuf.asFloatBuffer().get(scores);
    ...
```

### Low-level API
The low-level API consists of `ModelData`, `VariableProfileTable` and `Model`. You don't need to use them in most cases other than managing lifecycle of the builder objects and the variable buffers by hand.

## Building from Source
```bash
$ git clone https://github.com/pfnet-research/menoh-java.git
$ cd menoh-java
$ mvn package
```

Note that `mvn test` requires that Menoh Core is available in the [JNA search path](http://java-native-access.github.io/jna/4.5.2/javadoc/com/sun/jna/NativeLibrary.html).

## FAQ

### menoh-java fails with `java.lang.UnsatisfiedLinkError`
menoh-java depends on the native Menoh Core library. You'll get `java.lang.UnsatisfiedLinkError` at startup if it isn't located in [JNA search path](http://java-native-access.github.io/jna/4.5.2/javadoc/com/sun/jna/NativeLibrary.html) even if it exists in the local system.

```
java.lang.UnsatisfiedLinkError: Unable to load library 'menoh': Native library (win32-x86-64/menoh.dll) not found in resource path ([file:/C:/workspace/menoh-java/menoh-examples/target/classes/, file:/C:/Users/user/.m2/repository/jp/preferred/menoh/menoh/1.0.0-SNAPSHOT/menoh-1.0.0-SNAPSHOT.jar, file:/C:/Users/user/.m2/repository/net/java/dev/jna/jna/4.5.2/jna-4.5.2.jar])
	at com.sun.jna.NativeLibrary.loadLibrary(NativeLibrary.java:303)
	at com.sun.jna.NativeLibrary.getInstance(NativeLibrary.java:427)
	at com.sun.jna.Library$Handler.<init>(Library.java:179)
	at com.sun.jna.Native.loadLibrary(Native.java:569)
	at com.sun.jna.Native.loadLibrary(Native.java:544)
	at jp.preferred.menoh.MenohNative.<clinit> (MenohNative.java:11)
	...
```

If you fall into this situation, you need to configure the system property (`jna.library.path`) or install the library file to the JVM classpath or the system library path, which depends on your platform (`PATH` on Windows, `LD_LIBRARY_PATH` on Linux and `DYLD_LIBRARY_PATH` on OSX). See the [JNA's document](https://github.com/java-native-access/jna/blob/master/www/GettingStarted.md) for more details.

To inspect the problem, you may set the system property `jna.debug_load=true` to know what is getting wrong:

```
$ mvn exec:java ... -Djna.debug_load=true
Looking in classpath from java.net.URLClassLoader@3e997fa1 for /com/sun/jna/win32-x86-64/jnidispatch.dll
Found library resource at jar:file:/C:/Users/.../.m2/repository/net/java/dev/jna/jna/4.5.2/jna-4.5.2.jar!/com/sun/jna/win32-x86-64/jnidispatch.dll
Looking for library 'menoh'
Adding paths from jna.library.path: null
Trying menoh.dll
Adding system paths: []
Trying menoh.dll
Looking for lib- prefix
Trying libmenoh.dll
Looking in classpath from java.net.URLClassLoader@3e997fa1 for menoh
```

And you can also see `jna.platform.library.path`:

```java
NativeLibrary.getProcess(); // a trick to initialize `NativeLibrary` explicitly in this place
System.err.println("jna.library.path: " + System.getProperty("jna.library.path"));
System.err.println("jna.platform.library.path: " + System.getProperty("jna.platform.library.path"));
```

## Limitation
This library only works on 64-bit architecture at the moment.

## License
menoh-java is released under MIT License. Please see the [LICENSE](LICENSE) file for details.
