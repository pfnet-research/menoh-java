# Menoh-Java
Java binding for [Menoh](https://github.com/pfnet-research/menoh/) DNN inference library.

(This library only works on 64-bit architecture at the moment.)

## Requirements
This package depends on the native [Menoh](https://github.com/pfnet-research/menoh/) library. You need to [install](https://github.com/pfnet-research/menoh/blob/master/README.md#installation-using-package-manager-or-binary-packages) it before running.

### Linux
TODO

### Mac OS
Use [pfnet-research/homebrew-menoh](https://github.com/pfnet-research/homebrew-menoh).

```
brew tap pfnet-research/menoh
brew install menoh
```

### Windows
Download the following binaries from the [release page](https://github.com/pfnet-research/menoh/releases) in Menoh repository and place it to the classpath.

- menoh_prebuild_win_v?.?.?.zip

## Examples
Please see [menoh-examples](https://github.com/pfnet-research/menoh-java/tree/master/menoh-examples) directory in this repository.

## Usage
menoh-java provides two types of API: `Model` and `ModelRunner`. `ModelRunner` is a simple wrapper of `Model` and its low-level builder objects. You should choose `ModelRunner` in most cases because it manages their lifecycle on behalf of you.

### `ModelRunner`
`ModelRunner` is a high-level API of menoh-java. What you only need to do is to configure `ModelRunnerBuilder` by using your ONNX model, and `build()` a runner object.

Note that you must `close()` both the runner and its builder explicitly because it manages objects in the native heap which will not be garbage collected by JVM.

```java
try (
    ModelRunnerBuilder builder = ModelRunner
        // Load ONNX model data
        .fromOnnxFile(onnxModelPath)

        // Define input profile (name, dtype, dims) and output profile (name, dtype)
        // dims of output is automatically calculated later
        .addInputProfile(conv11InName, DType.FLOAT, new int[]{batchSize, channelNum, height, width})
        .addOutputProfile(fc6OutName, DType.FLOAT)
        .addOutputProfile(softmaxOutName, DType.FLOAT)

        // Configure backend
        .backendName("mkldnn")
        .backendConfig("");
    ModelRunner runner = builder.build()
) {
    // builder can be deleted explicitly after building a model runner
    builder.close();

    ...
```

Once you create the `ModelRunner`, you can `run()` the model with input data again and again:

```java
    // Run the inference
    runner.run(conv11InName, imageData);

    final Variable softmaxOut = runner.variable(softmaxOutName);
    final ByteBuffer softmaxOutputBuff = softmaxOut.buffer();
    final int[] softmaxDims = softmaxOut.dims();

    // Note: use `get()` instead of `array()` because it is a direct buffer
    final float[] scores = new float[softmaxDims[1]];
    softmaxOutputBuff.asFloatBuffer().get(scores);
```

### `Model`
The low-level API consists of `ModelData`, `VariableProfileTable` and `ModelBuilder` to build a `Model`. You don't need to use them in most cases other than managing lifecycle of the builder objects and the variable buffers by hand.

## Build
```bash
$ mvn package
```

Note that `mvn test` requires that the native Menoh library is available in your system.

## FAQ

### menoh-java fails with `java.lang.UnsatisfiedLinkError`
menoh-java depends on the native Menoh library. You'll get `java.lang.UnsatisfiedLinkError` if it isn't located in [JNA search path](http://java-native-access.github.io/jna/4.5.2/javadoc/com/sun/jna/NativeLibrary.html) at startup even if it exists in the local system.

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

If you fall into this situation, you need to configure the system property (`jna.library.path`) or place the native Menoh in the classpath or the path depend on your platform (`PATH` on Windows, `LD_LIBRARY_PATH` on Linux and `DYLD_LIBRARY_PATH` on OSX). See the [JNA's document](https://github.com/java-native-access/jna/blob/master/www/GettingStarted.md) for more details.

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

## License
The library license is as follows:

- [LICENSE](LICENSE)
