# Menoh-Java
Java binding for [Menoh](https://github.com/pfnet-research/menoh/) DNN inference library.

(This library only works on 64-bit architecture at the moment.)

## Requirements
This package depends on the native [Menoh](https://github.com/pfnet-research/menoh/) library. You need to [install](https://github.com/pfnet-research/menoh/blob/master/README.md#installation-using-package-manager-or-binary-packages) it before running.

## Examples
Please see [menoh-examples](https://github.com/pfnet-research/menoh-java/tree/master/menoh-examples) directory in this repository.

## Build
```bash
$ mvn package
```

Note that `mvn test` requires that the native Menoh library is available in your system.

## FAQ

### menoh-java fails with `java.lang.UnsatisfiedLinkError`
menoh-java depends on [Java Native Access (JNA)](https://github.com/java-native-access/jna) to access to the native Menoh library. JNA scans the local system to load the library at startup.

You'll get `java.lang.UnsatisfiedLinkError` if the native Menoh for your platform is not located in [JNA search path](http://java-native-access.github.io/jna/4.5.2/javadoc/com/sun/jna/NativeLibrary.html) even if it exists in your system.

```
java.lang.UnsatisfiedLinkError: Unable to load library 'menoh': Native library (win32-x86-64/menoh.dll) not found in resource path ([file:/C:/workspace/menoh-java/menoh-examples/target/classes/, ...])
    at com.sun.jna.NativeLibrary.loadLibrary (NativeLibrary.java:303)
    at com.sun.jna.NativeLibrary.getInstance (NativeLibrary.java:427)
    at com.sun.jna.Library$Handler.<init> (Library.java:179)
    at com.sun.jna.Native.loadLibrary (Native.java:569)
    at com.sun.jna.Native.loadLibrary (Native.java:544)
    at jp.preferred.menoh.MenohNative.<clinit> (MenohNative.java:11)
    at jp.preferred.menoh.ModelData.makeFromOnnx (ModelData.java:48)
    at jp.preferred.menoh.examples.Vgg16.main (Vgg16.java:54)
    at jdk.internal.reflect.NativeMethodAccessorImpl.invoke0 (Native Method)
    at jdk.internal.reflect.NativeMethodAccessorImpl.invoke (NativeMethodAccessorImpl.java:62)
    at jdk.internal.reflect.DelegatingMethodAccessorImpl.invoke (DelegatingMethodAccessorImpl.java:43)
    at java.lang.reflect.Method.invoke (Method.java:564)
    at org.codehaus.mojo.exec.ExecJavaMojo$1.run (ExecJavaMojo.java:282)
at java.lang.Thread.run (Thread.java:844)
```

If you fall into this situation, you need to configure the system property (`jna.library.path`) or locate the library in the appropriate place (`PATH` on Windows, `LD_LIBRARY_PATH` on Linux and `DYLD_LIBRARY_PATH` on OSX). See the [JNA's document](https://github.com/java-native-access/jna/blob/master/www/GettingStarted.md) for more details.

You may set the system property `jna.debug_load=true` to know what is getting wrong:

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

And you can also check `jna.platform.library.path`:

```java
NativeLibrary.getProcess(); // trick to initialize `NativeLibrary` explicitly in this place
System.err.println("jna.library.path: " + System.getProperty("jna.library.path"));
System.err.println("jna.platform.library.path: " + System.getProperty("jna.platform.library.path"));
```

## License
The library license is as follows:

- [LICENSE](LICENSE)
