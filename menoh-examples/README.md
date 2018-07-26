# menoh-examples

## Setup
You need to run the following to install `menoh` package before executing examples.

```bash
$ git clone https://github.com/pfnet-research/menoh-java.git
$ cd menoh-java
$ mvn install
```

## Usage

### VGG16
```bash
$ cd menoh-examples
$ mvn compile
$ mkdir data
$ wget -qN -P data/ https://upload.wikimedia.org/wikipedia/commons/5/54/Light_sussex_hen.jpg

$ mvn exec:java -Dexec.mainClass=jp.preferred.menoh.examples.Vgg16 -Dexec.args="data/Light_sussex_hen.jpg"
```
