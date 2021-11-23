package trainer;

import org.datavec.image.data.Image;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.ImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Random;

public class DigitTester {
    private static final int rngSeed = 123; // random number seed for reproducibility

    private static Logger log = LoggerFactory.getLogger(DigitTester.class);
    private static BaseImageLoader loader;
    private static MultiLayerNetwork classifer;

    public static MultiLayerNetwork loadNetwork(String resourcePath) throws Exception{
        return MultiLayerNetwork.load(new File(resourcePath),false);
    }

    public static ImageTransform createTransform(){
        ImageTransform imageTransformer = new MultiImageTransform(new Random(rngSeed),
                new ResizeImageTransform(28, 28));
        return imageTransformer;
    }

    public static void testImage(String classPathResource, int expected) throws Exception {
        INDArray input = loader.asMatrix(DigitTester.class.getResourceAsStream(classPathResource)).reshape(1,784);

        INDArray output = classifer.output(input);
        String putStr = output.toString();
        log.info("Prediction: " + classifer.predict(input)[0] + ", Actual: " + expected + ", Percents: " + putStr);
    }

    public static void main(String[] args) throws Exception {
        //classifer = loadNetwork("lenetmnist.zip");
        classifer = loadNetwork("mnist-model.zip");
        loader=new NativeImageLoader(28,28,1,true);
        //loader = new ImageLoader(28, 28, 1, true);

        log.info("Testing 'mnist-model.zip' with 1 & 2");
        //testImage("/number-1.bmp",1);
        testImage("/number-1-reverse.bmp",1);
        testImage("/number-4-reverse.bmp",4);
        ///testImage("/number-2.bmp",2);


        log.info("Testing 'lenetmnist.zip' with 1 & 2");
        classifer = loadNetwork("lenetmnist.zip");
        //testImage("/number-1.bmp",1);
        testImage("/number-1-reverse.bmp",1);
        testImage("/number-4-reverse.bmp",4);
        //testImage("/number-2.bmp",2);


        log.info("Testing 'mnist-model-simple.zip' with 1 & 2");
        classifer = loadNetwork("mnist-model-simple.zip");
        //testImage("/number-1.bmp",1);
        testImage("/number-1-reverse.bmp",1);
        testImage("/number-4-reverse.bmp",4);
        //testImage("/number-2.bmp",2);

    }

    public void ignore() throws Exception{





        Image imageMatrix = loader.asImageMatrix(DigitTester.class.getResourceAsStream("/number-2.bmp"));
        INDArray image = imageMatrix.getImage();

        //https://stackoverflow.com/questions/62050127/how-to-convert-a-jpeg-image-into-a-matrix-rank-2-array-for-a-model-to-predict-u
        INDArray reshapedImage = image.reshape(1, 28,28,1);


        log.info("Image Rank: " + image.rank() + " | " + reshapedImage.rank());

        log.info("Image Shape: " + image.shapeDescriptor() + " | " + reshapedImage.shapeDescriptor());
        log.info("Image Length: " + image.length() + " | " + reshapedImage.length());


        log.info("Evaluate model....");
        INDArray output = classifer.output(reshapedImage);
        String putStr = output.toString();
        log.info("Prediction: " + classifer.predict(reshapedImage)[0] + "\n " + putStr);
/*
        ImagePreProcessingScaler imagePreProcessingScaler = new ImagePreProcessingScaler(0, 1);
        imagePreProcessingScaler.transform(reshapedImage);

        INDArray output = model.output(reshapedImage);
        String putStr = output.toString();
        log.info("Prediction: " + model.predict(reshapedImage)[0] + "\n " + putStr);
*/
        log.info("****************Example finished********************");
    }

}
