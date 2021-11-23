package trainer;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

public class DigitTester2 {
    
    private static Logger log = LoggerFactory.getLogger(DigitTester2.class);
    
    public static void main(String[] args) throws Exception{
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File("mnist-model.zip"));

        int nChannels = 1; // Number of input channels
    	int iterations = 1;
        int outputNum = 10; // number of output classes
        int batchSize = 64; // batch size for each epoch
        int rngSeed = 123; // random number seed for reproducibility
        int numEpochs = 3; // number of epochs to perform

        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum); //create an evaluation object with 10 possible classes
        //while(mnistTest.hasNext()){
            DataSet next = mnistTest.next();
            //INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
            INDArray output = model.output(next.getFeatures()); //get the networks prediction

            eval.eval(next.getLabels(), output); //check the prediction against the true class
        //}
        
        log.info(eval.stats());
        log.info("****************Example finished********************");
        


    }

}
