package com.actico.plattform.ml;

import org.apache.log4j.Logger;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.ListStringRecordReader;
import org.datavec.api.split.ListStringSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.List;

/**
 * Created by Fabian Cotic on 22.05.2017.
 */
public class NetworkEvaluater {

    private final double GREEDY_THRESHOLD = 1.0;
    private Logger log = Logger.getLogger(NetworkEvaluater.class.getName());

    public void evaluateNetwork(List<List<String>> dataEval, String neuralNetworkFilePath) {

        MultiLayerNetwork model = null;
        int batchSize = 1;
        int numOutputs = 2;
        DataSetIterator testIter = null;

        try( RecordReader rrTest = new ListStringRecordReader();) {
            rrTest.initialize(new ListStringSplit(dataEval));
            testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 38, 2);
        } catch (Exception e) {
            log.warn(e);
        }

        try {
            model = ModelSerializer.restoreMultiLayerNetwork(neuralNetworkFilePath);
        } catch (Exception e) {
            log.warn(e);
        }

        log.debug("Evaluate model.......");
        Evaluation eval = new Evaluation(numOutputs);

        while (testIter.hasNext()) {
            DataSet t = testIter.next();
            INDArray features = t.getFeatureMatrix();
            INDArray lables = t.getLabels();
            INDArray predicted = model.output(features, false);
            eval.eval(lables, predicted);
            double labelValue = t.getLabels().data().getDouble(1L);
            log.info("Label is: " + labelValue + " Predited was: " + predicted);
        }
        log.info(eval.stats());
    }
}

//            if(labelValue == 1.0){
//                amountDefaults++;
//                if(GREEDY_THRESHOLD >= defaultProbability){
//                    positiveDefault++;
//                    savedMoney = savedMoney + Double.parseDouble(String.valueOf(priceMap.get(lineCounter)));
//                }else{
//                    falseDefault++;
//                }
//            }else if(labelValue == 0.0){
//                amountNonDefaults++;
//                if(GREEDY_THRESHOLD >= defaultProbability){
//                    positiveNonDefault++;
//                }else{
//                    falseNonDefault++;
//                }
//            }
//        HashMap priceMap = generator.getPriceMap();
//        int positiveDefault = 0;
//        int falseDefault = 0;
//        int positiveNonDefault = 0;
//        int falseNonDefault = 0;
//        int amountDefaults = 0;
//        int amountNonDefaults = 0;
//        int lineCounter = 1;
//        double savedMoney = 0;
