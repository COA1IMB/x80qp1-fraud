package com.actico.plattform.ml;

import org.apache.log4j.Logger;
//import org.nd4j.jita.conf.CudaEnvironment;

import java.util.List;

/**
 * Created by Fabian Cotic on 20.05.2017.
 */
public class FannieMaeStart {

    private static Logger log = Logger.getLogger(FannieMaeStart.class.getName());

    public static void main(String[] args) {

        log.info("Start Execution");

        //Merge performance files q1-q4 in a single file
        //Utilities.mergeTextFiles();

//        //extract features and labels to generate  learning & evaluation data and write them to a txt file
//        DataBuilder learnDataBuilder = new DataBuilder();
//        DataBuilder evalDataBuilder = new DataBuilder();
//        log.info("Start Learn Data Extraction");
//        learnDataBuilder.extractLearnData("aquisition2007", "performance2007", "filePathToStoreLearnData");
//        log.info("Start Evaluation Data Extraction");
//        evalDataBuilder.extractLearnData("aquisition2008Q1", "performance2008Q1", "filePathToStoreEvalData");

        DataPreprocessor prepper = new DataPreprocessor();
        NetworkEvaluater evaluater = new NetworkEvaluater();
        NetworkTrainerL2 trainerl2 = new NetworkTrainerL2();

        //load data
        List<List<String>> learnData = prepper.getDataAsList();
        List<List<String>> regulationData = prepper.getRegDataAsList();
        List<List<String>> evaluationData = prepper.getEvalDataAsList();

        //Start neural network training
        String neuralNetworkFilePath = Utilities.getPropertieValue("neuralNetworkFilePath");
        //trainer.trainNetwork(learnData, regulationData, neuralNetworkFilePath);
        trainerl2.trainNetwork(learnData, regulationData, neuralNetworkFilePath);

        //Evaluate network performance
        evaluater.evaluateNetwork(evaluationData, neuralNetworkFilePath);
        log.info("Finished Execution");
    }
}
