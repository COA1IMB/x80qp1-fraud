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

        DataPreprocessor prepper = new DataPreprocessor();
        NetworkEvaluater evaluater = new NetworkEvaluater();
        NetworkTrainerL2 trainerl2 = new NetworkTrainerL2();

        //load data
        List<List<String>> learnData = prepper.getDataAsList();
        List<List<String>> regulationData = prepper.getRegDataAsList();
        List<List<String>> evaluationData = prepper.getEvalDataAsList();

        //Start neural network training
        String neuralNetworkFilePath = "src\\NeuralNetwork.zip";
        //trainer.trainNetwork(learnData, regulationData, neuralNetworkFilePath);
        trainerl2.trainNetwork(learnData, regulationData, neuralNetworkFilePath);

        //Evaluate network performance
        evaluater.evaluateNetwork(evaluationData, neuralNetworkFilePath);
        log.info("Finished Execution");
    }
}
