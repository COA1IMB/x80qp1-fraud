package com.actico.plattform.ml;

import org.apache.log4j.Logger;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.ListStringRecordReader;
import org.datavec.api.split.ListStringSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.File;
import java.util.List;

import static org.deeplearning4j.nn.conf.GradientNormalization.ClipL2PerParamType;
import static org.deeplearning4j.nn.conf.Updater.*;

/**
 * Created by Fabian Cotic on 22.05.2017.
 */
public class NetworkTrainer {

    private static Logger log = Logger.getLogger(NetworkTrainer.class.getName());

    void trainNetwork(List<List<String>> data, List<List<String>> data2, String neuralNetworkFilePath) {

        int batchSize = Integer.parseInt(Utilities.getPropertieValue("batchSize"));
        DataSetIterator trainIter = null;
        DataSetIterator regIter = null;

        try (RecordReader rr = new ListStringRecordReader();
             RecordReader rr2 = new ListStringRecordReader()) {
            rr.initialize(new ListStringSplit(data));
            rr2.initialize(new ListStringSplit(data2));
            trainIter = new RecordReaderDataSetIterator(rr, batchSize, 38, 2);
            regIter = new RecordReaderDataSetIterator(rr2, batchSize, 38, 2);
        } catch (Exception e) {
            log.warn(e);
        }

        MultiLayerConfiguration conf = getNetworkConfig();

        // Apply Network and attach Listener to Web-UI
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        model.setListeners(new StatsListener(statsStorage));
        uiServer.attach(statsStorage);

        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(new ScoreImprovementEpochTerminationCondition(20))
                //.epochTerminationConditions(new MaxEpochsTerminationCondition(2))
                .scoreCalculator(new DataSetLossCalculator(regIter, true))
                .evaluateEveryNEpochs(1)
                .modelSaver(new LocalFileModelSaver("."))
                .build();

        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, model, trainIter);
        EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();

        //Print out the results:

        log.info("Termination reason: " + result.getTerminationReason());
        log.info("Termination details: " + result.getTerminationDetails());
        log.info("Total epochs: " + result.getTotalEpochs());
        log.info("Best epoch number: " + result.getBestModelEpoch());
        log.info("Score at best epoch: " + result.getBestModelScore());
        log.info("Batchsize was: " + batchSize);

        File locationToSave = new File(neuralNetworkFilePath);
        org.deeplearning4j.nn.api.Model bestModel = result.getBestModel();

        try {
            ModelSerializer.writeModel(bestModel, locationToSave, true);
        } catch (Exception e) {
            log.warn(e);
        }
    }

    private MultiLayerConfiguration getNetworkConfig() {
        return new NeuralNetConfiguration.Builder()
                .seed(1234)
                .gradientNormalization(ClipL2PerParamType)
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .regularization(true)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(38).nOut(1000).updater(ADAGRAD)
                        .gradientNormalizationThreshold(0.1).dropOut(0.8)
                        .weightInit(WeightInit.XAVIER).activation(Activation.RELU).build())
                .layer(1, new DenseLayer.Builder().nIn(1000).nOut(1000).updater(ADAGRAD)
                        .gradientNormalizationThreshold(0.1).dropOut(0.5)
                        .weightInit(WeightInit.XAVIER).activation(Activation.RELU).build())
                .layer(2, new DenseLayer.Builder().nIn(1000).nOut(1000).updater(ADAGRAD)
                        .gradientNormalizationThreshold(0.1).dropOut(0.5)
                        .weightInit(WeightInit.XAVIER).activation(Activation.RELU).build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .nIn(1000).nOut(2).updater(ADAGRAD)
                        .gradientNormalizationThreshold(0.1).dropOut(0.8)
                        .weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX).build()
                )
                .pretrain(true).backprop(true).build();

    }
}
