package com.actico.plattform.ml;

import org.apache.log4j.Logger;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.ListStringRecordReader;
import org.datavec.api.split.ListStringSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.AutoEncoder;
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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.Collections;
import java.util.List;

import static org.deeplearning4j.nn.conf.GradientNormalization.ClipL2PerParamType;
import static org.deeplearning4j.nn.conf.Updater.ADAGRAD;
import static org.deeplearning4j.nn.conf.Updater.NESTEROVS;

/**
 * Created by fabcot01 on 16.07.2017.
 */
public class NetworkTrainerL2 {

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
            regIter = new RecordReaderDataSetIterator(rr2, 1, 38, 2);
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

        log.info("Start Training");
        int iter = 1;
        int epoche = 1;
        int maxEpoche = 25;
        double currentBestF1 = -1.0;


        while (maxEpoche >= epoche) {
            while (trainIter.hasNext()) {
                model.fit(trainIter.next());
                iter++;
            }
            Evaluation eval = new Evaluation(2);
            while (regIter.hasNext()) {
                DataSet t = regIter.next();
                INDArray features = t.getFeatureMatrix();
                INDArray lables = t.getLabels();
                INDArray predicted = model.output(features, false);
                eval.eval(lables, predicted);
            }
            regIter.reset();
            double f1 = eval.f1(1);
            double recall = eval.recall(1, 0.0);
            double precision = eval.precision(1, 0.0);
            double accuracy = eval.accuracy();
            if (currentBestF1 < f1) {
                currentBestF1 = f1;
                log.info("New Best Model at epoche: " + epoche + " F1-Score: " + f1 + " Recall: " + recall + " Precision: " + precision + " Accuracy: " + accuracy);
                File locationToSave = new File(neuralNetworkFilePath);
                org.deeplearning4j.nn.api.Model model2 = model;
                try {
                    ModelSerializer.writeModel(model2, locationToSave, false);
                } catch (Exception e) {
                    log.warn(e);
                }
            } else {
                log.info("No Progress in epoche: " + epoche + " F1-Score: " + f1 + " Recall: " + recall + " Precision: " + precision + " Accuracy: " + accuracy);
            }
            eval = null;
            trainIter.reset();
            epoche++;
        }
        log.info("Batchsize was: " + batchSize);
    }

    private MultiLayerConfiguration getNetworkConfig() {
        return new NeuralNetConfiguration.Builder()
                .seed(123)
                .gradientNormalization(ClipL2PerParamType)
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .regularization(false)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(38).nOut(1000).updater(ADAGRAD)
                        .gradientNormalizationThreshold(0.1).dropOut(0.9)
                        .weightInit(WeightInit.XAVIER).activation(Activation.RELU).build())
                .layer(1, new DenseLayer.Builder().nIn(1000).nOut(1000).updater(ADAGRAD)
                        .gradientNormalizationThreshold(0.1).dropOut(0.9)
                        .weightInit(WeightInit.XAVIER).activation(Activation.RELU).build())
                .layer(2, new DenseLayer.Builder().nIn(1000).nOut(1000).updater(ADAGRAD)
                        .gradientNormalizationThreshold(0.1).dropOut(0.9)
                        .weightInit(WeightInit.XAVIER).activation(Activation.RELU).build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(1000).nOut(2).updater(ADAGRAD)
                        .gradientNormalizationThreshold(0.1).dropOut(0.9)
                        .weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX).build()
                )
                .pretrain(false).backprop(true).build();
    }

    private MultiLayerConfiguration getNetworkConfigAuto() {
        return new NeuralNetConfiguration.Builder()
                .seed(1234)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .updater(NESTEROVS).momentum(0.5)
                .momentumAfter(Collections.singletonMap(3, 0.9))
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .list()
                .layer(0, new AutoEncoder.Builder().nIn(38 * 63500).nOut(500)
                        .weightInit(WeightInit.XAVIER).lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                        .corruptionLevel(0.3)
                        .build())
                .layer(1, new AutoEncoder.Builder().nIn(500).nOut(250)
                        .weightInit(WeightInit.XAVIER).lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                        .corruptionLevel(0.3)

                        .build())
                .layer(2, new AutoEncoder.Builder().nIn(250).nOut(200)
                        .weightInit(WeightInit.XAVIER).lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                        .corruptionLevel(0.3)
                        .build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation("softmax")
                        .nIn(200).nOut(2).build())
                .pretrain(true).backprop(false)
                .build();
    }
}