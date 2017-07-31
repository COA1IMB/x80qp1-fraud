package com.actico.plattform.ml;

import org.apache.log4j.Logger;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by Fabian Cotic on 23.05.2017.
 */
public class DataPreprocessor {

    private Logger log = Logger.getLogger(DataPreprocessor.class.getName());

    List<List<String>> getDataAsList() {

        ArrayList<List<String>> data = new ArrayList<>();
        String sCurrentLine = "";
        String learnFilePath = Utilities.getPropertieValue("filePathToStoreLearnData");

        try (BufferedReader br = new BufferedReader(new FileReader(learnFilePath))) {

            while ((sCurrentLine = br.readLine()) != null) {
                String[] parts1 = sCurrentLine.split(",");
                List<String> data2 = Arrays.asList(parts1);
                data.add(data2);
            }
            return normalize(data);
        } catch (IOException e) {
            log.warn(e);
        }
        return normalize(data);
    }

    List<List<String>> getEvalDataAsList() {

        ArrayList<List<String>> data = null;
        String evaluationFilePath = Utilities.getPropertieValue("filePathToStoreEvalData");

        try (BufferedReader br = new BufferedReader(new FileReader(evaluationFilePath));) {
            String sCurrentLine;
            data = new ArrayList<>();

            while ((sCurrentLine = br.readLine()) != null) {
                String[] parts1 = sCurrentLine.split(",");
                List<String> data2 = Arrays.asList(parts1);
                data.add(data2);
            }
            return normalize(data);
        } catch (IOException e) {
            log.warn(e);
        }
        return normalize(data);
    }

    List<List<String>> getRegDataAsList() {

        ArrayList<List<String>> data = null;
        String evaluationFilePath = Utilities.getPropertieValue("filePathToStoreRegData");

        try (BufferedReader br = new BufferedReader(new FileReader(evaluationFilePath));) {
            String sCurrentLine;
            data = new ArrayList<>();

            while ((sCurrentLine = br.readLine()) != null) {
                String[] parts1 = sCurrentLine.split(",");
                List<String> data2 = Arrays.asList(parts1);
                data.add(data2);
            }
            return normalize(data);
        } catch (IOException e) {
            log.warn(e);
        }
        return normalize(data);
    }

    ArrayList<List<List<String>>> splitRegulationAndLearnData(List<List<String>> originalList) {
        ArrayList<List<List<String>>> regulationAndLearnDataList = new ArrayList<>();
        List<List<String>> regulationDataList = new ArrayList<>();
        List<List<String>> learnDataList = new ArrayList<>();
        int size = originalList.size();

        for (int index = 0; index < size; index++) {
            if (index % 10 == 0) {
                regulationDataList.add(originalList.get(index));
            } else {
                learnDataList.add(originalList.get(index));
            }
        }

        regulationAndLearnDataList.add(learnDataList);
        regulationAndLearnDataList.add(regulationDataList);

        return regulationAndLearnDataList;
    }


    List<List<String>> normalize(List<List<String>> data) {
        int numberOfColumns = data.get(0).size();
        double[] mins = Utilities.getMinValues();
        double[] maxs = Utilities.getMaxValues();

        for (int i = 0; i<mins.length; i++) {
            if (mins[i] == maxs[i]) {
                maxs[i] = maxs[i] + 1;
            }
        }

        for (List<String> temp : data) {
            for (int y = 0; y < numberOfColumns - 1; y++) {
                if (mins[y] != maxs[y]) {
                    double tempValue = (Double.parseDouble(temp.get(y)) - mins[y]) / (maxs[y] - mins[y]);

                    if (tempValue == 0.0) {
                        temp.set(y, "0");
                    } else if (tempValue == 1.0) {
                        temp.set(y, "1");
                    } else {
                        temp.set(y, Double.toString(tempValue));
                    }
                } else {
                    log.warn("Error - Min Max Values are equal");
                    break;
                }
            }
        }
        return data;
    }
}
