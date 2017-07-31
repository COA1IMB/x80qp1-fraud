package com.actico.plattform.ml;

import org.apache.log4j.Logger;

import java.io.*;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;

/**
 * Created by Fabian Cotic on 07.07.2017.
 */
public class Utilities {

    private static Logger log = Logger.getLogger(Utilities.class.getName());

    static String getPropertieValue(String key) {

        Properties prop = new Properties();

        try (InputStream input = new FileInputStream("src\\main\\resources\\runConfiguration.properties")) {
            prop.load(input);
        } catch (IOException e) {
            log.info(e);
        }
        return prop.getProperty(key);
    }

    static double[] getMinValues() {
        List<List<String>> data = getAcquisitionDataAsList();
        int numberOfDataSets = data.size();
        int numberOfColumns = data.get(0).size();
        double[] values = new double[numberOfDataSets];
        double[] mins = new double[numberOfColumns];

        for(int x = 0;x<data.size();x++){
            if(data.get(x).size() != 39){
                System.out.println(" Error " + x);
            }
        }

        for (int j = 0; j < data.get(0).size(); j++) {
            for (int i = 0; i < numberOfDataSets; i++) {
                values[i] = Double.parseDouble(data.get(i).get(j));
            }
            mins[j] = Arrays.stream(values).min().getAsDouble();
        }
        return mins;
    }

    static double[] getMaxValues() {

        List<List<String>> data = getAcquisitionDataAsList();
        int numberOfDataSets = data.size();
        int numberOfColumns = data.get(0).size();
        double[] values = new double[numberOfDataSets];
        double[] maxs = new double[numberOfColumns];

        for (int j = 0; j < data.get(0).size(); j++) {
            for (int i = 0; i < numberOfDataSets; i++) {
                values[i] = Double.parseDouble(data.get(i).get(j));
            }
            maxs[j] = Arrays.stream(values).max().getAsDouble();
        }
        return maxs;
    }

    private static List<List<String>> getAcquisitionDataAsList() {
        ArrayList<List<String>> data = new ArrayList<>();
        String CurrentLine = "";
        String FilePath = Utilities.getPropertieValue("aquisitionAll");

        try (BufferedReader br = new BufferedReader(new FileReader(FilePath))) {

            while ((CurrentLine = br.readLine()) != null) {
                String[] parts1 = CurrentLine.split(",");
                List<String> data2 = Arrays.asList(parts1);
                data.add(data2);
            }
            return data;
        } catch (IOException e) {
            log.warn(e);
        }
        return data;
    }

    public static void mergeTextFiles() {

        long position = 0;
        File performanceFile = new File("src\\main\\resources\\perf07.txt");
        try {
            Files.deleteIfExists(performanceFile.toPath());
            log.info("Deleted: " + performanceFile.toString());
        } catch (IOException e) {
            log.warn(e);
        }

        log.info("Start merging performance files");
        for (int i = 1; i < 5; i++) {
            try (FileChannel in = new RandomAccessFile("src\\main\\resources\\Performance_2007Q" + i + ".txt", "r").getChannel();
                 FileChannel out = new RandomAccessFile("src\\main\\resources\\perf07.txt", "rw").getChannel();) {
                out.transferFrom(in, position, Long.MAX_VALUE);
                position += in.position();
            } catch (IOException e) {
                log.warn(e);
            }
        }
    }
}
