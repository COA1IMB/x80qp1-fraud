package com.actico.plattform.ml;

import org.apache.log4j.Logger;

import java.io.*;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 * Created by Fabian Cotic on 22.05.2017.
 */
public class DataBuilder {

    private static Logger log = Logger.getLogger(DataBuilder.class.getName());

    void extractLearnData(String acquisitionKey, String performanceKey, String filePathKey) {

        HashMap idLabelMap = getLabels(performanceKey);
        int counterDefaults = 0;
        int counterNonDefaults = 0;
        String sCurrentLine;
        String filePathToStoreLearnData = Utilities.getPropertieValue(filePathKey);
        String aquisitionFilePath = Utilities.getPropertieValue(acquisitionKey);

        try (PrintWriter writer = new PrintWriter(filePathToStoreLearnData, "UTF-8");
             BufferedReader br = new BufferedReader(new FileReader(aquisitionFilePath))) {

            while ((sCurrentLine = br.readLine()) != null) {
                String[] output = sCurrentLine.split("[|]");

                if(output.length < 2) {
                    continue;
                }

                output[1] = decompositionOfAttributeChannel(output[1]);
                output[13] = decompositionOfAttributeFirstTimeHomeOwner(output[13]);
                output[14] = decompositionOfAttributeLoanPurpose(output[14]);
                output[15] = decompositionOfAttributePropertyType(output[15]);
                output[17] = decompositionOfAttributeOccupancyStatus(output[17]);
                output[18] = decompositionOfAttributeStateCode(output[18]);
                output[19] = decompositionOfAttributeCoBorrower(output[22]);
                output[6] = calculateFirstPayment(output[6], output[7]);
                String label = determineLabel(idLabelMap, output[0]);

                if (label.equals("-1") || output[1].equals("-1") || output[6].equals("-1") || output[13].equals("-1") || output[14].equals("-1") ||
                        output[15].equals("-1") || output[16].equals("-1") || output[17].equals("-1") || output[18].equals("-1")
                        || output[19].equals("-1") || output[1].isEmpty() || output[3].isEmpty() || output[4].isEmpty() ||
                        output[5].isEmpty() || output[6].isEmpty() || output[8].isEmpty() || output[9].isEmpty() ||
                        output[10].isEmpty() || output[11].isEmpty() || output[12].isEmpty() || output[13].isEmpty() ||
                        output[14].isEmpty() || output[15].isEmpty() || output[16].isEmpty() || output[17].isEmpty()) {
                    continue;
                }

                String nextLine;

                switch (label) {
                    case "1":
                        nextLine = output[1] + "," + output[3] + "," + output[4] + ","
                                + output[5] + "," + output[6] + "," + output[8] + "," + output[9] + "," + output[10] + "," + output[11]
                                + "," + output[12] + "," + output[13] + "," + output[14] + "," + output[15] + "," + output[16] + "," + output[17] + "," + output[18] + "," + output[19] + "," + label;
                        counterDefaults++;
                        break;
                    case "0":
                        nextLine = output[1] + "," + output[3] + "," + output[4] + ","
                                + output[5] + "," + output[6] + "," + output[8] + "," + output[9] + "," + output[10] + "," + output[11]
                                + "," + output[12] + "," + output[13] + "," + output[14] + "," + output[15] + "," + output[16] + "," + output[17] + "," + output[18] + "," + output[19] + "," + label;
                        counterNonDefaults++;
                        break;
                    default:
                        continue;
                }
                writer.println(nextLine);
            }
        } catch (IOException e) {
            log.warn(e);
        }
        log.info(counterDefaults + "  " + counterNonDefaults);
    }

    private HashMap getLabels(String performanceKey) {

        String sCurrentLine;
        HashMap map = new HashMap();
        String performanceFilePath = Utilities.getPropertieValue(performanceKey);

        try (BufferedReader br1 = new BufferedReader(new FileReader(performanceFilePath));) {
            while ((sCurrentLine = br1.readLine()) != null) {
                String[] strArr = sCurrentLine.split("|");
                String[] output = new String[30];
                int i = 0;
                StringBuilder builder = new StringBuilder();


                for (String str : strArr) {
                    if (i > 12) {
                        break;
                    }
                    if (str.equals("|")) {
                        output[i] = builder.toString();
                        builder = new StringBuilder();
                        i++;
                    } else {
                        builder.append(str);
                    }
                }
                if (output[4].isEmpty() || output[4] == null) {
                    output[4] = "0";
                }
                //Overwrite until last row with same ID
                map.put(output[0], output[10]);
            }
        } catch (IOException e) {
            log.warn(e);
        }
        return map;
    }

    private String decompositionOfAttributeStateCode(String state) {

        switch (state) {
            case "AL":
                return "1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "AK":
                return "0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "AZ":
                return "0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "AR":
                return "0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "CA":
                return "0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "CO":
                return "0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "CT":
                return "0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "DE":
                return "0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "DC":
                return "0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "FL":
                return "0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "GA":
                return "0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "HI":
                return "0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "ID":
                return "0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "IL":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "IN":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "IA":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "KS":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "KY":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "LA":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "ME":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "MD":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "MA":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "MI":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "MN":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "MS":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "MO":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "MT":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "NE":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "NV":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "NH":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "NJ":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "NM":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "NY":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "NC":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "ND":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "OH":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "OK":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "OR":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "PA":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0";
            case "RI":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0";
            case "SC":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0";
            case "SD":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0";
            case "TN":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0";
            case "TX":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0";
            case "UT":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0";
            case "VT":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0";
            case "VA":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0";
            case "WA":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0";
            case "WV":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0";
            case "WI":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0";
            case "WY":
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0";
            default:
                return "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1";
        }
    }

    private String decompositionOfAttributeCoBorrower(String coScore) {
        if (coScore.equals("") || coScore.isEmpty()) {
            return "0,1";
        }
        return coScore + ",0";
    }

    // R-B-C -> 1-2-3
    private String decompositionOfAttributeChannel(String channel) {
        switch (channel) {
            case "R":
                return "1,0,0";
            case "B":
                return "0,1,0";
            case "C":
                return "0,0,1";
            default:
                return "-1";
        }
    }

    // Calculates difference in months between two dates
    private String calculateFirstPayment(String originationDate, String firstPaymentDate) {
        String pattern = "MM/yyyy";
        SimpleDateFormat format = new SimpleDateFormat(pattern);

        try {
            Date dateOrigination = format.parse(originationDate);
            Date dateFirstPayment = format.parse(firstPaymentDate);

            Calendar startCalendar = new GregorianCalendar();
            startCalendar.setTime(dateOrigination);
            Calendar endCalendar = new GregorianCalendar();
            endCalendar.setTime(dateFirstPayment);

            int diffYear = endCalendar.get(Calendar.YEAR) - startCalendar.get(Calendar.YEAR);
            int diffMonth = diffYear * 12 + endCalendar.get(Calendar.MONTH) - startCalendar.get(Calendar.MONTH);
            return String.valueOf(diffMonth);
        } catch (ParseException e) {
            log.warn(e);
        }
        return "-1";
    }

    //Y-N-U -> 1-2-3
    private String decompositionOfAttributeFirstTimeHomeOwner(String firstTimeHomeOwner) {
        switch (firstTimeHomeOwner) {
            case "Y":
                return "1,0,0";
            case "N":
                return "0,1,0";
            case "U":
                return "0,0,1";
            default:
                return "-1";
        }
    }

    //P-C-R-U -> 1-2-3-4
    private String decompositionOfAttributeLoanPurpose(String loanPurpose) {
        switch (loanPurpose) {
            case "P":
                return "1,0,0,0";
            case "C":
                return "0,1,0,0";
            case "R":
                return "0,0,1,0";
            case "U":
                return "0,0,0,1";
            default:
                return "-1";
        }
    }

    //SF-CO-CP-MH-PU -> 1-2-3-4-5
    private String decompositionOfAttributePropertyType(String propertyType) {
        switch (propertyType) {
            case "SF":
                return "1,0,0,0,0";
            case "CO":
                return "0,1,0,0,0";
            case "CP":
                return "0,0,1,0,0";
            case "MH":
                return "0,0,0,1,0";
            case "PU":
                return "0,0,0,0,1";
            default:
                return "-1";
        }
    }

    //P-S-I-U -> 1-2-3-4
    private String decompositionOfAttributeOccupancyStatus(String occupancyStatus) {
        switch (occupancyStatus) {
            case "P":
                return "1,0,0,0";
            case "S":
                return "0,1,0,0";
            case "I":
                return "0,0,1,0";
            case "U":
                return "0,0,0,1";
            default:
                return "-1";
        }
    }

    private String determineLabel(HashMap idLabelMap, String id) {
        String performance = (String) idLabelMap.get(id);
        if (performance.isEmpty() || performance.equals("") || performance.equalsIgnoreCase("X")) {
            return "-1";
        } else if (Integer.parseInt(performance) >= 3) {
            return "1";
        } else if (Integer.parseInt(performance) == 0) {
            return "0";
        } else {
            return "-1";
        }
    }
}