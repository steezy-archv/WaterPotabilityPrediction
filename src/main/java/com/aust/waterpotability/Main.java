package com.aust.waterpotability;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import java.io.File;

public class Main {
    public static void main(String[] args) throws Exception {
        String csvPath = "data/water_potability.csv";
        String arffPath = "data/processed_water_potability.arff";

        // Run preprocessing
        Instances processedData = DataPreprocessor.preprocess(csvPath, arffPath);

        // Verify
        System.out.println("Final dataset has " + processedData.numInstances() + " instances and " +
                processedData.numAttributes() + " attributes.");
    }
}

