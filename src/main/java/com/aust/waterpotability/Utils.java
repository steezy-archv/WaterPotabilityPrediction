package com.aust.waterpotability;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Utils {
    // Load dataset from file
    public static Instances loadDataset(String path) throws Exception {
        Instances data = DataSource.read(path);
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    // Print basic info
    public static void printDatasetInfo(Instances data) {
        System.out.println("Dataset has " + data.numInstances() + " instances and " +
                           data.numAttributes() + " attributes.");
        System.out.println("Class attribute: " + data.classAttribute().name());
        System.out.println("Class values: " + data.classAttribute());
    }
}
