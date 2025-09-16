package com.aust.waterpotability;

import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.Normalize;

import java.io.File;

public class DataPreprocessor {

    // Convert CSV to ARFF
    public static Instances loadCSV(String csvPath) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(csvPath));
        Instances data = loader.getDataSet();
        return data;
    }

    // Save dataset as ARFF
    public static void saveARFF(Instances data, String arffPath) throws Exception {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(arffPath));
        saver.writeBatch();
    }

    // Handle missing values
    public static Instances handleMissingValues(Instances data) throws Exception {
        ReplaceMissingValues replace = new ReplaceMissingValues();
        replace.setInputFormat(data);
        return Filter.useFilter(data, replace);
    }

    // Normalize numeric attributes
    public static Instances normalizeData(Instances data) throws Exception {
        Normalize normalize = new Normalize();
        normalize.setInputFormat(data);
        return Filter.useFilter(data, normalize);
    }

    public static Instances NumericToNominal(Instances data, String range) throws Exception {
        weka.filters.unsupervised.attribute.NumericToNominal convert = new weka.filters.unsupervised.attribute.NumericToNominal();
        convert.setAttributeIndices(range);
        convert.setInputFormat(data);
        return Filter.useFilter(data, convert);
    }

    // Pipeline: Load CSV -> Handle Missing -> Normalize -> NumericToNominal(class attribute) -> Save ARFF
    public static Instances preprocess(String csvPath, String arffPath) throws Exception {
        Instances data = loadCSV(csvPath);
        System.out.println("Loaded CSV dataset with " + data.numInstances() + " instances.");

        data = handleMissingValues(data);
        System.out.println("Handled missing values.");

        data = normalizeData(data);
        System.out.println("Normalized numeric attributes.");

        data = NumericToNominal(data, "last"); 
        System.out.println("Converted Potability to nominal.");

        saveARFF(data, arffPath);
        System.out.println("Saved preprocessed dataset to " + arffPath);

        return data;
    }
}
