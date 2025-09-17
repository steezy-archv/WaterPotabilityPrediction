package com.aust.waterpotability;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.io.File;

public class Preprocessor {
    public static void main(String[] args) throws Exception {

        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("data/water_potability.csv"));
        Instances data = loader.getDataSet();
        System.out.println("Loaded CSV dataset with " + data.numInstances() + " instances.");

        // Handle missing values
        ReplaceMissingValues rmv = new ReplaceMissingValues();
        rmv.setInputFormat(data);
        data = Filter.useFilter(data, rmv);
        System.out.println("Handled missing values.");

        // Convert last attribute (Potability) to nominal
        NumericToNominal numToNom = new NumericToNominal();
        numToNom.setAttributeIndices("" + (data.numAttributes())); // last attribute
        numToNom.setInputFormat(data);
        data = Filter.useFilter(data, numToNom);
        System.out.println("Converted Potability to nominal.");

        // Balance dataset using SMOTE
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1); 
        }

        SMOTE smote = new SMOTE();
        smote.setInputFormat(data);
        smote.setPercentage(100.0); // oversample minority class by 100%
        data = Filter.useFilter(data, smote);
        
        // Normalize numeric attributes
        Normalize norm = new Normalize();
        norm.setInputFormat(data);
        data = Filter.useFilter(data, norm);
        System.out.println("Normalized numeric attributes.");

        // Save as ARFF
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File("data/processed_water_potability.arff"));
        saver.writeBatch();
        System.out.println("Saved preprocessed dataset to data/processed_water_potability.arff");

        // Print final dataset info
        data.setClassIndex(data.numAttributes() - 1);
        System.out.println("Final dataset has " + data.numInstances() +
                           " instances and " + data.numAttributes() + " attributes.");
        System.out.println("Class attribute: " + data.classAttribute().name() +
                           " → " + data.classAttribute());
    }
}
