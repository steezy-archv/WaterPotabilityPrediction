package com.aust.waterpotability;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.SerializationHelper;

public class PredictModel {
    public static void main(String[] args) throws Exception {
        // Load new unseen dataset
        Instances newData = DataSource.read("data/new_water_samples.arff");
        newData.setClassIndex(newData.numAttributes() - 1);

        // Load saved ensemble model
        Classifier model = (Classifier) SerializationHelper.read("models/water_quality_voting.model");

        // Predict each instance
        for (int i = 0; i < newData.numInstances(); i++) {
            Instance inst = newData.instance(i);
            double pred = model.classifyInstance(inst);
            System.out.println("Instance " + i + " predicted as: " +
                newData.classAttribute().value((int) pred));
        }
    }
}
