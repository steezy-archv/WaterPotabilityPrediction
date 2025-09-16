package com.aust.waterpotability;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Random;

public class SupervisedModel {
    public static void main(String[] args) throws Exception {
        Instances data = DataSource.read("data/processed_water_potability.arff");
        data.setClassIndex(data.numAttributes() - 1);

        runClassifier(new NaiveBayes(), data, "Naive Bayes");
        runClassifier(new J48(), data, "Decision Tree (J48)");
        runClassifier(new RandomForest(), data, "Random Forest");
    }

    private static void runClassifier(Classifier model, Instances data, String name) throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(model, data, 10, new Random(1));
        System.out.println("===== " + name + " =====");
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
    }
}
