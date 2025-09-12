package com.aust.waterpotability;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;

import java.util.Random;

public class ClassifierEvaluation {

    public static void main(String[] args) throws Exception {
        // Load the processed dataset
        DataSource source = new DataSource("data/processed_water_potability.arff");
        Instances dataset = source.getDataSet();

        // Set the class index (last attribute = Potability)
        dataset.setClassIndex(dataset.numAttributes() - 1);

        // List of classifiers we’ll test
        Classifier[] models = {
                new NaiveBayes(),
                new J48(),
                new RandomForest()
        };

        String[] modelNames = {"Naive Bayes", "Decision Tree (J48)", "Random Forest"};

        // Run evaluation for each classifier
        for (int i = 0; i < models.length; i++) {
            System.out.println("===== Evaluating " + modelNames[i] + " =====");

            Evaluation eval = new Evaluation(dataset);
            eval.crossValidateModel(models[i], dataset, 10, new Random(1)); // 10-fold CV

            System.out.println(eval.toSummaryString("\nResults\n======\n", false));
            System.out.println("Confusion Matrix:");
            double[][] cmMatrix = eval.confusionMatrix();
            for (double[] row : cmMatrix) {
                for (double val : row) {
                    System.out.print((int) val + " ");
                }
                System.out.println();
            }
            System.out.println("Accuracy: " + String.format("%.2f", eval.pctCorrect()) + "%");
            System.out.println();
        }
    }
}
